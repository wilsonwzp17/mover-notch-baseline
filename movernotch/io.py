"""MOVER waveform decoding."""
from __future__ import annotations
import base64
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"

#: From the official decoder; a no-op for GE_ART on v2 (see module docstring).
GAIN_OVERRIDES: dict[str, float] = {"GE_ART": 0.25, "INVP1": 0.01}

#: mmHg bounds for invasive arterial pressure.
ABP_PLAUSIBLE = (20.0, 300.0)


@dataclass
class WaveRecord:
    channel: str
    signal: np.ndarray      # physical units
    fs: float               # Hz actually used downstream: measured when available
    gain: float
    offset: float
    n_points_declared: int | None = None
    fs_declared: float | None = None   # the XML Hz attribute, wrong on IP files
    fs_measured: float | None = None   # samples / packet-timestamp span
    fs_source: str = "declared"

    def __len__(self) -> int:
        return len(self.signal)


def decode_wave(b64_payload: str, gain: float, offset: float) -> np.ndarray:
    """base64 -> little-endian int16 -> physical units. Vectorized."""
    raw = base64.b64decode(b64_payload)
    if len(raw) % 2:
        raw = raw[:-1]
    ints = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return ints * float(gain) + float(offset)


def decode_wave_reference(b64_payload: str, gain: float, offset: float) -> list[float]:
    """Literal transcription of the official loop, for testing the fast path."""
    wave = base64.b64decode(b64_payload)
    out = []
    for i in range(0, len(wave) - 1, 2):
        t = wave[i] + wave[i + 1] * 256
        t = (t & ~(1 << 15)) + (-32768) * (t >> 15)
        out.append(t * gain + offset)
    return out


def _group_fields(group: ET.Element) -> dict:
    f = {}
    for m in group.iter("m"):
        name = m.attrib.get("name")
        if name:
            f[name] = m.text
    return f


def parse_group(group: ET.Element, apply_gain_overrides: bool = True) -> WaveRecord | None:
    """Decode one measurement group into a WaveRecord."""
    channel = group.get("name") or group.attrib.get("name") or "UNKNOWN"
    f = _group_fields(group)
    if not f.get("Wave"):
        return None
    offset = float(f.get("Offset") or 0.0)
    gain = float(f.get("Gain") or 1.0)
    if apply_gain_overrides and channel in GAIN_OVERRIDES:
        gain = GAIN_OVERRIDES[channel]
    hz = float(f["Hz"]) if f.get("Hz") else float("nan")
    pts = int(f["Points"]) if f.get("Points") else None
    sig = decode_wave(f["Wave"], gain, offset)
    return WaveRecord(channel, sig, hz, gain, offset, pts)


def load_waveform_records(path: str | Path,
                          apply_gain_overrides: bool = True) -> list[WaveRecord]:
    """Parse every measurement group in one file, found by having a 'Wave' child."""
    tree = ET.parse(str(path))
    root = tree.getroot()
    out: list[WaveRecord] = []
    for el in root.iter():
        names = {m.attrib.get("name") for m in el.findall("m")}
        if "Wave" in names:
            rec = parse_group(el, apply_gain_overrides)
            if rec is not None:
                out.append(rec)
    return out


def diagnose_gain_convention(path: str | Path, channel_hint: str = "GE_ART") -> dict:
    """Decode a record both ways and report which gives plausible mmHg."""
    def stats(recs):
        r = [x for x in recs if channel_hint.upper() in x.channel.upper()] or recs
        if not r:
            return None
        s = r[0].signal
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        return {"channel": r[0].channel, "gain": r[0].gain, "min": lo, "max": hi,
                "plausible_abp": ABP_PLAUSIBLE[0] <= lo and hi <= ABP_PLAUSIBLE[1]}

    return {"with_overrides": stats(load_waveform_records(path, True)),
            "xml_gain_only": stats(load_waveform_records(path, False)),
            "note": "Exactly one of these should look like mmHg. Pin it and record the choice."}


def _packet_datetime(cpc) -> "datetime | None":
    """Parse a cpc packet datetime, tolerating a missing or odd fractional part."""
    s = cpc.attrib.get("datetime")
    if not s:
        return None
    s = s.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def measure_rate(n_samples_excl_last: int, span_s: float) -> float | None:
    """Effective sampling rate from sample count and packet-timestamp span.

    N packets span N-1 intervals, so the samples of the final packet are excluded:
    the span covers exactly the first N-1 packets. Returns None when undeterminable.
    """
    if span_s is None or span_s <= 0 or n_samples_excl_last <= 0:
        return None
    return n_samples_excl_last / span_s


#: Relative disagreement between declared and measured rate that is worth flagging.
FS_TOLERANCE = 0.05


#: Missing-data sentinels.
SENTINELS = (-32768, -32767, 32767)


@dataclass
class Archive:
    """One file, assembled."""
    path: str
    duration_attr: str | None
    channels: dict[str, "WaveRecord"]
    packets_total: int
    packets_with_data: int
    packets_datadown: int
    gaps: dict[str, int]          # channel -> count of short packets
    span_s: float | None = None   # packet-timestamp span, first to last
    fs_declared: dict[str, float] | None = None
    fs_measured: dict[str, float] | None = None
    packets_duplicate: int = 0


def _f(text, default=None, cast=float):
    try:
        return cast(text)
    except (TypeError, ValueError):
        return default


def load_archive(path: str | Path, apply_gain_overrides: bool = False,
                 mask_sentinels: bool = True) -> Archive:
    """Assemble a waveform XML into continuous per-channel signals.

    Concatenates packets in file order, drops byte-identical duplicate payloads,
    counts DATADOWN packets without dropping them, and
    masks int16 rails to NaN.
    """
    tree = ET.parse(str(path))
    root = tree.getroot()
    chunks: dict[str, list[np.ndarray]] = {}
    meta: dict[str, dict] = {}
    n_total = n_data = n_down = 0
    times: list[datetime] = []
    ch_times: dict[str, list[datetime]] = {}
    n_dupe = 0
    _last_sig: dict[str, bytes] = {}

    for cpc in root.iter("cpc"):
        n_total += 1
        _dt = _packet_datetime(cpc)
        if _dt is not None:
            times.append(_dt)
        status = "".join(s.text or "" for s in cpc.iter("status")).strip().upper()
        if status and status != "UP":
            n_down += 1
        got = False
        for mg in cpc.iter("mg"):
            channel = mg.get("name") or "UNKNOWN"
            f = _group_fields(mg)
            if not f.get("Wave"):
                continue
            gain = _f(f.get("Gain"), 1.0)
            if apply_gain_overrides and channel in GAIN_OVERRIDES:
                gain = GAIN_OVERRIDES[channel]
            offset = _f(f.get("Offset"), 0.0)
            nbytes = _f(f.get("PointsBytes"), 2, int)
            if nbytes != 2:
                raise NotImplementedError(
                    f"{path}: channel {channel} uses PointsBytes={nbytes}; "
                    "only 16-bit samples are implemented")
            _payload = f["Wave"]
            if _last_sig.get(channel) == _payload:
                # Byte-identical consecutive payload: a duplicated packet. Counting
                # it would inflate the sample count and therefore the measured rate.
                n_dupe += 1
                continue
            _last_sig[channel] = _payload
            if _dt is not None:
                ch_times.setdefault(channel, []).append(_dt)
            raw = np.frombuffer(base64.b64decode(_payload), dtype="<i2")
            vals = raw.astype(np.float64)
            if mask_sentinels:
                vals[np.isin(raw, SENTINELS)] = np.nan
            chunks.setdefault(channel, []).append(vals * gain + offset)
            meta.setdefault(channel, {
                "fs": _f(f.get("Hz")), "gain": gain, "offset": offset,
                "points": _f(f.get("Points"), None, int),
                "min_scale": _f(f.get("Min")), "max_scale": _f(f.get("Max"))})
            got = True
        if got:
            n_data += 1

    span_s = (max(times) - min(times)).total_seconds() if len(times) > 1 else None

    channels, gaps = {}, {}
    fs_declared: dict[str, float] = {}
    fs_measured: dict[str, float] = {}
    for ch, parts in chunks.items():
        m = meta[ch]
        sig = np.concatenate(parts)
        expected = m.get("points")
        gaps[ch] = sum(1 for p in parts if expected and len(p) != expected)

        declared = m["fs"] if m["fs"] else float("nan")
        # Span must come from the packets carrying THIS channel, not from all
        # packets: a leading/trailing DATADOWN run or a channel dropout otherwise
        # stretches the denominator and silently halves the measured rate.
        cts = sorted(ch_times.get(ch, []))
        ch_span = (cts[-1] - cts[0]).total_seconds() if len(cts) > 1 else None
        # The span covers the first N-1 packets, so exclude the last packet's samples.
        n_excl_last = int(sum(len(p) for p in parts) - len(parts[-1])) if parts else 0
        measured = measure_rate(n_excl_last, ch_span) if len(parts) > 1 else None

        if measured is not None and np.isfinite(measured) and measured > 0:
            fs_used, source = float(measured), "measured"
            if np.isfinite(declared) and declared > 0:
                rel = abs(measured - declared) / declared
                if rel > FS_TOLERANCE:
                    source = "measured_disagrees_with_declared"
        else:
            fs_used, source = float(declared), "declared"

        fs_declared[ch] = float(declared)
        if measured is not None:
            fs_measured[ch] = float(measured)

        channels[ch] = WaveRecord(ch, sig, fs_used, m["gain"], m["offset"], expected,
                                  fs_declared=float(declared),
                                  fs_measured=(float(measured) if measured else None),
                                  fs_source=source)
    return Archive(str(path), root.get("duration"), channels,
                   n_total, n_data, n_down, gaps,
                   span_s=span_s, fs_declared=fs_declared, fs_measured=fs_measured,
                   packets_duplicate=n_dupe)


def iter_windows(sig: np.ndarray, fs: float, window_s: float = 4.0,
                 stride_s: float | None = None):
    """4 s windows, matching the paper."""
    n = int(window_s * fs)
    step = int((stride_s or window_s) * fs)
    for start in range(0, max(0, len(sig) - n + 1), step):
        yield start, sig[start:start + n]
