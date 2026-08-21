"""Decoder tests. These prove the loader is correct WITHOUT any MOVER data,
by round-tripping known signals through the documented encoding."""
import base64
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movernotch import io as mio


def encode(ints, ) -> str:
    """Encode int16 samples the way MOVER stores them: LE int16, then base64."""
    return base64.b64encode(np.asarray(ints, dtype="<i2").tobytes()).decode()


def test_roundtrip_positive_and_negative():
    vals = [0, 1, -1, 32767, -32768, 1234, -4321]
    b64 = encode(vals)
    out = mio.decode_wave(b64, gain=1.0, offset=0.0)
    assert np.allclose(out, vals), out


def test_gain_and_offset_applied():
    vals = [100, 200, -300]
    out = mio.decode_wave(encode(vals), gain=0.25, offset=10.0)
    assert np.allclose(out, np.array(vals) * 0.25 + 10.0)


def test_vectorized_matches_official_reference_loop():
    """The official script's integer/sign-bit arithmetic, transcribed, must
    agree with the fast path on the full int16 range including the sign bit."""
    vals = [-32768, -32767, -1, 0, 1, 32766, 32767, 5000, -5000]
    b64 = encode(vals)
    fast = mio.decode_wave(b64, gain=0.25, offset=-3.0)
    ref = mio.decode_wave_reference(b64, gain=0.25, offset=-3.0)
    assert np.allclose(fast, ref), (fast, ref)


def _xml(channel="GE_ART", gain="1.0", offset="0", hz="256", vals=(1, 2, 3)):
    return f"""<root><mg name="{channel}">
      <m name="Offset">{offset}</m><m name="Gain">{gain}</m>
      <m name="Hz">{hz}</m><m name="Points">{len(vals)}</m>
      <m name="Wave">{encode(list(vals))}</m></mg></root>"""


def test_parse_reads_sampling_rate_from_record(tmp_path):
    p = tmp_path / "w.xml"; p.write_text(_xml(hz="500"))
    recs = mio.load_waveform_records(p)
    assert len(recs) == 1
    assert recs[0].fs == 500.0          # never assume 256
    assert recs[0].channel == "GE_ART"


def test_gain_override_toggles(tmp_path):
    p = tmp_path / "w.xml"; p.write_text(_xml(channel="GE_ART", gain="7.0", vals=(4,)))
    with_ov = mio.load_waveform_records(p, apply_gain_overrides=True)[0]
    without = mio.load_waveform_records(p, apply_gain_overrides=False)[0]
    assert with_ov.gain == 0.25          # official override for GE_ART
    assert without.gain == 7.0           # value as written in the XML
    assert with_ov.signal[0] != without.signal[0]


def test_odd_length_payload_is_tolerated():
    raw = np.asarray([1, 2, 3], dtype="<i2").tobytes() + b"\x01"
    out = mio.decode_wave(base64.b64encode(raw).decode(), 1.0, 0.0)
    assert len(out) == 3


def test_measured_rate_overrides_wrong_declared_rate():
    """MOVER IP files declare Hz=180 but run at ~120 Hz. Lock the fix.

    measure_rate must return samples/span, and the loader must prefer it, so a
    wrong declaration can never silently scale every time-domain result again.
    """
    from movernotch.io import measure_rate
    # 1200 packets x 180 points, spanning 1798.6 s of packet timestamps
    assert abs(measure_rate(1200 * 180, 1798.6) - 120.1) < 0.5
    # degenerate inputs fall back rather than dividing by zero
    assert measure_rate(0, 100.0) is None
    assert measure_rate(1000, 0.0) is None


def _synthetic_archive(tmp_path, declared_hz=180, points=180, n_packets=60, dt_s=1.5):
    """Fake MOVER XML where the timestamps disagree with the declared Hz."""
    import base64, datetime as _dt
    rows = []
    t0 = _dt.datetime(2020, 1, 1, 0, 0, 0)
    for i in range(n_packets):
        ts = (t0 + _dt.timedelta(seconds=i * dt_s)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        payload = base64.b64encode(
            (np.arange(points, dtype="<i2") + i).tobytes()).decode()
        rows.append(
            f'<cpc datetime="{ts}"><device><status>UP</status><measurements>'
            f'<mg name="GE_ART"><m name="Wave">{payload}</m>'
            f'<m name="Points">{points}</m><m name="PointsBytes">2</m>'
            f'<m name="Offset">0</m><m name="Gain">0.25</m>'
            f'<m name="Hz">{declared_hz}</m></mg></measurements></device></cpc>')
    xml = '<cpcArchive duration="30">' + "".join(rows) + "</cpcArchive>"
    f = tmp_path / "synthetic_IP.xml"
    f.write_text(xml)
    return f


def test_declared_rate_is_rejected_when_timestamps_disagree(tmp_path):
    """180 points every 1.5 s is 120 Hz, so the loader shouldn't trust the 180."""
    from movernotch.io import load_archive
    f = _synthetic_archive(tmp_path, declared_hz=180, points=180, dt_s=1.5)
    rec = load_archive(f).channels["GE_ART"]
    assert rec.fs_declared == 180.0
    assert abs(rec.fs - 120.0) < 1.0, f"loader returned {rec.fs}, expected ~120"
    assert rec.fs != rec.fs_declared
    assert "measured" in rec.fs_source


def test_duplicate_packets_do_not_inflate_the_measured_rate(tmp_path):
    """Repeated identical payload should be dropped, not concatenated."""
    from movernotch.io import load_archive
    import re
    f = _synthetic_archive(tmp_path, dt_s=1.5, n_packets=40)
    xml = f.read_text()
    first = re.search(r"<cpc .*?</cpc>", xml).group(0)
    f.write_text(xml.replace(first, first + first, 1))
    rec = load_archive(f).channels["GE_ART"]
    assert abs(rec.fs - 120.0) < 1.5, f"duplicate inflated rate to {rec.fs}"


def test_leading_datadown_does_not_halve_the_measured_rate(tmp_path):
    """Span should come from the channel's own packets."""
    from movernotch.io import load_archive
    f = _synthetic_archive(tmp_path, dt_s=1.5, n_packets=40)
    pad = ('<cpc datetime="2019-12-31T23:00:00.000Z"><device><status>DATADOWN'
           '</status><measurements/></device></cpc>')
    f.write_text(f.read_text().replace('<cpcArchive duration="30">',
                                       '<cpcArchive duration="30">' + pad, 1))
    rec = load_archive(f).channels["GE_ART"]
    assert abs(rec.fs - 120.0) < 1.5, f"leading DATADOWN skewed rate to {rec.fs}"
