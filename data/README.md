# data

nothing committed here, `data/**` is gitignored except this file. MOVER says
don't redistribute.

## access

form at https://mover.ics.uci.edu/download.html, creds come by email. then

    MOVER_USER=<user> bash scripts/fetch_mover.sh metadata
    MOVER_USER=<user> bash scripts/fetch_mover.sh peek epic_wave_1_v2.tar.gz
    MOVER_USER=<user> bash scripts/fetch_mover.sh sample epic_wave_1_v2.tar.gz 120

sis_wave.tar.gz is 119 GB so don't pull whole archives. tar.gz is a stream so
`sample` caps the compressed bytes and bails.

## notes

use `_v2`. the README inside README.tar.gz (2024-05) says the waveforms were
replaced because "some of the wave gains were off".

the published listings (list.txt, all_size_listing, all_md5sum_listing) predate
v2 and don't mention any _v2 file, so the md5 mismatch on README.tar.gz is a stale
listing not corruption (gzip -t passes). v2 archives have no published size or
checksum, and HEAD returns no Content-Length.

format: cpcArchive > cpc > device > measurements > mg. each packet is ~1 s per
channel so you have to concatenate, one mg is not a usable segment. IP file is
~1200 packets, ~30 min at the real ~120 Hz. payload is base64 LE int16,
`raw * gain + offset`. read PointsBytes, don't assume 2. DATADOWN packets carry no
measurements. int16 rails (-32768, -32767, 32767) are missing-data sentinels,
which is why raw min/max looks insane.

their `waveform_decode.py` doesn't run - indentation has U+2002 in it, and the
elif chain for Wave/Hz/Points is nested inside the Gain branch so those wouldn't
parse even after you fix the whitespace.

channels: IP has GE_ART + GE_ECG, both declared 180, actually ~120. CB has ECG1
300, INVP1 100, PLETH 100, and those declarations are fine. no file type has both
ABP and PPG, so pairing means matching IP and CB by case id + timestamp.

gain: official decoder hardcodes an override for GE_ART but on v2 the XML already
says 0.25 so it's a no-op. left it off. didn't check INVP1.

yield is bad. 120 IP files scanned, 2 had a usable art line. channel existing
doesn't mean a transducer was on. it's per-record not per-window, a record is
either mostly fine or nothing.
