#!/usr/bin/env bash
# Fetch MOVER files, smallest and most informative first.
#
# Password is read interactively into a 0600 netrc that is deleted on exit, so it
# never lands in shell history or ps.
#
# Usage:
#   bash scripts/fetch_mover.sh metadata   # do this FIRST: tiny files + sizes
#   bash scripts/fetch_mover.sh sizes      # HEAD requests: how big are the v2 files
#   bash scripts/fetch_mover.sh peek  [archive]      # list first entries, cheap
#   bash scripts/fetch_mover.sh sample [archive] [MB] # stream only MB megabytes
#   bash scripts/fetch_mover.sh waves      # full pull, only if you truly need it
#   bash scripts/fetch_mover.sh emr
set -euo pipefail
BASE="https://mover-download.ics.uci.edu"
HOST="mover-download.ics.uci.edu"
OUT="data/raw"
mkdir -p "$OUT"

: "${MOVER_USER:=}"
if [ -z "$MOVER_USER" ]; then read -r -p "MOVER username: " MOVER_USER; fi
read -r -s -p "MOVER password: " MOVER_PASS; echo

NETRC="$(umask 077; mktemp "${TMPDIR:-/tmp}/.mover_netrc.XXXXXX")"
chmod 600 "$NETRC"
cleanup () { rm -f "$NETRC"; }
trap cleanup EXIT INT TERM
printf 'machine %s login %s password %s\n' "$HOST" "$MOVER_USER" "$MOVER_PASS" > "$NETRC"
unset MOVER_PASS

get () {
  echo ">> $1"
  curl -f -S -L -C - --netrc-file "$NETRC" -o "$OUT/$1" "$BASE/$1"
}

case "${1:-metadata}" in
  metadata)
    for f in all_size_listing.txt all_md5sum_listing.txt list.txt README.tar.gz; do get "$f"; done
    mkdir -p "$OUT/readme" && tar xzf "$OUT/README.tar.gz" -C "$OUT/readme" 2>/dev/null || true
    echo; echo "=== SIZES ==="; cat "$OUT/all_size_listing.txt"
    echo; echo "=== README CONTENTS ==="; find "$OUT/readme" -type f | head -40
    ;;
  sizes)
    # v2 archives have no published size; ask the server.
    echo "archive                     size"
    for f in epic_wave_1_v2.tar.gz epic_wave_2_v2.tar.gz epic_wave_3_v2.tar.gz \
             sis_wave_v2.tar.gz; do
      len=$(curl -fsSI -L --netrc-file "$NETRC" "$BASE/$f" \
            | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{print v+0}' | tr -d '\r')
      if [ "${len:-0}" -gt 0 ]; then
        printf '%-28s %s\n' "$f" "$(echo "$len" | awk '{printf "%.1f GB", $1/1073741824}')"
      else
        printf '%-28s %s\n' "$f" "(no content-length)"
      fi
    done
    ;;
  peek)
    # List the first entries and stop; costs a few MB.
    ARCHIVE="${2:-epic_wave_1_v2.tar.gz}"
    echo ">> first 40 entries of $ARCHIVE"
    curl -fsS -L --netrc-file "$NETRC" "$BASE/$ARCHIVE" | tar -tzv 2>/dev/null | head -40 || true
    ;;
  sample)
    # Extract a bounded PREFIX of a huge archive. tar.gz is a stream, so piping
    # through `head` closes the pipe and stops the download early. This is what
    # makes a 119G archive usable for a 20-50 patient study.
    # Bound compressed bytes, not file count: BSD tar writes -v to stderr, so
    # piping stdout to head never stops it and you download the whole archive.
    ARCHIVE="${2:-epic_wave_1_v2.tar.gz}"; MB="${3:-120}"
    mkdir -p "$OUT/sample"
    echo ">> streaming first ${MB} MB (compressed) of $ARCHIVE into $OUT/sample"
    curl -fsS -L --netrc-file "$NETRC" "$BASE/$ARCHIVE" \
      | head -c $(( MB * 1024 * 1024 )) \
      | tar -xz -C "$OUT/sample" 2>/dev/null || true
    echo ">> done (truncated-archive warning above is expected)"
    du -sh "$OUT/sample" 2>/dev/null || true
    find "$OUT/sample" -name '*.xml' | wc -l | xargs echo "   XML files:"
    ;;
  waves)  for f in epic_wave_1_v2.tar.gz; do get "$f"; done ;;   # full pull, corrected release
  emr)    for f in EPIC_EMR.tar.gz; do get "$f"; done ;;
  *) echo "unknown target: $1"; exit 1;;
esac

echo; echo ">> checksum verification"
if command -v md5sum >/dev/null 2>&1; then
  ( cd "$OUT" && md5sum -c --ignore-missing all_md5sum_listing.txt 2>/dev/null || true )
elif command -v md5 >/dev/null 2>&1; then
  ( cd "$OUT" && for f in *.tar.gz; do [ -e "$f" ] && echo "$(md5 -q "$f")  $f"; done )
  echo "   compare the above against all_md5sum_listing.txt"
fi
