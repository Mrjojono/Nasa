#!/bin/bash
url="$1"; tok="$2"
dir="$HOME/donnees_pluie_prevue/04_imerg_final"
f=$(basename "$url")
out="$dir/$f"
[ -s "$out" ] && [ "$(stat -c%s "$out")" -gt 1000000 ] && exit 0
curl -sS -L --fail --http1.1 --retry 8 --retry-delay 10 --retry-all-errors \
     -C - -H "Authorization: Bearer $tok" -o "$out" "$url" \
  && echo "OK $f $(stat -c%s "$out")" || echo "FAIL $f"
