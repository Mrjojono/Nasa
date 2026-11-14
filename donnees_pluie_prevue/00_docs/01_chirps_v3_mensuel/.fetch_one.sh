#!/bin/bash
# $1 = filename chirps-v3.0.YYYY.MM.tif
f="$1"
dir="$(cd "$(dirname "$0")" && pwd)"
out="$dir/$f"
[ -s "$out" ] && [ "$(stat -c%s "$out")" -gt 1000000 ] && exit 0
curl -sS --fail --retry 6 --retry-delay 15 --retry-all-errors \
     -C - -o "$out" \
     "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/$f" \
  && echo "OK $f $(stat -c%s "$out")" || echo "FAIL $f"
