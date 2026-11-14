#!/bin/bash
# Avancement des téléchargements CHIRPS + IMERG, avec débit et ETA.
#   ./avancement.sh
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
C="$ROOT/00_docs/01_chirps_v3_mensuel"
I="$ROOT/04_imerg_final"

h()  { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1"; }
pct(){ awk -v a="$1" -v b="$2" 'BEGIN{printf "%5.1f%%", 100*a/b}'; }

chirps_ok=$(find "$C" -maxdepth 1 -name 'chirps-*.tif' -size +20M | wc -l)
imerg_ok=$(find "$I" -maxdepth 1 -name '*.HDF5' | wc -l)
chirps_b=$(du -sb --exclude='.*' "$C" 2>/dev/null | cut -f1)
imerg_b=$(du -sb --exclude='.*' "$I" 2>/dev/null | cut -f1)

sleep 5
chirps_b2=$(du -sb --exclude='.*' "$C" 2>/dev/null | cut -f1)
imerg_b2=$(du -sb --exclude='.*' "$I" 2>/dev/null | cut -f1)
rc=$(( (chirps_b2-chirps_b)/5 ))
ri=$(( (imerg_b2-imerg_b)/5 ))
tot_b=$((chirps_b2+imerg_b2))

# Estimation des restes : ~23 Mo/fichier CHIRPS, ~18,5 Mo/granule IMERG
rem=$(( (550-chirps_ok)*23*1024*1024 + (333-imerg_ok)*19*1024*1024 ))
rate=$((rc+ri))

printf 'CHIRPS  %3d/550 fichiers  %s  ' "$chirps_ok" "$(h "$chirps_b2")"
printf '(%s)  %s/s\n' "$(pct "$chirps_ok" 550)" "$(h "$rc")"
printf 'IMERG   %3d/333 granules   %s  ' "$imerg_ok" "$(h "$imerg_b2")"
printf '(%s)  %s/s\n' "$(pct "$imerg_ok" 333)" "$(h "$ri")"
echo
printf 'Total téléchargé : %s\n' "$(h "$tot_b")"
if [ "$rate" -gt 1000 ]; then
  printf 'Débit            : %s/s\n' "$(h "$rate")"
  printf 'Reste / ETA      : %s  ->  %dh%02d\n' "$(h "$rem")" $((rem/rate/3600)) $(( (rem/rate/60)%60 ))
else
  echo 'Débit            : 0 (rien ne progresse ?)'
fi
echo
echo "Dernières lignes de journal :"
tail -n 2 "$C/.download.log" 2>/dev/null | sed 's/^/  CHIRPS  /'
tail -n 2 "$I/.download.log" 2>/dev/null | sed 's/^/  IMERG   /'
