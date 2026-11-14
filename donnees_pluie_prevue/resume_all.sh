#!/bin/bash
# Relance les téléchargements en cours. Idempotent : les fichiers déjà
# complets sont ignorés, les partiels sont repris (curl -C -).
#
#   ./resume_all.sh
#
# Journaux : 01_chirps_v3_mensuel/.download.log, 04_imerg_final/.download.log
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
TOK="$(cat "$ROOT/00_docs/edl_token.txt")"

# --- CHIRPS (2 connexions max : le serveur renvoie 429 au-delà) ---
cd "$ROOT/00_docs/01_chirps_v3_mensuel" || exit 1
nohup setsid bash -c 'cat .filelist.txt | xargs -P 2 -n 1 ./.fetch_one.sh' \
  >> .download.log 2>&1 < /dev/null &
echo "CHIRPS relancé (pid $!)"

# --- IMERG (3 connexions) ---
cd "$ROOT/04_imerg_final" || exit 1
export TOK
nohup setsid bash -c 'cat .urls.txt | xargs -P 3 -I{} ./.fetch_one.sh "{}" "$TOK"' \
  >> .download.log 2>&1 < /dev/null &
echo "IMERG relancé (pid $!)"

sleep 2
echo
echo "CHIRPS : $(ls "$ROOT"/01_chirps_v3_mensuel/*.tif 2>/dev/null | wc -l)/550"
echo "IMERG  : $(ls "$ROOT"/04_imerg_final/*.HDF5 2>/dev/null | wc -l)/333"
