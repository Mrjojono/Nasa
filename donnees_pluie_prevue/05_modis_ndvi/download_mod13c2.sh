#!/bin/bash
# MOD13C2 v061 — NDVI composite mensuel 0,05° (2000 -> présent)
#
# Variante sans Python : CMR pour lister les granules, puis curl avec le token
# Earthdata sur l'URL LP DAAC. Utilisable dès que le réseau atteint
# *.cloudfront.net (bloqué depuis la machine de Joan le 14/11/2026).
#
#   EARTHDATA_TOKEN=... ./download_mod13c2.sh            # tout
#   EARTHDATA_TOKEN=... ./download_mod13c2.sh --test     # 1 fichier
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
TOK="${EARTHDATA_TOKEN:-$(cat "$DIR/../00_docs/edl_token.txt" 2>/dev/null)}"
[ -z "${TOK:-}" ] && { echo "token manquant"; exit 1; }

TMP="$(mktemp -d)"
curl -sS "https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name=MOD13C2&version=061&page_size=2000" -o "$TMP/g.json"
python3 - "$TMP/g.json" "$TMP/urls.txt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
urls = []
for it in d["items"]:
    for u in it["umm"].get("RelatedUrls", []):
        if u.get("Type") == "GET DATA" and "lp-prod-protected" in u["URL"]:
            urls.append(u["URL"])
            break
open(sys.argv[2], "w").write("\n".join(sorted(set(urls))) + "\n")
print(f"{len(urls)} granules")
PY

[ "${1:-}" = "--test" ] && { head -1 "$TMP/urls.txt" > "$TMP/t"; mv "$TMP/t" "$TMP/urls.txt"; }

cd "$DIR" || exit 1
while read -r url; do
    f="$(basename "$url")"
    [ -s "$f" ] && continue
    curl -sS -L --fail --http1.1 --retry 8 --retry-delay 10 --retry-all-errors \
         -H "Authorization: Bearer $TOK" -o "$f.part" "$url" \
      && mv "$f.part" "$f" && echo "OK $f" || echo "FAIL $f"
done < "$TMP/urls.txt"
rm -rf "$TMP"
