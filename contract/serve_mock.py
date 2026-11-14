#!/usr/bin/env python3
"""Serveur API minimal qui sert les fichiers mock du contrat de données.

Endpoints (définis dans contract/mock/index.json) :
  GET /api/meta
  GET /api/frames?layer=<rain_mm|anomaly_mm|anomaly_pct|anomaly_z>
  GET /api/forecast?lead=<1|3>
  GET /api/point?lat=<float>&lon=<float>
  GET /api/alerts

Usage :
    .venv/bin/python contract/serve_mock.py          # port 8000 par défaut
    .venv/bin/python contract/serve_mock.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOCK = HERE / "mock"

# Mapping endpoint -> fichier mock (d'après contract/mock/index.json)
ROUTES = {
    ("GET", "/api/meta"): "meta.json",
    ("GET", "/api/frames"): lambda q: {
        "rain_mm": "frames_rain_mm.json",
        "anomaly_mm": "frames_anomaly_mm.json",
        "anomaly_pct": "frames_anomaly_pct.json",
        "anomaly_z": "frames_anomaly_z.json",
    }.get(q.get("layer", [None])[0]),
    ("GET", "/api/forecast"): lambda q: {
        "1": "forecast_lead1.json",
        "3": "forecast_lead3.json",
    }.get(q.get("lead", [None])[0]),
    ("GET", "/api/point"): lambda q: {
        ("13.5", "2.1"): "point_niamey.json",
        ("12.4", "-1.5"): "point_ouagadougou.json",
        ("14.7", "-17.4"): "point_dakar.json",
    }.get((q.get("lat", [None])[0], q.get("lon", [None])[0])),
    ("GET", "/api/alerts"): "alerts.json",
}


class MockHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Recherche de la route
        target = None
        for (method, route_path), value in ROUTES.items():
            if method == "GET" and route_path == path:
                target = value(query) if callable(value) else value
                break

        if target is None:
            self.send_error(404, f"Endpoint inconnu : {self.path}")
            return

        file_path = MOCK / target
        if not file_path.is_file():
            self.send_error(404, f"Fichier mock absent : {target}")
            return

        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sert les mocks JSON du contrat pluie-prevue")
    ap.add_argument("--port", type=int, default=8000, help="Port d'écoute (défaut 8000)")
    ap.add_argument("--bind", default="127.0.0.1", help="Adresse d'écoute (défaut 127.0.0.1)")
    args = ap.parse_args()

    server = HTTPServer((args.bind, args.port), MockHandler)
    print(f"Serveur mock sur http://{args.bind}:{args.port}")
    print("Endpoints :")
    print("  GET /api/meta")
    print("  GET /api/frames?layer=rain_mm|anomaly_mm|anomaly_pct|anomaly_z")
    print("  GET /api/forecast?lead=1|3")
    print("  GET /api/point?lat=13.5&lon=2.1  (niamey)")
    print("  GET /api/point?lat=12.4&lon=-1.5  (ouagadougou)")
    print("  GET /api/point?lat=14.7&lon=-17.4  (dakar)")
    print("  GET /api/alerts")
    print("Ctrl+C pour arreter.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArret.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
