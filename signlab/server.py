"""API de démonstration locale ; catalogue, résultats et inférence d'images."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import warnings
from PIL import Image, UnidentifiedImageError
from .artifacts import load_bundle, predict_image, read_json
from .registry import CATALOG

MAX_UPLOAD = 8 * 1024 * 1024


def create_server(run_dir=None, port=8000):
    run = Path(run_dir) if run_dir else None
    available = {}
    if run:
        for row in read_json(run / "comparison.json"):
            available[row["model"]] = load_bundle(run / row["artifact"])
    web = Path(__file__).parent / "web"
    assets = {"/": ("index.html", "text/html"), "/app.css": ("app.css", "text/css"),
              "/app.js": ("app.js", "text/javascript")}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload, content_type="application/json; charset=utf-8"):
            raw = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            for key, value in {"Content-Type": content_type, "Content-Length": str(len(raw)),
                               "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
                               "Content-Security-Policy": "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'"}.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in assets:
                filename, media_type = assets[path]
                return self._send(200, (web / filename).read_bytes(), f"{media_type}; charset=utf-8")
            if path == "/api/health":
                return self._send(200, {"status": "ok", "trained_models": len(available), "scope": "local demonstration"})
            if path == "/api/catalog":
                return self._send(200, [{"name": n, "family": s.family, "description": s.description,
                                         "candidates": s.candidates, "scale": s.scale, "trained": n in available} for n, s in CATALOG.items()])
            if path == "/api/results":
                return self._send(200, {"comparison": read_json(run / "comparison.json") if run else [],
                                         "selection": read_json(run / "selection.json") if run else None,
                                         "test": read_json(run / "test_results.json") if run and (run / "test_results.json").exists() else None})
            return self._send(404, {"error": "Route inconnue."})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/predict":
                return self._send(404, {"error": "Route inconnue."})
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
                return self._send(403, {"error": "Origine interdite."})
            name = parse_qs(parsed.query).get("model", [""])[0]
            if name not in available:
                return self._send(400, {"error": "Choisissez un modèle entraîné. Ajoutez les images puis exécutez benchmark."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD:
                    return self._send(413, {"error": "Image vide ou supérieure à 8 Mo."})
                if self.headers.get("Content-Type", "").split(";")[0] not in {"image/png", "image/jpeg", "image/webp", "image/bmp", "application/octet-stream"}:
                    return self._send(415, {"error": "Format attendu : PNG, JPEG, WebP ou BMP."})
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Envoi incomplet.")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(BytesIO(raw)) as image:
                        if image.width * image.height > 16_000_000:
                            raise ValueError("Image supérieure à 16 mégapixels.")
                        image.load()
                        result = predict_image(available[name], image)
                return self._send(200, result)
            except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
                return self._send(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(run_dir=None, port=8000):
    server = create_server(run_dir, port)
    print(f"SignLab : http://127.0.0.1:{server.server_port} (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
