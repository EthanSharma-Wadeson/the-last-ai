"""Stdlib HTTP + SSE server for the live spectator (no heavy web framework)."""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from the_last_ai.spectator.session import SpectatorSession

STATIC_DIR = Path(__file__).resolve().parent / "static"


class SpectatorHandler(BaseHTTPRequestHandler):
    session: SpectatorSession
    server_version = "LastAISpectator/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Quieter logs; still useful for debugging via stderr if needed.
        if "/events" in str(args[0] if args else ""):
            return
        super().log_message(format, *args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            return self._serve_file(STATIC_DIR / "index.html")
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            return self._serve_file(STATIC_DIR / rel)
        if path == "/api/state":
            return self._json(200, self.session.snapshot())
        if path == "/api/world":
            return self._json(200, {"type": "world", **self.session.world_static})
        if path == "/api/observatory":
            return self._json(200, {"paths": self.session.observatory_paths})
        if path == "/events":
            return self._sse()
        if path.startswith("/observatory/"):
            # Serve exported observatory HTML/JSON from output dir
            name = path[len("/observatory/") :]
            base = self.session.output_dir / "observatory"
            target = (base / name).resolve()
            if not str(target).startswith(str(base.resolve())):
                return self._json(403, {"error": "forbidden"})
            if target.exists():
                return self._serve_file(target)
            return self._json(404, {"error": "not found"})

        self._json(404, {"error": "not found", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            body = {}

        if parsed.path == "/api/control":
            action = str(body.get("action", ""))
            snap = self.session.control(action, **{k: v for k, v in body.items() if k != "action"})
            return self._json(200, snap)

        self._json(404, {"error": "not found"})

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return self._json(404, {"error": "file not found", "path": str(path)})
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "application/javascript"
        elif path.suffix == ".css":
            ctype = "text/css"
        elif path.suffix == ".html":
            ctype = "text/html"
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _sse(self) -> None:
        q = self.session.subscribe()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                while q:
                    event = q.popleft()
                    payload = json.dumps(event, ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                if self.session.finished and not q:
                    # keep connection briefly then end
                    time.sleep(0.2)
                    if not q:
                        break
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.session.unsubscribe(q)


def run_spectator_server(
    session: SpectatorSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> ThreadingHTTPServer:
    handler = type(
        "BoundSpectatorHandler",
        (SpectatorHandler,),
        {"session": session},
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Live spectator: {url}")
    print("Python simulation is authoritative. Browser is a renderer only.")
    session.start_background()
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping spectator…")
    finally:
        session.stop()
        if not session.finished:
            session.control("finish")
        httpd.server_close()
    return httpd


def serve_once_for_tests(
    session: SpectatorSession,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ThreadingHTTPServer, str]:
    """Start server on an ephemeral port (tests)."""
    handler = type(
        "BoundSpectatorHandler",
        (SpectatorHandler,),
        {"session": session},
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    host, port = httpd.server_address[:2]
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, url
