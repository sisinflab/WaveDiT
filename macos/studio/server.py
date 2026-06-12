"""Local HTTP backend for WaveDiT Studio.

Stdlib-only HTTP layer: serves the static UI from ``studio/ui``, exposes the
JSON API described in ARCHITECTURE.md, and streams server-sent events through
a broadcast Hub. Binds to 127.0.0.1 only. Research tool for synthetic images,
not a medical device.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from studio import paths
from studio.library import library
from studio.settings import settings
from studio.weights import weights

try:
    from studio import __version__ as APP_VERSION
except Exception:  # pragma: no cover - dev tree without __init__ metadata
    APP_VERSION = "0.0.0"

UI_DIR = Path(__file__).resolve().parent / "ui"
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
WEIGHT_FILE_RE = re.compile(r"^[A-Za-z0-9._-]+\.pth$")
MAX_BODY_BYTES = 1024 * 1024
MAX_CLIENT_QUEUE = 256
PING_INTERVAL_S = 15.0

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}

# Set by main.py in window mode: (src_path, suggested_name) -> chosen path or
# None when the user cancels the native save dialog.
export_handler: Callable[[str, str], str | None] | None = None


def _engine() -> Any:
    """Import the engine lazily so server startup does not pay the torch import."""
    from studio.engine import engine

    return engine


def _is_engine_busy(exc: BaseException) -> bool:
    return any(cls.__name__ == "EngineBusy" for cls in type(exc).__mro__)


class ApiError(Exception):
    """JSON API error carrying an HTTP status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class Hub:
    """Broadcast hub: one Queue per SSE client, slow clients are dropped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queues: list[queue.Queue] = []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass

    def publish(self, event: str, data: dict) -> None:
        with self._lock:
            clients = list(self._queues)
        for q in clients:
            if q.qsize() > MAX_CLIENT_QUEUE:
                self.unsubscribe(q)
                try:
                    q.put_nowait(None)  # sentinel: tells the stream loop to stop
                except Exception:
                    pass
                continue
            q.put((event, data))


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type, hub: Hub) -> None:
        super().__init__(address, handler)
        self.hub = hub


class StudioHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "WaveDiTStudio/" + APP_VERSION

    # ---------------------------------------------------------------- logging

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr log
        pass

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/") and path != "/api/events":
            print(f"[studio] {self.command} {path} -> {code}", flush=True)

    # ----------------------------------------------------------------- helpers

    def _send_json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            # Responding before draining the body would desync keep-alive framing.
            self.close_connection = True
            raise ApiError(400, "invalid Content-Length") from exc
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            raise ApiError(400, "request body too large (limit 1 MiB)")
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(400, "invalid JSON body") from exc
        if not isinstance(obj, dict):
            raise ApiError(400, "JSON object expected")
        return obj

    def _handle_error(self, path: str, exc: Exception) -> None:
        if _is_engine_busy(exc):
            self._send_json({"error": str(exc) or "engine is busy"}, 409)
            return
        if isinstance(exc, (ValueError, KeyError)):
            self._send_json({"error": f"bad request: {exc}"}, 400)
            return
        line = traceback.format_exc().strip().splitlines()[-1]
        print(f"[studio] 500 {self.command} {path}: {line}", flush=True)
        self._send_json({"error": str(exc) or type(exc).__name__}, 500)

    # --------------------------------------------------------------------- GET

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/events":
                self._serve_events()
            elif path == "/api/state":
                self._send_json(self._state())
            elif path == "/api/weights":
                refresh = parse_qs(parsed.query).get("refresh", ["0"])[0]
                self._send_json(weights.list(refresh=refresh.lower() in ("1", "true", "yes")))
            elif path == "/api/library":
                self._send_json(library.list())
            elif path.startswith("/api/"):
                raise ApiError(404, f"unknown endpoint: {path}")
            elif path.startswith("/volumes/"):
                self._serve_library_file(
                    path, "/volumes/", ".nii.gz", "application/gzip", library.volume_path
                )
            elif path.startswith("/thumbs/"):
                self._serve_library_file(
                    path, "/thumbs/", ".png", "image/png", library.thumb_path
                )
            else:
                self._serve_static(path)
        except ApiError as err:
            self._send_json({"error": err.message}, err.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._handle_error(path, exc)

    def _state(self) -> dict[str, Any]:
        state = dict(_engine().state())
        cfg = settings.get()
        state.update(
            {
                "version": APP_VERSION,
                "weights": weights.list(),
                "settings": cfg,
                "calibration": cfg.get("calibration", {}),
                "library_count": library.count(),
            }
        )
        return state

    def _serve_library_file(
        self,
        path: str,
        prefix: str,
        suffix: str,
        ctype: str,
        resolver: Callable[[str], Any],
    ) -> None:
        name = path[len(prefix):]
        if not name.endswith(suffix):
            raise ApiError(404, "not found")
        item_id = name[: -len(suffix)]
        if not item_id or not ID_RE.match(item_id):
            raise ApiError(404, "not found")
        fpath = resolver(item_id)
        if fpath is None or not Path(fpath).is_file():
            raise ApiError(404, "not found")
        self._send_file(Path(fpath), ctype, "no-store")

    def _serve_static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        ui_root = UI_DIR.resolve()
        target = (ui_root / rel).resolve()
        if not (target == ui_root or target.is_relative_to(ui_root)):
            raise ApiError(404, "not found")
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            raise ApiError(404, "not found")
        suffix = target.suffix.lower()
        ctype = CONTENT_TYPES.get(suffix, "application/octet-stream")
        in_vendor = "vendor" in target.relative_to(ui_root).parts
        if suffix == ".html":
            cache = "no-store"
        elif suffix in (".js", ".mjs") and in_vendor:
            cache = "max-age=3600"
        else:
            cache = "no-store"
        self._send_file(target, ctype, cache)

    def _send_file(self, fpath: Path, ctype: str, cache: str) -> None:
        # Open (and stat) before sending headers so an OSError can still be framed
        # as a clean 404/500 instead of corrupting an already-started response.
        fh = fpath.open("rb")
        try:
            size = fpath.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            try:
                shutil.copyfileobj(fh, self.wfile, length=256 * 1024)
            except OSError:
                # Headers are out: never append a second response to this socket.
                self.close_connection = True
        finally:
            fh.close()

    def _serve_events(self) -> None:
        hub: Hub = self.server.hub  # type: ignore[attr-defined]
        q = hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                try:
                    item = q.get(timeout=PING_INTERVAL_S)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if item is None:  # dropped as a slow client
                    break
                event, data = item
                payload = f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)

    # -------------------------------------------------------------------- POST

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        path = urlparse(self.path).path
        routes: dict[str, Callable[[dict[str, Any]], Any]] = {
            "/api/generate": self._post_generate,
            "/api/sweep": self._post_sweep,
            "/api/cancel": self._post_cancel,
            "/api/weights/download": self._post_weights_download,
            "/api/weights/delete": self._post_weights_delete,
            "/api/library/delete": self._post_library_delete,
            "/api/export": self._post_export,
            "/api/settings": self._post_settings,
        }
        try:
            func = routes.get(path)
            if func is None:
                # The unread request body would desync keep-alive framing.
                self.close_connection = True
                raise ApiError(404, f"unknown endpoint: {path}")
            self._send_json(func(self._read_json()))
        except ApiError as err:
            self._send_json({"error": err.message}, err.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._handle_error(path, exc)

    def _post_generate(self, body: dict[str, Any]) -> dict[str, Any]:
        hub: Hub = self.server.hub  # type: ignore[attr-defined]
        return {"job_id": _engine().generate(body, hub.publish)}

    def _post_sweep(self, body: dict[str, Any]) -> dict[str, Any]:
        hub: Hub = self.server.hub  # type: ignore[attr-defined]
        return {"job_id": _engine().sweep(body, hub.publish)}

    def _post_cancel(self, body: dict[str, Any]) -> dict[str, Any]:
        job_id = body.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ApiError(400, "missing 'job_id'")
        if not _engine().cancel(job_id):
            raise ApiError(404, f"unknown job: {job_id}")
        return {"ok": True}

    @staticmethod
    def _weight_file(body: dict[str, Any]) -> str:
        file = body.get("file")
        if not isinstance(file, str) or not WEIGHT_FILE_RE.match(file):
            raise ApiError(400, "missing or invalid 'file' (expected <name>.pth)")
        return file

    def _post_weights_download(self, body: dict[str, Any]) -> dict[str, Any]:
        file = self._weight_file(body)
        hub: Hub = self.server.hub  # type: ignore[attr-defined]

        def run() -> None:
            try:
                def progress(p: dict) -> None:
                    hub.publish("weights_progress", {"file": file, **p})

                weights.download(file, progress)
                hub.publish("weights_done", {"file": file})
            except Exception as exc:
                hub.publish("weights_error", {"file": file, "message": str(exc)})

        threading.Thread(target=run, daemon=True, name=f"weights-dl-{file}").start()
        return {"ok": True}

    def _post_weights_delete(self, body: dict[str, Any]) -> dict[str, Any]:
        file = self._weight_file(body)
        if not weights.delete(file):
            raise ApiError(409, f"cannot delete {file} (downloading, in use, or missing)")
        return {"ok": True}

    @staticmethod
    def _item_id(body: dict[str, Any]) -> str:
        item_id = body.get("id")
        if not isinstance(item_id, str) or not ID_RE.match(item_id):
            raise ApiError(400, "missing or invalid 'id'")
        return item_id

    def _post_library_delete(self, body: dict[str, Any]) -> dict[str, Any]:
        item_id = self._item_id(body)
        if not library.delete(item_id):
            raise ApiError(404, f"unknown item: {item_id}")
        return {"ok": True}

    def _post_export(self, body: dict[str, Any]) -> dict[str, Any]:
        item_id = self._item_id(body)
        src = Path(library.volume_path(item_id))
        if not src.is_file():
            raise ApiError(404, f"unknown item: {item_id}")
        suggested = f"{item_id}.nii.gz"
        handler = export_handler  # module attribute, set by main.py in window mode
        if handler is not None:
            chosen = handler(str(src), suggested)
            if chosen is None:
                return {"cancelled": True}
            return {"path": str(chosen)}
        dest = paths.exports_dir() / suggested
        shutil.copyfile(src, dest)
        return {"path": str(dest)}

    def _post_settings(self, body: dict[str, Any]) -> dict[str, Any]:
        return settings.update(body)


def serve_in_thread() -> tuple[ThreadingHTTPServer, int, Hub]:
    """Start the server on 127.0.0.1 in a daemon thread; return (server, port, hub)."""
    hub = Hub()
    try:
        port = int(os.environ.get("WAVEDIT_STUDIO_PORT") or 0)
    except ValueError:
        port = 0
    httpd = StudioHTTPServer(("127.0.0.1", port), StudioHandler, hub)
    real_port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="studio-http")
    thread.start()
    print(f"[studio] serving on http://127.0.0.1:{real_port}/", flush=True)
    return httpd, real_port, hub
