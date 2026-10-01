"""Rebuild notebook HTML on Python saves, with optional local preview server."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
VERSION = 0
ERROR = ""

# Keep legacy static plots safe when builds run without a GUI.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rl-puzzles-matplotlib"))


def sources():
    snapshot = {}
    for path in ROOT.glob("*.py"):
        try:
            snapshot[path] = path.stat().st_mtime_ns
        except FileNotFoundError:
            # Editors can replace files while we scan them.
            continue
    return snapshot


def report_error(message):
    global ERROR
    ERROR = message[-4000:]
    print(f"Build failed; waiting for the next save.\n{ERROR}", file=sys.stderr, flush=True)


def rebuild():
    global VERSION, ERROR
    try:
        result = subprocess.run([sys.executable, str(ROOT / "build_html.py")], cwd=ROOT,
                                capture_output=True, text=True)
    except OSError as exc:
        report_error(str(exc))
        return False
    if result.returncode == 0:
        VERSION += 1
        ERROR = ""
        print(result.stdout.strip(), flush=True)
        return True
    else:
        report_error(result.stderr or result.stdout or f"Builder exited with code {result.returncode}.")
        return False


def watch(previous=None):
    if previous is None:
        previous = sources()
    while True:
        time.sleep(1)
        try:
            current = sources()
            if current != previous:
                previous = current
                rebuild()
        except Exception as exc:
            report_error(f"Watcher error: {exc}")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BUILD), **kwargs)

    def do_GET(self):
        if self.path == "/__preview_status":
            body = f"{VERSION}\n{ERROR}".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            try:
                page = (BUILD / "index.html").read_text()
            except FileNotFoundError:
                page = "<html><body><p>Waiting for the first successful build. Save a Python file to retry.</p></body></html>"
            reload_script = """<script>
            let version = null;
            setInterval(async () => {
              try {
                const response = await fetch('/__preview_status', {cache: 'no-store'});
                const [next, ...error] = (await response.text()).split('\\n');
                if (version !== null && next !== version) location.reload();
                version = next;
                let notice = document.getElementById('preview-error');
                if (error.join('\\n').trim()) {
                  if (!notice) {
                    notice = document.createElement('pre');
                    notice.id = 'preview-error';
                    notice.style = 'position:fixed;bottom:0;left:0;right:0;max-height:35vh;overflow:auto;margin:0;padding:12px;background:#5b2020;color:white;z-index:9999';
                    document.body.append(notice);
                  }
                  notice.textContent = error.join('\\n');
                } else if (notice) notice.remove();
              } catch (_) {}
            }, 1500);
            </script>"""
            body = page.replace("</body>", reload_script + "</body>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


if __name__ == "__main__":
    initial_sources = sources()
    if "--watch-only" in sys.argv:
        rebuild()
        print("Watching Python files; saving one rebuilds build/index.html. Press Ctrl-C to stop.", flush=True)
        watch(initial_sources)
    else:
        rebuild()
        threading.Thread(target=watch, args=(initial_sources,), daemon=True).start()
        with ThreadingHTTPServer(("127.0.0.1", 8765), Handler) as server:
            print("Preview: http://127.0.0.1:8765", flush=True)
            server.serve_forever()
