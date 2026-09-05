"""Serve the MedLens website and provide API endpoints for medical report extraction."""

import base64
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from extractor import (
    _GOOGLE_GENAI_AVAILABLE,
    extract_medical_report,
    get_demo_report_data,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8765  # Using 8765 to avoid port 8000 collisions with Windows services or zombie processes


class MedLensHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MedLens AI Insight."""

    protocol_version = "HTTP/1.0"

    def do_GET(self):
        """Serve static files and health checks."""
        self.close_connection = True
        try:
            clean_path = self.path.split("?")[0]
            print(f"[HTTP GET] {clean_path} from {self.client_address[0]}")
            sys.stdout.flush()

            # 1. API Health Check
            if clean_path == "/api/health":
                self._send_json(
                    {
                        "status": "healthy",
                        "google_genai_available": _GOOGLE_GENAI_AVAILABLE,
                        "service": "MedLens AI Insight",
                    },
                    status=200,
                )
                return

            # 2. Static File Resolution
            if clean_path in ("", "/"):
                target_file = ROOT / "index.html"
            else:
                relative_path = clean_path.lstrip("/\\")
                target_file = (ROOT / relative_path).resolve()

            # Security check
            try:
                target_file.relative_to(ROOT)
            except ValueError:
                self._send_text("403 Forbidden: Access denied.", status=403)
                return

            if not target_file.is_file():
                self._send_text(f"404 Not Found: '{clean_path}' does not exist.", status=404)
                return

            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(target_file))
            if not mime_type:
                mime_type = "application/octet-stream"
            if mime_type.startswith("text/") or mime_type in ("application/json", "application/javascript"):
                mime_type += "; charset=utf-8"

            content_bytes = target_file.read_bytes()

            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content_bytes)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()

            self.wfile.write(content_bytes)
            self.wfile.flush()

        except Exception as exc:
            self._send_json({"error": f"Server error: {exc}"}, status=500)

    def do_POST(self):
        """Handle report processing and demo requests."""
        self.close_connection = True
        clean_path = self.path.split("?")[0]
        print(f"[HTTP POST] {clean_path} from {self.client_address[0]}")
        sys.stdout.flush()

        if clean_path == "/api/process-report":
            self._handle_process_report()
        else:
            self._send_json({"error": f"Endpoint '{clean_path}' not found."}, status=404)

    def _handle_process_report(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            content_length = 0

        if content_length == 0:
            self._send_json({"success": False, "error": "Request body cannot be empty."}, status=400)
            return

        try:
            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body.decode("utf-8"))
            user_intake = body.get("user_intake", {})

            # 1. Handle Demo Mode
            if body.get("is_demo"):
                demo_data = get_demo_report_data()
                response_payload = {
                    "success": True,
                    "is_demo": True,
                    "user_provided": {
                        "source": "User-Provided",
                        "name": user_intake.get("name", "").strip() or "Eleanor Vance (Demo Patient)",
                        "age": user_intake.get("age", "").strip() or "46",
                        "sex": user_intake.get("sex", "").strip() or "Female",
                        "symptoms": (
                            user_intake.get("symptoms", "").strip()
                            or "Fatigue, mild exertion dyspnea, and dizziness for 3 weeks."
                        ),
                        "allergies": user_intake.get("allergies", "").strip() or "Penicillin",
                    },
                    "ai_extracted": demo_data.model_dump(),
                }
                self._send_json(response_payload, status=200)
                return

            # 2. Check for missing google-genai library
            if not _GOOGLE_GENAI_AVAILABLE:
                self._send_json(
                    {
                        "success": False,
                        "error": (
                            "The 'google-genai' library is not yet installed in this Python environment.\n"
                            "Please run: pip install -r requirements.txt in your terminal to enable live Gemini extraction.\n"
                            "(Tip: Click 'Load Sample Lab Report (Demo)' to test the interface right away!)"
                        ),
                    },
                    status=400,
                )
                return

            # 3. Validate uploaded file
            file_base64 = body.get("file_base64")
            if not file_base64:
                self._send_json({"success": False, "error": "No report file was provided."}, status=400)
                return

            mime_type = body.get("mime_type", "application/pdf")
            api_key = body.get("api_key") or os.environ.get("GEMINI_API_KEY")

            if not api_key:
                self._send_json(
                    {
                        "success": False,
                        "error": (
                            "Gemini API key is required for live extraction.\n"
                            "Please enter your key in the form or set the GEMINI_API_KEY environment variable."
                        ),
                    },
                    status=400,
                )
                return

            file_bytes = base64.b64decode(file_base64)

            # 4. Call live extraction engine
            report_data = extract_medical_report(
                file_input=file_bytes,
                api_key=api_key,
                mime_type=mime_type,
            )

            # 5. Format structured response
            response_payload = {
                "success": True,
                "is_demo": False,
                "user_provided": {
                    "source": "User-Provided",
                    "name": user_intake.get("name", "").strip() or "Anonymous / Unspecified",
                    "age": user_intake.get("age", "").strip() or "Not stated",
                    "sex": user_intake.get("sex", "").strip() or "Not stated",
                    "symptoms": user_intake.get("symptoms", "").strip() or "None specified",
                    "allergies": user_intake.get("allergies", "").strip() or "None known",
                },
                "ai_extracted": report_data.model_dump(),
            }
            self._send_json(response_payload, status=200)

        except Exception as exc:
            self._send_json({"success": False, "error": str(exc)}, status=500)

    def _send_json(self, payload: dict, status: int = 200):
        """Send JSON HTTP response."""
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def _send_text(self, text: str, status: int = 200):
        """Send plain text HTTP response."""
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def log_message(self, format, *args):
        """Print server requests cleanly to stdout."""
        sys.stdout.write(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


def run_server():
    """Start the MedLens HTTP Server on port 8765 and auto-launch browser."""
    ports_to_try = [DEFAULT_PORT, 8080, 5000, 8000]
    server = None
    active_port = None

    for port in ports_to_try:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), MedLensHandler)
            active_port = port
            break
        except OSError:
            continue

    if not server:
        sys.stderr.write("ERROR: Could not bind to any available port.\n")
        sys.exit(1)

    url = f"http://127.0.0.1:{active_port}/"

    print("\n" + "=" * 65)
    print("  MedLens AI Insight Server is RUNNING")
    print("=" * 65)
    print(f"  --> Opening URL automatically: {url}")
    print(f"  Health check: http://127.0.0.1:{active_port}/api/health")
    print(f"  Gemini SDK:   {'[AVAILABLE]' if _GOOGLE_GENAI_AVAILABLE else '[NOT INSTALLED] (Demo mode ready!)'}")
    print("=" * 65)
    print("  Press Ctrl+C in this terminal to stop the server.\n")
    sys.stdout.flush()

    # Automatically launch default browser in 1 second
    def launch():
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass

    threading.Timer(0.8, launch).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down MedLens server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
