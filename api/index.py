"""MedLens AI — Vercel Python Serverless Function.

Handles all /api/* requests. Vercel automatically detects any Python file
inside the /api directory and runs it as a serverless function.

Routes:
  GET  /api/           → health check
  POST /api/process    → extract medical report via Gemini AI
"""

from __future__ import annotations

import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# ── resolve project root so extractor.py is importable ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── try to import the Gemini extraction engine ───────────────────────────────
try:
    from extractor import (
        _GOOGLE_GENAI_AVAILABLE,
        extract_medical_report,
        get_demo_report_data,
    )
    _EXTRACTOR_LOADED = True
except Exception as _import_err:
    _GOOGLE_GENAI_AVAILABLE = False
    _EXTRACTOR_LOADED = False
    _import_err_msg = str(_import_err)


# ─────────────────────────────────────────────────────────────────────────────
# Vercel Handler
# ─────────────────────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):
    """Serverless request handler called by Vercel for every HTTP request."""

    # ── routing ──────────────────────────────────────────────────────────────

    def do_OPTIONS(self):
        """Preflight CORS for browser requests."""
        self._send_response(204, b"", content_type="text/plain")

    def do_GET(self):
        path = self._clean_path()
        if path in ("/api", "/api/", "/api/health"):
            self._health()
        else:
            self._not_found()

    def do_POST(self):
        path = self._clean_path()
        if path in ("/api", "/api/", "/api/process", "/api/process-report"):
            self._process_report()
        else:
            self._not_found()

    # ── handlers ─────────────────────────────────────────────────────────────

    def _health(self):
        self._json(200, {
            "status": "ok",
            "service": "MedLens AI Insight",
            "platform": "Vercel Serverless",
            "gemini_available": _GOOGLE_GENAI_AVAILABLE,
            "extractor_loaded": _EXTRACTOR_LOADED,
        })

    def _process_report(self):
        # ── parse body ───────────────────────────────────────────────────────
        try:
            content_length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            content_length = 0

        MAX_SIZE = 25 * 1024 * 1024  # 25 MB
        if content_length > MAX_SIZE:
            self._json(413, {"success": False, "error": "File too large. Maximum 25 MB."})
            return

        try:
            raw = self.rfile.read(content_length) if content_length else b""
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"success": False, "error": "Invalid JSON in request body."})
            return

        if not isinstance(body, dict):
            self._json(400, {"success": False, "error": "Request body must be a JSON object."})
            return

        user_intake = body.get("user_intake") or {}
        if not isinstance(user_intake, dict):
            user_intake = {}

        # ── check extractor availability ─────────────────────────────────────
        if not _EXTRACTOR_LOADED:
            self._json(500, {
                "success": False,
                "error": (
                    f"Extractor module failed to load: {_import_err_msg if '_import_err_msg' in dir() else 'unknown error'}. "
                    "Check that requirements.txt includes google-genai and pydantic."
                ),
            })
            return

        # ── demo mode ────────────────────────────────────────────────────────
        if body.get("is_demo"):
            demo = get_demo_report_data()
            self._json(200, {
                "success": True,
                "is_demo": True,
                "user_provided": _build_user_provided(user_intake, demo=True),
                "ai_extracted": demo.model_dump(),
            })
            return

        # ── live extraction ──────────────────────────────────────────────────
        if not _GOOGLE_GENAI_AVAILABLE:
            self._json(400, {
                "success": False,
                "error": (
                    "The google-genai library is not installed in this environment.\n"
                    "Tip: try the Demo mode to explore the interface without an API key."
                ),
            })
            return

        file_b64 = body.get("file_base64", "")
        if not file_b64:
            self._json(400, {"success": False, "error": "No file provided. Please upload a lab report."})
            return

        allowed_mimes = {"application/pdf", "image/jpeg", "image/png", "image/webp"}
        mime_type = body.get("mime_type", "application/pdf")
        if mime_type not in allowed_mimes:
            self._json(400, {
                "success": False,
                "error": f"Unsupported file type '{mime_type}'. Allowed: PDF, JPG, PNG, WebP.",
            })
            return

        api_key = (body.get("api_key") or "").strip() or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            self._json(400, {
                "success": False,
                "error": (
                    "A Gemini API key is required for live extraction.\n"
                    "Enter your key in the sidebar, or set GEMINI_API_KEY in your Vercel project "
                    "environment variables."
                ),
            })
            return

        try:
            file_bytes = base64.b64decode(file_b64)
        except Exception:
            self._json(400, {"success": False, "error": "Could not decode the uploaded file. Please try again."})
            return

        try:
            report = extract_medical_report(
                file_input=file_bytes,
                api_key=api_key,
                mime_type=mime_type,
            )
        except Exception as exc:
            self._json(500, {"success": False, "error": f"Extraction failed: {exc}"})
            return

        self._json(200, {
            "success": True,
            "is_demo": False,
            "user_provided": _build_user_provided(user_intake, demo=False),
            "ai_extracted": report.model_dump(),
        })

    def _not_found(self):
        self._json(404, {"error": f"Endpoint not found: {self._clean_path()}"})

    # ── helpers ───────────────────────────────────────────────────────────────

    def _clean_path(self) -> str:
        return self.path.split("?")[0].rstrip("/") or "/"

    def _json(self, status: int, payload: dict):
        data = json.dumps(payload, indent=2).encode("utf-8")
        self._send_response(status, data, content_type="application/json; charset=utf-8")

    def _send_response(self, status: int, body: bytes, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress default noisy request logging; Vercel captures stdout anyway
        print(f"[MedLens] {fmt % args}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_provided(intake: dict, demo: bool) -> dict:
    def g(key, fallback=""):
        return (intake.get(key) or "").strip() or fallback

    if demo:
        return {
            "source": "User-Provided",
            "name": g("name", "Eleanor Vance (Demo)"),
            "age": g("age", "46"),
            "sex": g("sex", "Female"),
            "symptoms": g("symptoms", "Fatigue, mild exertion dyspnea, and dizziness for 3 weeks."),
            "allergies": g("allergies", "Penicillin (rash)"),
            "medications": g("medications", "Multivitamin, Vitamin D3 1000 IU daily"),
            "conditions": g("conditions", "None known"),
        }
    return {
        "source": "User-Provided",
        "name": g("name", "Anonymous / Unspecified"),
        "age": g("age", "Not stated"),
        "sex": g("sex", "Not stated"),
        "symptoms": g("symptoms", "None specified"),
        "allergies": g("allergies", "None known"),
        "medications": g("medications", "None specified"),
        "conditions": g("conditions", "None specified"),
    }
