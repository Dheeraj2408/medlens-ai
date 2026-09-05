"""Serve the MedLens website and provide API endpoints for medical report extraction.

This version is rewritten as a Flask (WSGI) app so it works with Vercel's
Python runtime, which requires a module-level `app` object rather than a
long-running custom HTTP server.
"""

import base64
import json
import os
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, abort

from extractor import (
    _GOOGLE_GENAI_AVAILABLE,
    extract_medical_report,
    get_demo_report_data,
)
from firebase_db import save_report, get_recent_reports, is_available as firebase_available

# --- Optional Google Cloud services (same fallback pattern as before) ---
try:
    from gcs_storage import is_available as gcs_available, upload_report
    _GCS_AVAILABLE = gcs_available()
except ImportError:
    _GCS_AVAILABLE = False
    upload_report = None

try:
    from vision_ocr import is_available as vision_available, enhance_with_vision_preprocessing
    _VISION_AVAILABLE = vision_available()
except ImportError:
    _VISION_AVAILABLE = False
    enhance_with_vision_preprocessing = None

try:
    from natural_language import is_available as nlp_available, enhance_extraction_with_nlp
    _NLP_AVAILABLE = nlp_available()
except ImportError:
    _NLP_AVAILABLE = False
    enhance_extraction_with_nlp = None

try:
    from bigquery_analytics import is_available as bq_available, save_report_to_bigquery
    _BQ_AVAILABLE = bq_available()
except ImportError:
    _BQ_AVAILABLE = False
    save_report_to_bigquery = None

try:
    from healthcare_api import is_available as healthcare_available, save_fhir_resources
    _HEALTHCARE_AVAILABLE = healthcare_available()
except ImportError:
    _HEALTHCARE_AVAILABLE = False
    save_fhir_resources = None

ROOT = Path(__file__).resolve().parent

# Vercel's Python runtime looks for this module-level "app" object.
app = Flask(__name__)


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "google_genai_available": _GOOGLE_GENAI_AVAILABLE,
            "firebase_available": firebase_available(),
            "gcs_available": _GCS_AVAILABLE,
            "vision_available": _VISION_AVAILABLE,
            "nlp_available": _NLP_AVAILABLE,
            "bigquery_available": _BQ_AVAILABLE,
            "healthcare_available": _HEALTHCARE_AVAILABLE,
            "service": "MedLens AI Insight",
        }
    ), 200


@app.route("/api/reports", methods=["GET"])
def reports():
    try:
        limit = int(request.args.get("limit", 10))
    except (ValueError, TypeError):
        limit = 10
    reports_data = get_recent_reports(limit=min(limit, 50))
    return jsonify({"success": True, "reports": reports_data}), 200


@app.route("/api/process-report", methods=["POST"])
def process_report():
    try:
        body = request.get_json(force=True, silent=True)

        if not isinstance(body, dict):
            return jsonify({"success": False, "error": "Invalid request format."}), 400

        user_intake = body.get("user_intake", {})
        if not isinstance(user_intake, dict):
            user_intake = {}

        # 1. Demo mode
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
                    "medications": user_intake.get("medications", "").strip() or "Multivitamin, Vitamin D3",
                    "conditions": user_intake.get("conditions", "").strip() or "None known",
                },
                "ai_extracted": demo_data.model_dump(),
            }
            return jsonify(response_payload), 200

        # 2. Missing google-genai library
        if not _GOOGLE_GENAI_AVAILABLE:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "The 'google-genai' library is not yet installed in this Python environment.\n"
                        "Please add google-genai to requirements.txt to enable live Gemini extraction.\n"
                        "(Tip: Click 'Load Sample Lab Report (Demo)' to test the interface right away!)"
                    ),
                }
            ), 400

        # 3. Validate uploaded file
        file_base64 = body.get("file_base64")
        if not file_base64:
            return jsonify({"success": False, "error": "No report file was provided."}), 400

        allowed_mime_types = ["application/pdf", "image/jpeg", "image/png", "image/webp"]
        mime_type = body.get("mime_type", "application/pdf")
        if mime_type not in allowed_mime_types:
            return jsonify(
                {
                    "success": False,
                    "error": f"Invalid file type. Allowed types: {', '.join(allowed_mime_types)}",
                }
            ), 400

        api_key = body.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Gemini API key is required for live extraction.\n"
                        "Please enter your key in the form or set the GEMINI_API_KEY environment variable."
                    ),
                }
            ), 400

        file_bytes = base64.b64decode(file_base64)

        # 4. Call live extraction engine
        report_data = extract_medical_report(
            file_input=file_bytes,
            api_key=api_key,
            mime_type=mime_type,
        )

        report_dict = report_data.model_dump()

        # 5. Enhance with Google Cloud services
        if _VISION_AVAILABLE and enhance_with_vision_preprocessing:
            try:
                report_dict = enhance_with_vision_preprocessing(file_bytes, report_dict)
                print("[Google Services] Applied Vision API preprocessing")
            except Exception as exc:
                print(f"[Google Services] Vision preprocessing failed: {exc}")

        if _NLP_AVAILABLE and enhance_extraction_with_nlp:
            try:
                report_dict = enhance_extraction_with_nlp(report_dict)
                print("[Google Services] Applied NLP analysis")
            except Exception as exc:
                print(f"[Google Services] NLP analysis failed: {exc}")

        gcs_object_name = None
        if _GCS_AVAILABLE and upload_report:
            try:
                gcs_object_name = upload_report(
                    file_data=file_bytes,
                    original_filename=f"report_{body.get('filename', 'upload')}",
                    mime_type=mime_type,
                    metadata={"user_intake": str(user_intake)},
                )
                print(f"[Google Services] Uploaded to GCS: {gcs_object_name}")
            except Exception as exc:
                print(f"[Google Services] GCS upload failed: {exc}")

        if _BQ_AVAILABLE and save_report_to_bigquery:
            try:
                report_id = str(uuid.uuid4())
                save_report_to_bigquery(report_id, user_intake, report_dict, is_demo=False)
                print("[Google Services] Saved to BigQuery analytics")
            except Exception as exc:
                print(f"[Google Services] BigQuery save failed: {exc}")

        if _HEALTHCARE_AVAILABLE and save_fhir_resources:
            try:
                report_id = str(uuid.uuid4())
                fhir_result = save_fhir_resources(report_dict, user_intake, report_id)
                print(f"[Google Services] Generated FHIR resources: {fhir_result.get('status')}")
            except Exception as exc:
                print(f"[Google Services] Healthcare API export failed: {exc}")

        # 6. Format structured response
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
                "medications": user_intake.get("medications", "").strip() or "None specified",
                "conditions": user_intake.get("conditions", "").strip() or "None specified",
            },
            "ai_extracted": report_dict,
            "google_services": {
                "gcs_storage": gcs_object_name if gcs_object_name else "Not used",
                "vision_ocr": "Applied" if _VISION_AVAILABLE else "Not available",
                "nlp_analysis": "Applied" if _NLP_AVAILABLE else "Not available",
                "bigquery_analytics": "Saved" if _BQ_AVAILABLE else "Not available",
                "healthcare_fhir": "Generated" if _HEALTHCARE_AVAILABLE else "Not available",
            },
        }
        return jsonify(response_payload), 200

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_static(path):
    """Serve index.html and other static files from the project root."""
    if not path:
        path = "index.html"

    target_file = (ROOT / path).resolve()

    # Security: block path traversal outside the project root
    try:
        target_file.relative_to(ROOT)
    except ValueError:
        abort(403)

    if not target_file.is_file():
        abort(404)

    return send_from_directory(ROOT, path)


# Only used when running locally (e.g. `python app.py`). Vercel calls the
# `app` object directly and never executes this block.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    app.run(host="0.0.0.0", port=port, debug=False)