"""Medical Report Extraction Module for MedLens AI.

Uses the official `google-genai` SDK and Gemini models to parse uploaded medical
reports (PDF, JPG, PNG), extracting test names, values, and reference ranges with
strict grounding (no invented ranges) and clear 'AI-Extracted' provenance tagging.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import BinaryIO, Optional, Union

# ==============================================================================
# Dependency Imports with Graceful Fallback
# ==============================================================================

try:
    from google import genai
    from google.genai import types
    _GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    _GOOGLE_GENAI_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Minimal fallback shim so the module can load before dependencies are installed
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            out = {}
            for k, v in self.__dict__.items():
                if k.startswith("_"):
                    continue
                if hasattr(v, "model_dump"):
                    out[k] = v.model_dump()
                elif isinstance(v, list):
                    out[k] = [item.model_dump() if hasattr(item, "model_dump") else item for item in v]
                else:
                    out[k] = v
            return out
        def model_dump_json(self, indent=2):
            import json
            return json.dumps(self.model_dump(), indent=indent)
        @classmethod
        def model_validate_json(cls, json_str):
            import json
            return cls(**json.loads(json_str))

    def Field(*args, **kwargs):  # type: ignore
        return kwargs.get("default", None)


class LabTestItem(BaseModel):
    """Represents a single test extracted from a lab report."""

    data_source: str = Field(
        default="AI-Extracted",
        description="Provenance tag. Always 'AI-Extracted' to distinguish from 'User-Provided' intake data.",
    )
    test_name: str = Field(
        description="The full name of the laboratory or diagnostic test (e.g., 'Hemoglobin', 'Total Cholesterol')."
    )
    value: str = Field(
        description="The observed measurement, result, or value reported for this test."
    )
    unit: Optional[str] = Field(
        default=None,
        description="The measurement unit printed in the document (e.g., 'g/dL', 'mg/dL', 'x10^3/uL', '%'). If absent, null.",
    )
    reference_range: Optional[str] = Field(
        default=None,
        description=(
            "The normal/reference range EXPLICITLY printed in the uploaded document. "
            "CRITICAL: If the document does not show a reference range, this MUST be null. "
            "Do NOT extrapolate, deduce, or supply external reference ranges."
        ),
    )
    status: Optional[str] = Field(
        default=None,
        description=(
            "Interpretation based SOLELY on the printed reference range or printed flags "
            "(e.g., 'Within Range', 'High', 'Low', 'Abnormal', or 'Not Stated'). "
            "If no reference range or flag was found in the report, this MUST be 'Not Stated'."
        ),
    )


class ExtractedMedicalReport(BaseModel):
    """Root model for data extracted from an uploaded medical lab report."""

    data_source: str = Field(
        default="AI-Extracted",
        description="Data provenance label. Always 'AI-Extracted' to differentiate from 'User-Provided' data.",
    )
    patient_name_in_document: Optional[str] = Field(
        default=None,
        description="Patient name if explicitly visible on the report.",
    )
    collection_or_report_date: Optional[str] = Field(
        default=None,
        description="Date of collection, test, or report printed on the document.",
    )
    facility_or_lab_name: Optional[str] = Field(
        default=None,
        description="Name of the testing laboratory, clinic, or hospital printed on the document.",
    )
    tests: list[LabTestItem] = Field(
        default_factory=list,
        description="List of lab tests and findings extracted from the report.",
    )
    notes_and_impressions: Optional[str] = Field(
        default=None,
        description="Any physician comments, specimen notes, or clinical impressions explicitly written in the report.",
    )
    extraction_warnings: list[str] = Field(
        default_factory=list,
        description="Any notices regarding blurry text, missing reference intervals, or illegible fields.",
    )


# ==============================================================================
# System Instructions & Prompts
# ==============================================================================

SYSTEM_INSTRUCTION = """\
You are a specialized clinical document extraction engine for MedLens AI.
Your sole job is to read uploaded lab reports or diagnostic documents and digitize the test results into structured data.

Follow these STRICT rules at all times:
1. SPECIFIC FOCUS: Look specifically for all diagnostic test names and their measured values.
2. STRICT REFERENCE RANGES (DO NOT INVENT):
   - You must ONLY use the reference ranges that are explicitly printed in the uploaded document.
   - NEVER invent, guess, estimate, or pull reference ranges from external clinical knowledge or training memory.
   - If the document does not show a reference range for a test, set `reference_range` to null.
3. STATUS EVALUATION:
   - Only determine if a test is 'Within Range', 'High', 'Low', or 'Abnormal' if the document explicitly prints a reference range or flags the value (e.g., with 'H', 'L', '*', bold, or 'FLAG').
   - If no reference range or document flag is available, set `status` to 'Not Stated'.
4. PROVENANCE LABELING:
   - Ensure the `data_source` field is always set to 'AI-Extracted'.
   - This differentiates data extracted by AI from 'User-Provided' patient intake data.
5. ACCURACY & FIDELITY:
   - Preserve exact test names, numbers, decimal points, and units as printed.
   - Do not make medical diagnoses or provide treatment advice.
"""

USER_PROMPT = """\
Analyze this uploaded medical lab report.
1. Extract all test names and values.
2. For each test, extract its printed unit, printed reference range, and status.
3. Remember: Only include reference ranges that are explicitly visible in this document. Never invent your own.
4. Label all data as 'AI-Extracted'.
"""


# ==============================================================================
# Helper Functions
# ==============================================================================

def _resolve_file_bytes_and_mime(
    file_input: Union[str, Path, bytes, BinaryIO],
    mime_type: Optional[str] = None,
) -> tuple[bytes, str]:
    """Helper to convert various file input types into raw bytes and a MIME type."""
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        if not path.is_file():
            raise FileNotFoundError(f"Medical report file not found at: {path}")

        if not mime_type:
            guessed_mime, _ = mimetypes.guess_type(str(path))
            mime_type = guessed_mime or (
                "application/pdf" if path.suffix.lower() == ".pdf" else "image/jpeg"
            )
        data = path.read_bytes()
        return data, mime_type

    if isinstance(file_input, bytes):
        if not mime_type:
            raise ValueError(
                "When passing raw bytes, you must supply the `mime_type` parameter "
                "(e.g., 'application/pdf', 'image/png', or 'image/jpeg')."
            )
        return file_input, mime_type

    if hasattr(file_input, "read"):
        data = file_input.read()
        if not mime_type:
            filename = getattr(file_input, "name", None) or getattr(file_input, "filename", None)
            if filename:
                guessed_mime, _ = mimetypes.guess_type(str(filename))
                mime_type = guessed_mime
            if not mime_type:
                raise ValueError("Could not infer MIME type from file-like object. Please provide `mime_type`.")
        return data, mime_type

    raise TypeError(f"Unsupported file_input type: {type(file_input)}")


# ==============================================================================
# Primary Extraction Function
# ==============================================================================

def extract_medical_report(
    file_input: Union[str, Path, bytes, BinaryIO],
    api_key: Optional[str] = None,
    mime_type: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> ExtractedMedicalReport:
    """Read an uploaded medical report and extract lab test names and values.

    Args:
        file_input: Path to report file (PDF, PNG, JPG), raw bytes, or file-like stream.
        api_key: Gemini API key. If omitted, `GEMINI_API_KEY` from the environment is used.
        mime_type: MIME type of the file (e.g., 'application/pdf', 'image/png', 'image/jpeg').
            Automatically detected for file paths.
        model: Gemini model identifier (default: 'gemini-2.5-flash').

    Returns:
        ExtractedMedicalReport: A validated Pydantic model containing:
            - `data_source`: Marked as 'AI-Extracted'
            - `tests`: List of LabTestItem with test_name, value, unit, reference_range, status
            - `notes_and_impressions`: Extracted clinical notes
            - `extraction_warnings`: Any warnings regarding missing ranges or poor scan quality

    Raises:
        ValueError: If API key is missing or file format is invalid.
        genai.errors.APIError: If the Gemini API request fails.
    """
    # 0. Check library availability
    if not _GOOGLE_GENAI_AVAILABLE:
        raise ModuleNotFoundError(
            "The 'google-genai' library is not installed in your Python environment.\n"
            "Please run: pip install -r requirements.txt (or 'pip install google-genai pydantic') in your terminal."
        )

    # 1. Resolve API Key
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Gemini API key is required. Pass `api_key='...'` or set the "
            "GEMINI_API_KEY environment variable."
        )

    # 2. Read file data and MIME type
    file_bytes, resolved_mime = _resolve_file_bytes_and_mime(file_input, mime_type)

    # 3. Initialize Google GenAI client
    client = genai.Client(api_key=resolved_api_key)

    # 4. Construct content part from bytes
    document_part = types.Part.from_bytes(
        data=file_bytes,
        mime_type=resolved_mime,
    )

    # 5. Call Gemini with strict structured output schema & instructions
    response = client.models.generate_content(
        model=model,
        contents=[
            document_part,
            USER_PROMPT,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,  # 0.0 for deterministic, factual extraction
            response_mime_type="application/json",
            response_schema=ExtractedMedicalReport,
        ),
    )

    # 6. Return the parsed Pydantic object
    # When response_schema is passed, client.models.generate_content parses
    # directly into `response.parsed`.
    if response.parsed is not None:
        return response.parsed

    # Fallback in case raw text with markdown formatting was returned:
    text = (response.text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return ExtractedMedicalReport.model_validate_json(text)


# ==============================================================================
# Demo & Offline Verification Utility
# ==============================================================================

def get_demo_report_data() -> ExtractedMedicalReport:
    """Generate realistic lab report data for UI demonstration and offline verification.

    Demonstrates:
      - Rigorous 'AI-Extracted' provenance tagging.
      - Grounded reference ranges printed in the document.
      - Explicit handling of tests where reference ranges were NOT printed in the
        document (e.g. Serum Ferritin) to show that ranges are never invented.
    """
    return ExtractedMedicalReport(
        data_source="AI-Extracted",
        patient_name_in_document="Eleanor Vance",
        collection_or_report_date="2026-08-14",
        facility_or_lab_name="MetroHealth Clinical Laboratories",
        tests=[
            LabTestItem(
                data_source="AI-Extracted",
                test_name="Hemoglobin",
                value="11.2",
                unit="g/dL",
                reference_range="12.0 - 15.5 g/dL",
                status="Low",
            ),
            LabTestItem(
                data_source="AI-Extracted",
                test_name="Hematocrit",
                value="34.1",
                unit="%",
                reference_range="37.0 - 48.0 %",
                status="Low",
            ),
            LabTestItem(
                data_source="AI-Extracted",
                test_name="WBC Count",
                value="6.8",
                unit="x10^3/uL",
                reference_range="4.5 - 11.0 x10^3/uL",
                status="Within Range",
            ),
            LabTestItem(
                data_source="AI-Extracted",
                test_name="Platelets",
                value="245",
                unit="x10^3/uL",
                reference_range="150 - 450 x10^3/uL",
                status="Within Range",
            ),
            LabTestItem(
                data_source="AI-Extracted",
                test_name="Serum Ferritin",
                value="14",
                unit="ng/mL",
                reference_range=None,
                status="Not Stated",
            ),
            LabTestItem(
                data_source="AI-Extracted",
                test_name="Fasting Glucose",
                value="108",
                unit="mg/dL",
                reference_range="70 - 99 mg/dL",
                status="High",
            ),
        ],
        notes_and_impressions=(
            "Specimen hemolyzed: None. "
            "Automated differential complete. Mild microcytic picture noted on blood smear."
        ),
        extraction_warnings=[
            "No reference range was printed in the report for Serum Ferritin; "
            "marked as 'Not Stated' per MedLens strict document-grounding policy."
        ],
    )



if __name__ == "__main__":
    print("MedLens AI Report Extractor module ready.")
    print("Usage:")
    print("  from extractor import extract_medical_report")
    print("  result = extract_medical_report('path/to/report.pdf', api_key='YOUR_GEMINI_API_KEY')")
    print("  print(result.model_dump_json(indent=2))")
