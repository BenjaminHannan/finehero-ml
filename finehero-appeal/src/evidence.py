# evidence.py - Processes user-submitted evidence files (photos, PDFs, text)
# using Claude's vision API to extract descriptions for appeal letter context.

import base64
import mimetypes
import os
from pathlib import Path

import anthropic

SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SUPPORTED_DOC   = {".pdf"}
SUPPORTED_TEXT  = {".txt", ".md"}

client = anthropic.Anthropic()


def _encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    suffix = Path(path).suffix.lower()
    mt_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    media_type = mt_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type


def _pdf_to_images(path: str) -> list[tuple[str, str]]:
    """Convert PDF pages to base64 PNG images using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            png_bytes = pix.tobytes("png")
            b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
            images.append((b64, "image/png"))
        return images
    except ImportError:
        print("  [WARN] PyMuPDF not installed; PDF converted as text placeholder.")
        return []


def analyze_image(path: str, exhibit_num: int) -> dict:
    """
    Send an image to Claude vision and get a structured description
    useful for an NYC parking ticket appeal letter.
    """
    print(f"    Analyzing image: {os.path.basename(path)} (Exhibit {exhibit_num})...")
    b64, media_type = _encode_image(path)

    prompt = (
        "You are analyzing evidence for an NYC parking ticket appeal. "
        "Describe what this image shows in 2-4 sentences, focusing on: "
        "parking signs and their restrictions, meter status, road conditions, "
        "hydrant locations, street markings, timestamps visible, any obstructions, "
        "or anything else relevant to challenging a parking violation. "
        "Be factual and specific — your description will appear in a legal letter."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    description = response.content[0].text.strip()
    return {
        "exhibit_num": exhibit_num,
        "path": path,
        "filename": os.path.basename(path),
        "type": "image",
        "description": description,
    }


def analyze_document(path: str, exhibit_num: int) -> dict:
    """Process a PDF document — convert pages to images then analyze."""
    print(f"    Analyzing document: {os.path.basename(path)} (Exhibit {exhibit_num})...")
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        pages = _pdf_to_images(path)
        if not pages:
            return {
                "exhibit_num": exhibit_num,
                "path": path,
                "filename": os.path.basename(path),
                "type": "document",
                "description": "[PDF document — contents could not be extracted. Please describe this document in your narrative.]",
            }

        # Analyze each page
        page_descriptions = []
        for i, (b64, media_type) in enumerate(pages[:6], 1):  # cap at 6 pages
            prompt = (
                f"This is page {i} of a document submitted as evidence in an NYC parking ticket appeal. "
                "Extract and summarize the key information: dates, amounts, official stamps, "
                "registration info, meter receipts, permit numbers, doctor/employer letters, "
                "or any other relevant facts. Be concise and factual."
            )
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            page_descriptions.append(f"[Page {i}] {response.content[0].text.strip()}")

        return {
            "exhibit_num": exhibit_num,
            "path": path,
            "filename": os.path.basename(path),
            "type": "document",
            "description": " | ".join(page_descriptions),
        }

    elif suffix in SUPPORTED_TEXT:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(3000)
        return {
            "exhibit_num": exhibit_num,
            "path": path,
            "filename": os.path.basename(path),
            "type": "text",
            "description": text,
        }

    return {
        "exhibit_num": exhibit_num,
        "path": path,
        "filename": os.path.basename(path),
        "type": "unknown",
        "description": "[File type not supported for automatic analysis.]",
    }


def process_evidence(file_paths: list[str]) -> list[dict]:
    """Process all evidence files and return list of exhibit dicts."""
    exhibits = []
    for i, path in enumerate(file_paths, 1):
        suffix = Path(path).suffix.lower()
        if suffix in SUPPORTED_IMAGE:
            exhibit = analyze_image(path, i)
        elif suffix in SUPPORTED_DOC or suffix in SUPPORTED_TEXT:
            exhibit = analyze_document(path, i)
        else:
            print(f"    [SKIP] Unsupported file type: {path}")
            continue
        exhibits.append(exhibit)
    return exhibits
