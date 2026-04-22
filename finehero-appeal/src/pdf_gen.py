# pdf_gen.py - Assembles the appeal letter and exhibits into a single PDF
# using ReportLab for the letter and PyMuPDF/Pillow for exhibit pages.

import base64
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _make_styles():
    base = getSampleStyleSheet()
    styles = {
        "Normal": base["Normal"],
        "header": ParagraphStyle(
            "header", parent=base["Normal"],
            fontSize=10, leading=14, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=10, leading=15, spaceAfter=8, firstLineIndent=0,
        ),
        "exhibit_title": ParagraphStyle(
            "exhibit_title", parent=base["Heading2"],
            fontSize=12, spaceAfter=6, spaceBefore=12, textColor=colors.HexColor("#1a1a2e"),
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"],
            fontSize=9, leading=12, textColor=colors.grey, spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"],
            fontSize=8, alignment=TA_CENTER, textColor=colors.grey,
        ),
        "title": ParagraphStyle(
            "title", parent=base["Heading1"],
            fontSize=14, spaceAfter=20, alignment=TA_CENTER,
            textColor=colors.HexColor("#1a1a2e"),
        ),
    }
    return styles


def _letter_paragraphs(letter_text: str, styles: dict) -> list:
    """Split letter text into Paragraph flowables."""
    elems = []
    for line in letter_text.split("\n"):
        stripped = line.strip()
        if stripped == "":
            elems.append(Spacer(1, 6))
        else:
            safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            elems.append(Paragraph(safe, styles["body"]))
    return elems


def _exhibit_page(exhibit: dict, styles: dict) -> list:
    """Build flowables for a single exhibit page."""
    elems = [PageBreak()]
    title = f"Exhibit {exhibit['exhibit_num']}: {exhibit['filename']}"
    elems.append(Paragraph(title, styles["exhibit_title"]))
    elems.append(Paragraph(exhibit["description"], styles["body"]))
    elems.append(Spacer(1, 12))

    suffix = Path(exhibit["path"]).suffix.lower()
    if suffix in SUPPORTED_IMAGE and os.path.exists(exhibit["path"]):
        try:
            img = Image(exhibit["path"])
            img.drawWidth = 6 * inch
            img.drawHeight = img.drawWidth * (img.imageHeight / img.imageWidth)
            if img.drawHeight > 7 * inch:
                img.drawHeight = 7 * inch
                img.drawWidth = img.drawHeight * (img.imageWidth / img.imageHeight)
            elems.append(img)
        except Exception as exc:
            elems.append(Paragraph(f"[Image could not be embedded: {exc}]", styles["caption"]))

    elif suffix == ".pdf":
        try:
            import fitz
            doc = fitz.open(exhibit["path"])
            for page_num, page in enumerate(doc, 1):
                pix = page.get_pixmap(dpi=120)
                png_path = exhibit["path"] + f"_page{page_num}.png"
                pix.save(png_path)
                img = Image(png_path)
                img.drawWidth = 6 * inch
                img.drawHeight = img.drawWidth * (pix.height / pix.width)
                if img.drawHeight > 7.5 * inch:
                    img.drawHeight = 7.5 * inch
                    img.drawWidth = img.drawHeight * (pix.width / pix.height)
                elems.append(img)
                elems.append(Spacer(1, 6))
                os.remove(png_path)
        except Exception as exc:
            elems.append(Paragraph(f"[PDF pages could not be embedded: {exc}]", styles["caption"]))

    return elems


def _add_page_numbers(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(4.25 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.drawCentredString(4.25 * inch, 0.35 * inch, "FineHero Appeal Letter Generator")
    canvas.restoreState()


def build_pdf(letter_text: str, exhibits: list[dict], output_path: str, ticket: dict) -> str:
    """
    Build the complete PDF: cover, letter, then one page per exhibit.
    Returns the output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    styles = _make_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=1.1 * inch,
        rightMargin=1.1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    story = []

    # Cover / header
    story.append(Paragraph("NOTICE OF CONTEST — NYC PARKING VIOLATION", styles["title"]))
    story.append(Spacer(1, 4))

    cover_data = [
        ["Summons Number:", ticket.get("summons_number", "N/A")],
        ["Violation:", f"{ticket.get('violation_code', '')} - {ticket.get('violation_description', '')}"],
        ["Date of Violation:", f"{ticket.get('issue_date', 'N/A')} at {ticket.get('violation_time', 'N/A')}"],
        ["Fine Amount:", f"${ticket.get('fine_amount', 'N/A')}"],
        ["Total Exhibits:", str(len(exhibits))],
    ]
    tbl = Table(cover_data, colWidths=[2 * inch, 4.2 * inch])
    tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 10),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4ff")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 24))

    # Letter body
    story.extend(_letter_paragraphs(letter_text, styles))

    # Exhibits
    for exhibit in exhibits:
        story.extend(_exhibit_page(exhibit, styles))

    doc.build(story, onFirstPage=_add_page_numbers, onLaterPages=_add_page_numbers)
    return output_path
