import io
import os
from typing import List, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable, Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String

# ── Brand colours ─────────────────────────────────────────────────────────
GREEN = colors.HexColor("#2e7d32")
LIGHT_GREEN = colors.HexColor("#e8f5e9")
MID_GREEN = colors.HexColor("#a5d6a7")
DARK_TEXT = colors.HexColor("#1a1a1a")
GREY_TEXT = colors.HexColor("#616161")

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")

# ── Helpers ───────────────────────────────────────────────────────────────

def _stat_cell(label: str, value: str) -> Table:
    cell = Table(
        [
            [Paragraph(value, ParagraphStyle("SV", fontSize=16, fontName="Helvetica-Bold",
                                              textColor=GREEN, alignment=TA_CENTER,
                                              leading=18))],
            [Paragraph(label, ParagraphStyle("SL", fontSize=7, textColor=GREY_TEXT,
                                              alignment=TA_CENTER, leading=9))],
        ],
        rowHeights=[11 * mm, 6 * mm],
    )
    cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREEN),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("VALIGN", (0, 0), (0, 0), "BOTTOM"),
        ("VALIGN", (0, 1), (0, 1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, MID_GREEN),
    ]))
    return cell

def draw_logo(canvas, doc):
    logo_width = 37.5 * mm
    logo_height = 20 * mm

    x = MARGIN  # left edge
    y = PAGE_H - logo_height - 5 * mm  # VERY TOP (slightly above margin)

    canvas.drawImage(
        _LOGO_PATH,
        x,
        y,
        width=logo_width,
        height=logo_height,
        preserveAspectRatio=True,
        mask='auto'
    )

# ── Public API ────────────────────────────────────────────────────────────

def generate_pdf(
    species_data: List[Tuple[str, int]],
    total_species: int,
    total_identifications: int,
    total_feedback: int,
    accuracy_pct: str,
    export_date: str,
) -> io.BytesIO:
    """Build a branded PDF report and return the bytes buffer."""

    styles = getSampleStyleSheet()

    section_title_style = ParagraphStyle(
        "SectionTitle", parent=styles["Normal"],
        fontSize=9, textColor=GREY_TEXT, fontName="Helvetica-Bold",
        spaceAfter=3 * mm, spaceBefore=6 * mm,
        borderPadding=(0, 0, 2, 0),
    )
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=7, textColor=GREY_TEXT, alignment=TA_CENTER,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"],
        fontSize=8, textColor=GREY_TEXT, alignment=TA_RIGHT,
    )

    elements: list = []

    # ── Header bar (green background with logo + title) ──────────────────
    HEADER_H = 22 * mm

    # Build the header as a Table: [logo | title block]
    title_para = Paragraph(
        "PetreeBlitz - Archaeobotany AI Tree Identification System",
        ParagraphStyle("HTitle", fontSize=18, fontName="Helvetica-Bold",
                        textColor=colors.white, leading=20),
    )
    subtitle_para = Paragraph(
        "Species Identification Report",
        ParagraphStyle("HSub", fontSize=9, fontName="Helvetica",
                        textColor=colors.HexColor("#c8e6c9"), leading=11),
    )
    title_cell = Table(
        [[title_para], [subtitle_para]],
        rowHeights=[12 * mm, 6 * mm],
    )
    title_cell.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (0, 0), "BOTTOM"),
        ("VALIGN", (0, 1), (0, 1), "TOP"),
    ]))

    header_data = [[title_cell]]
    header_col_widths = [CONTENT_W]

    header_table = Table(header_data, colWidths=header_col_widths, rowHeights=[HEADER_H])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREEN),
        ("ROUNDEDCORNERS", [4, 4, 0, 0]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 4 * mm))

    # ── Export metadata line ──────────────────────────────────────────────
    elements.append(Paragraph(f"Exported from dashboard on {export_date}", meta_style))
    elements.append(Spacer(1, 3 * mm))

    # ── Key stats row (4 boxes) ───────────────────────────────────────────
    box_w = (CONTENT_W - 3 * 4 * mm) / 4
    stats_row = Table(
        [[
            _stat_cell("Species Known", str(total_species)),
            _stat_cell("Identifications", str(total_identifications)),
            _stat_cell("Feedback Received", str(total_feedback)),
            _stat_cell("Accuracy", accuracy_pct),
        ]],
        colWidths=[box_w] * 4,
        rowHeights=[20 * mm],
        hAlign="LEFT",
    )
    stats_row.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(stats_row)
    elements.append(Spacer(1, 5 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREEN))

    # ── Species table ─────────────────────────────────────────────────────
    elements.append(Paragraph("SPECIES DATABASE", section_title_style))

    table_data = [[
        Paragraph("<b>#</b>", ParagraphStyle("TH", fontSize=9, textColor=colors.white, alignment=TA_CENTER)),
        Paragraph("<b>Species Name</b>", ParagraphStyle("TH", fontSize=9, textColor=colors.white)),
        Paragraph("<b>Samples</b>", ParagraphStyle("TH", fontSize=9, textColor=colors.white, alignment=TA_CENTER)),
    ]]
    for i, (name, count) in enumerate(species_data, 1):
        num_style = ParagraphStyle("TD_num", fontSize=8, textColor=GREY_TEXT, alignment=TA_CENTER)
        name_style = ParagraphStyle("TD_name", fontSize=8, textColor=DARK_TEXT, fontName="Helvetica-Oblique")
        count_style = ParagraphStyle("TD_count", fontSize=8, textColor=DARK_TEXT, alignment=TA_CENTER)
        table_data.append([
            Paragraph(str(i), num_style),
            Paragraph(name, name_style),
            Paragraph(str(count), count_style),
        ])

    col_widths = [12 * mm, CONTENT_W - 12 * mm - 22 * mm, 22 * mm]
    species_table = Table(table_data, colWidths=col_widths, repeatRows=1)

    row_backgrounds = [("BACKGROUND", (0, 0), (-1, 0), GREEN)]
    for i in range(1, len(table_data)):
        bg = colors.white if i % 2 == 1 else LIGHT_GREEN
        row_backgrounds.append(("BACKGROUND", (0, i), (-1, i), bg))

    species_table.setStyle(TableStyle([
        *row_backgrounds,
        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREEN),
        ("LINEBELOW", (0, 0), (-1, 0), 1, GREEN),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(species_table)

    # ── Footer ────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 6 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREEN))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        f"PetreeBlitz &bull; Federated Wood Anatomy Identification &bull; {export_date}",
        footer_style,
    ))

    # ── Build ─────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=MARGIN, bottomMargin=MARGIN,
        leftMargin=MARGIN, rightMargin=MARGIN,
    )
    doc.build(elements, onFirstPage=draw_logo)
    buf.seek(0)
    return buf
