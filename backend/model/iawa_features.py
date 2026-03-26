"""IAWA feature definitions and result structures for wood anatomy analysis."""

from dataclasses import dataclass
from typing import List, Dict

# ── Category labels ────────────────────────────────────────────────────────────
CATEGORY_GROWTH_RINGS = "Growth Rings"
CATEGORY_POROSITY     = "Porosity"
CATEGORY_VESSELS      = "Vessels"
CATEGORY_VESSEL_PITS  = "Vessel Pits"
CATEGORY_FIBERS       = "Fibers"
CATEGORY_PARENCHYMA   = "Parenchyma"
CATEGORY_RAYS         = "Rays"
CATEGORY_CRYSTALS     = "Crystals"


@dataclass
class IAWAFeatureDef:
    """Static IAWA feature definition (catalogued character)."""
    id: int
    name: str
    category: str
    description: str


@dataclass
class FeatureResult:
    """Runtime detection result for one IAWA feature on a sample."""
    id: int
    name: str
    category: str
    description: str
    is_present: bool   # whether this feature was detected
    confidence: float  # confidence in the detection (0–1)


# ── Master feature catalogue  (IAWA character numbers, InsideWood standard) ───
IAWA_FEATURES: List[IAWAFeatureDef] = [
    # Growth Rings
    IAWAFeatureDef(1,  "Growth ring boundaries distinct",
                   CATEGORY_GROWTH_RINGS,
                   "Abrupt changes in wood density mark clear boundaries between growth increments."),
    IAWAFeatureDef(2,  "Growth ring boundaries indistinct or absent",
                   CATEGORY_GROWTH_RINGS,
                   "No clear boundary visible between successive growth increments."),

    # Porosity
    IAWAFeatureDef(5,  "Wood diffuse-porous",
                   CATEGORY_POROSITY,
                   "Vessels of roughly similar size distributed uniformly across the growth ring."),
    IAWAFeatureDef(6,  "Wood ring-porous",
                   CATEGORY_POROSITY,
                   "Distinctly larger vessels in early wood than in late wood."),

    # Vessels – Perforation Plates
    IAWAFeatureDef(13, "Simple perforation plates",
                   CATEGORY_VESSELS,
                   "A single, unobstructed opening constitutes the perforation plate."),
    IAWAFeatureDef(14, "Scalariform perforation plates",
                   CATEGORY_VESSELS,
                   "Multiple parallel bars span the perforation plate like a ladder."),
    IAWAFeatureDef(16, "Scalariform plates with 10–20 bars",
                   CATEGORY_VESSELS,
                   "Scalariform perforation plate bearing 10 to 20 horizontal bars."),
    IAWAFeatureDef(17, "Scalariform plates with 20–40 bars",
                   CATEGORY_VESSELS,
                   "Scalariform perforation plate bearing 20 to 40 horizontal bars."),

    # Vessel Pits
    IAWAFeatureDef(20, "Intervessel pits scalariform",
                   CATEGORY_VESSEL_PITS,
                   "Elongate pits arranged in vertical rows, resembling a ladder pattern."),
    IAWAFeatureDef(21, "Intervessel pits opposite",
                   CATEGORY_VESSEL_PITS,
                   "Pits in horizontal rows perpendicular to the vessel axis."),
    IAWAFeatureDef(22, "Intervessel pits alternate",
                   CATEGORY_VESSEL_PITS,
                   "Pits in diagonal rows, hexagonally packed – the most common tropical pattern."),
    IAWAFeatureDef(30, "Vessel-ray pits with reduced borders",
                   CATEGORY_VESSEL_PITS,
                   "Vessel-to-ray pits simple or nearly simple (borders strongly reduced)."),

    # Vessel Dimensions
    IAWAFeatureDef(41, "Vessel diameter 50–100 µm",
                   CATEGORY_VESSELS,
                   "Mean tangential lumen diameter between 50 and 100 µm."),
    IAWAFeatureDef(42, "Vessel diameter 100–200 µm",
                   CATEGORY_VESSELS,
                   "Mean tangential lumen diameter between 100 and 200 µm."),
    IAWAFeatureDef(49, "40–100 vessels per mm²",
                   CATEGORY_VESSELS,
                   "Vessel frequency between 40 and 100 per square millimetre."),
    IAWAFeatureDef(50, "< 5 vessels per mm²",
                   CATEGORY_VESSELS,
                   "Very low vessel density – fewer than 5 per square millimetre."),
    IAWAFeatureDef(53, "Vessel element length 350–800 µm",
                   CATEGORY_VESSELS,
                   "Mean vessel element length between 350 and 800 µm."),

    # Fibers
    IAWAFeatureDef(61, "Fibres with simple pits",
                   CATEGORY_FIBERS,
                   "Simple to minutely bordered pits present on fiber walls."),
    IAWAFeatureDef(62, "Fibres with bordered pits",
                   CATEGORY_FIBERS,
                   "Distinctly bordered pits on fiber walls."),
    IAWAFeatureDef(65, "Septate fibres present",
                   CATEGORY_FIBERS,
                   "Fibres divided by transverse cross-walls (septa)."),

    # Axial Parenchyma
    IAWAFeatureDef(75, "Axial parenchyma absent or extremely rare",
                   CATEGORY_PARENCHYMA,
                   "Virtually no axial parenchyma cells visible in cross-section."),
    IAWAFeatureDef(79, "Vasicentric axial parenchyma",
                   CATEGORY_PARENCHYMA,
                   "Parenchyma forming a narrow sheath directly surrounding each vessel."),

    # Rays
    IAWAFeatureDef(96,  "All rays 1–3 seriate",
                   CATEGORY_RAYS,
                   "All rays uniseriate, or up to 3 cells wide."),
    IAWAFeatureDef(98,  "Larger rays 4–10 seriate",
                   CATEGORY_RAYS,
                   "The largest rays are commonly 4 to 10 cells wide."),
    IAWAFeatureDef(102, "Rays heterocellular",
                   CATEGORY_RAYS,
                   "Rays composed of both upright/square and procumbent cells."),
    IAWAFeatureDef(107, "Ray cells procumbent, 2–4 marginal rows",
                   CATEGORY_RAYS,
                   "Body ray cells procumbent with mostly 2–4 rows of upright/square marginal cells."),
    IAWAFeatureDef(115, "Ray frequency 4–12 per mm",
                   CATEGORY_RAYS,
                   "4 to 12 rays per millimetre."),

    # Crystals
    IAWAFeatureDef(136, "Prismatic crystals present",
                   CATEGORY_CRYSTALS,
                   "Box-shaped prismatic crystals present in axial parenchyma or ray cells."),
]

FEATURE_BY_ID: Dict[int, IAWAFeatureDef] = {f.id: f for f in IAWA_FEATURES}
