"""
IAWA feature simulator for wood anatomy analysis.

Generates plausible IAWA feature profiles for known tropical tree species.
Profiles are species-aware but image-agnostic – the goal is to demonstrate
the identification workflow and collaborative learning loop, not to perform
real computer-vision-based anatomy extraction (deferred to future work).
"""

import random
from typing import Dict, List, Optional, Tuple

from .iawa_features import IAWA_FEATURES, FEATURE_BY_ID, FeatureResult

# ── Type alias ─────────────────────────────────────────────────────────────────
# Maps feature_id → True (present) | False (absent) | None (uncertain)
Profile = Dict[int, Optional[bool]]

# ── Helper ─────────────────────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    """Normalise species name for dictionary lookup (spaces → underscores, title-case)."""
    return name.strip().replace(" ", "_")


# ── Default profile for tropical hardwoods ────────────────────────────────────
# Represents a "generic Amazonian diffuse-porous hardwood".
_DEFAULT: Profile = {
    1:   False,   # growth rings indistinct → typical for tropical
    2:   True,
    5:   True,    # diffuse-porous
    6:   False,
    13:  True,    # simple perforation plates
    14:  False,
    16:  False,
    17:  False,
    20:  False,
    21:  False,
    22:  True,    # alternate intervessel pits
    30:  True,    # vessel-ray pits reduced
    41:  True,    # 50–100 µm vessels
    42:  False,
    49:  True,    # 40–100 vessels/mm²
    50:  False,
    53:  True,    # 350–800 µm vessel length
    61:  True,    # simple fiber pits
    62:  False,
    65:  False,
    75:  False,
    79:  True,    # vasicentric parenchyma
    96:  False,
    98:  True,    # rays 4–10 seriate
    102: True,    # heterocellular rays
    107: True,
    115: True,    # 4–12 rays/mm
    136: False,
}

# ── Per-species profiles  (only differences from _DEFAULT need listing) ────────
# Keys: normalised species name.  Values: feature_id → bool override.
_OVERRIDES: Dict[str, Profile] = {
    "Apuleia_molaris": {
        1: True, 2: False,       # visible growth rings
        136: False,
    },
    "Aspidosperma_populifolium": {
        42: False, 41: True,
        49: True,
        75: True, 79: False,     # parenchyma almost absent
        96: True, 98: False,     # narrow rays
        136: False,
    },
    "Astronium_gracile": {
        1: True, 2: False,
        79: True,
        136: True,               # crystals in Anacardiaceae
    },
    "Byrsonima_coriaceae": {
        41: True, 42: False,
        49: True,
        136: False,
    },
    "Calophyllum_brasiliensis": {
        42: True, 41: False,     # larger vessels
        50: True, 49: False,     # low density
        30: True,
        65: True,                # septate fibers
        136: False,
    },
    "Cecropia_glaziovii": {
        42: True, 41: False,
        50: True, 49: False,
        75: True, 79: False,
        96: True, 98: False,
    },
    "Cecropia_sciadophylla": {
        42: True, 41: False,
        50: True, 49: False,
        75: True, 79: False,
        96: True, 98: False,
    },
    "Cedrelinga_catenaeformis": {
        42: True, 41: False,
        50: True, 49: False,
        136: False,
    },
    "Cochlospermum_orinoccense": {
        20: True, 22: False,     # scalariform pits
        14: True, 13: False,     # scalariform perforations
        16: True,
        65: True,
    },
    "Combretum_leprosum": {
        1: True, 2: False,
        136: True,               # crystals
        79: True,
    },
    "Copaifera_langsdorfii": {
        1: True, 2: False,
        79: True,
        136: False,
    },
    "Croton_argyrophylloides": {
        41: True, 42: False,
        49: True,
        75: True, 79: False,
        96: True, 98: False,
    },
    "Diplotropis_purpurea": {
        1: True, 2: False,
        79: True,
        136: False,
    },
    "Dipteryx_odorata": {
        1: False, 2: True,
        41: True, 42: False,
        136: True,               # characteristic crystals
        79: True,
    },
    "Enterolobium_schomburgkii": {
        1: True, 2: False,
        42: True, 41: False,
        50: True, 49: False,
        79: True,
    },
    "Erisma_uncinatum": {
        65: True,                # septate fibers common in Vochysiaceae
        53: True,
        98: True,
        136: False,
    },
    "Goupia_glabra": {
        75: True, 79: False,     # parenchyma very sparse
        62: True, 61: False,     # bordered fiber pits
        136: False,
    },
    "Hieronyma_laxiflora": {
        1: True, 2: False,
        79: True,
        136: False,
    },
    "Hymenaea_courbaril": {
        1: True, 2: False,
        79: True,
        136: True,               # crystals in Fabaceae
    },
    "Hymenolobium_petraeum": {
        1: True, 2: False,
        79: True,
        136: False,
    },
    "Jacaranda_copaia": {
        21: True, 22: False,    # opposite pits (Bignoniaceae character)
        65: True,               # septate fibers
        136: False,
    },
    "Jatropha_mutabilis": {
        42: True, 41: False,
        50: True, 49: False,
        75: True, 79: False,
    },
    "Licaria_cannela": {
        21: True, 22: False,    # opposite pits (Lauraceae)
        41: True, 42: False,
        49: True,
        65: True,               # septate fibers
    },
    "Luitzelburgia_auriculata": {
        41: True, 42: False,
        79: True,
        136: False,
    },
    "Mezilaurus_itauba": {
        21: True, 22: False,    # opposite pits (Lauraceae)
        41: True, 42: False,
        49: True,
        65: True,               # septate fibers
    },
    "Mimosa_scabrella": {
        1: True, 2: False,
        41: True, 42: False,
        79: True,
        136: False,
    },
    "Ocotea_leucoxylon": {
        14: True, 13: False,    # scalariform perforations
        16: True,
        21: True, 22: False,    # opposite pits
        41: True, 42: False,
        65: True,               # septate fibers
    },
    "Ocotea_odorifera": {
        14: True, 13: False,
        16: True,
        21: True, 22: False,
        41: True, 42: False,
        65: True,
    },
    "Ocotea_porosa": {
        14: True, 13: False,
        16: True, 17: False,
        21: True, 22: False,
        41: True, 42: False,
        49: True,
        65: True,
    },
    "Parkia_pendula": {
        42: True, 41: False,
        50: True, 49: False,
        79: True,
        136: False,
    },
    "Pera_glabrata": {
        41: True, 42: False,
        49: True,
        75: True, 79: False,
        96: True, 98: False,
    },
    "Piptadenia_communis": {
        41: True, 42: False,
        49: True,
        79: True,
        136: False,
    },
    "Poeppigia_procera": {
        79: True,
        136: False,
    },
    "Poincianella_bracteosa": {
        1: True, 2: False,
        41: True, 42: False,
        79: True,
        136: False,
    },
    "Qualea_paraensis": {
        65: True,               # septate fibers (Vochysiaceae)
        42: True, 41: False,
        79: True,
    },
    "Sapium_glandulatum": {
        42: True, 41: False,
        75: True, 79: False,
        96: True, 98: False,
    },
    "Schefflera_morototoni": {
        42: True, 41: False,
        50: True, 49: False,
        75: True, 79: False,
    },
    "Sclerolobium_paniculatum": {
        79: True,
        136: False,
    },
    "Tabebuia_alba": {
        1: True, 2: False,
        79: True,
        65: True,               # septate fibers (Bignoniaceae)
    },
    "Trattinnickia_burseraefolia": {
        79: True,
        136: False,
    },
    "Vatairea_guianensis": {
        1: True, 2: False,
        79: True,
        136: False,
    },
    "Vatairea_paraensis": {
        1: True, 2: False,
        79: True,
        136: False,
    },
    "Vochysia_densiflora": {
        65: True,               # septate fibers
        42: True, 41: False,
        79: True,
        136: False,
    },
    "Vochysia_maxima": {
        65: True,
        42: True, 41: False,
        50: True, 49: False,    # larger vessels, lower density
        79: True,
        136: False,
    },
}


def _build_profile(species_key: str) -> Profile:
    """Merge the default tropical profile with species-specific overrides."""
    profile = dict(_DEFAULT)
    overrides = _OVERRIDES.get(species_key, {})
    profile.update(overrides)
    return profile


def generate_features(species_name: str) -> List[FeatureResult]:
    """
    Generate a plausible IAWA feature profile for *species_name*.

    Values are mock/simulated – confidence scores are randomised around
    species-typical baselines to emulate real detection uncertainty.
    """
    key = _normalise(species_name)
    profile = _build_profile(key)

    results: List[FeatureResult] = []
    for feat_def in IAWA_FEATURES:
        presence = profile.get(feat_def.id)

        if presence is True:
            is_present = True
            confidence = round(random.uniform(0.76, 0.97), 3)
        elif presence is False:
            is_present = False
            confidence = round(random.uniform(0.80, 0.96), 3)
        else:
            # Uncertain – model is unsure
            is_present = random.choice([True, False])
            confidence = round(random.uniform(0.42, 0.63), 3)

        results.append(FeatureResult(
            id=feat_def.id,
            name=feat_def.name,
            category=feat_def.category,
            description=feat_def.description,
            is_present=is_present,
            confidence=confidence,
        ))

    return results


def compute_feature_species_support(
    detected: List[FeatureResult],
    predictions: List[dict],
) -> Dict[str, dict]:
    """
    For each predicted species, determine which detected features match or
    mismatch the expected species profile.

    Returns a dict keyed by species label:
        {
            "matched": [feature_id, ...],
            "mismatched": [feature_id, ...],
        }
    """
    support: Dict[str, dict] = {}

    for pred in predictions[:5]:          # only top-5
        label = pred.get("label", "")
        key   = _normalise(label)
        profile = _build_profile(key)

        matched: List[int]    = []
        mismatched: List[int] = []

        for feat in detected:
            expected = profile.get(feat.id)
            if expected is None:
                continue               # unknown for this species – skip
            if feat.is_present == expected:
                matched.append(feat.id)
            else:
                mismatched.append(feat.id)

        support[label] = {"matched": matched, "mismatched": mismatched}

    return support
