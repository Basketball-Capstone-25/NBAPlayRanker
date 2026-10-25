"""Where each Synergy play type usually ends: shot-zone weights and labels.

Shared by the court-map renderer and Quick-Call Mode so both describe the
same zones for a play type.
"""

from __future__ import annotations

from typing import Dict

PLAYTYPE_ZONES: Dict[str, Dict[str, float]] = {
    "Spotup": {"corner3": 1.0, "arc3": 0.8},
    "Cut": {"rim": 1.0},
    "Transition": {"rim": 0.9, "corner3": 0.6},
    "Isolation": {"mid": 0.8, "rim": 0.4},
    "OffScreen": {"arc3": 0.9, "mid": 0.4},
    "HandOff": {"arc3": 0.7, "mid": 0.4},
    "PostUp": {"paint": 1.0},
    "P&R Ball Handler": {"rim": 0.7, "mid": 0.5, "arc3": 0.4},
    "P&R Roll Man": {"rim": 1.0, "paint": 0.6},
}

DEFAULT_ZONES: Dict[str, float] = {"rim": 0.6, "arc3": 0.4}

ZONE_LABELS: Dict[str, str] = {
    "rim": "At the rim",
    "paint": "Paint",
    "mid": "Mid-range",
    "corner3": "Corner 3",
    "arc3": "Above-the-break 3",
}


def _norm(value: str) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def zones_for_playtype(play_type: str) -> Dict[str, float]:
    """Zone weights for a play type; Synergy spellings vary ("Spot Up", "Spotup")."""
    pt = _norm(play_type)
    for key, zones in PLAYTYPE_ZONES.items():
        if _norm(key) in pt:
            return zones
    return DEFAULT_ZONES


def best_shot_zone(play_type: str) -> str:
    """Readable label of the highest-weight zone for a play type."""
    zones = zones_for_playtype(play_type)
    zone = max(zones, key=zones.get)
    return ZONE_LABELS.get(zone, zone)
