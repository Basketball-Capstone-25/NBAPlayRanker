from fastapi import HTTPException

from domain.shot_analysis.shot_quality import shot_data_quality


def require_valid_shot_data() -> None:
    if not shot_data_quality().get("valid_attribution"):
        raise HTTPException(409, "Shot results are unavailable while the source team labels are being repaired. Play-type rankings are still available.")
