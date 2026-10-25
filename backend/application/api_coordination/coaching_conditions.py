"""The coaching inputs supported by the play-type ranker."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class CoachingConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    need: Optional[Literal["need3", "quick2", "two_for_one", "last_shot", "must_score", "safe"]] = None
    defense_style: Optional[Literal["switch", "drop", "hedge", "blitz", "ice", "under", "top_lock", "zone_2_3", "zone_3_2", "zone_1_3_1", "box_and_1", "matchup_zone", "generic_zone"]] = None
    pace: Optional[Literal["push", "slow"]] = None
    shot_clock: Optional[float] = Field(None, ge=0, le=24)
    after_timeout: StrictBool = False
    late_clock: StrictBool = False
    need3: StrictBool = False
    protect_lead: StrictBool = False
    vs_switching: StrictBool = False
    must_stop: StrictBool = False
    quick2: StrictBool = False
    two_for_one: StrictBool = False
    slob: StrictBool = False
    blob: StrictBool = False
    advance_ball: StrictBool = False
    hold_for_last: StrictBool = False
    safe: StrictBool = False
    must_score: StrictBool = False
    no_three: StrictBool = False
