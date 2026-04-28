"""contracts.vision.is_ethnic_outfit / is_canonical_ethnic helper pinning.

canonical_extractor._select_wear_class_ids 의 ethnic 분기와 정확히 일치하는지 검증
(single source of truth — 2026-04-28).
"""
from __future__ import annotations

import pytest

from contracts.vision import (
    CanonicalOutfit,
    EthnicOutfit,
    OutfitMember,
    is_canonical_ethnic,
    is_ethnic_outfit,
)


def _outfit(
    *,
    upper_is_ethnic: bool | None,
    lower_is_ethnic: bool | None,
    dress_as_single: bool,
) -> EthnicOutfit:
    return EthnicOutfit(
        person_bbox=(0.1, 0.1, 0.5, 0.7),
        person_bbox_area_ratio=0.35,
        upper_garment_type="x",
        upper_is_ethnic=upper_is_ethnic,
        lower_garment_type=None if dress_as_single else "y",
        lower_is_ethnic=lower_is_ethnic,
        dress_as_single=dress_as_single,
    )


# --------------------------------------------------------------------------- #
# is_ethnic_outfit — dress_as_single 분기

@pytest.mark.parametrize(
    ("upper", "expected"),
    [(True, True), (False, False), (None, False)],
)
def test_dress_as_single_uses_only_upper(
    upper: bool | None, expected: bool
) -> None:
    rep = _outfit(upper_is_ethnic=upper, lower_is_ethnic=None, dress_as_single=True)
    assert is_ethnic_outfit(rep) is expected


# --------------------------------------------------------------------------- #
# is_ethnic_outfit — 2-piece OR 분기

@pytest.mark.parametrize(
    ("upper", "lower", "expected"),
    [
        (True, True, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
        (None, None, False),
        (True, None, True),
        (None, True, True),
        (False, None, False),
        (None, False, False),
    ],
)
def test_two_piece_uses_or(
    upper: bool | None, lower: bool | None, expected: bool
) -> None:
    rep = _outfit(
        upper_is_ethnic=upper, lower_is_ethnic=lower, dress_as_single=False
    )
    assert is_ethnic_outfit(rep) is expected


# --------------------------------------------------------------------------- #
# is_canonical_ethnic — representative 위임

def test_canonical_ethnic_delegates_to_representative() -> None:
    rep = _outfit(
        upper_is_ethnic=True, lower_is_ethnic=False, dress_as_single=False
    )
    canonical = CanonicalOutfit(
        canonical_index=0,
        representative=rep,
        members=[OutfitMember(image_id="a", outfit_index=0,
                              person_bbox=(0.1, 0.1, 0.5, 0.7))],
    )
    assert is_canonical_ethnic(canonical) is True


def test_canonical_non_ethnic_returns_false() -> None:
    rep = _outfit(
        upper_is_ethnic=False, lower_is_ethnic=False, dress_as_single=False
    )
    canonical = CanonicalOutfit(
        canonical_index=0,
        representative=rep,
        members=[OutfitMember(image_id="a", outfit_index=0,
                              person_bbox=(0.1, 0.1, 0.5, 0.7))],
    )
    assert is_canonical_ethnic(canonical) is False
