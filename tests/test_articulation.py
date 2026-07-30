"""The bearing table: what an engineer knows, turned into frames."""
from __future__ import annotations

import pytest

from seismic_column.articulation import NONE, derive, describe, to_table

# The EN bridge: simple spans, one continuous frame A7->A10 on free bearings at
# BOTH ends, simple spans again.
EN_NAMES = [f"A{i}" for i in range(1, 19)]
EN_BACK = [NONE] + ["roller"] * 5 + ["roller", "integral", "integral",
                                     "roller"] + ["pin"] * 8
EN_AHEAD = ["pin"] * 6 + ["roller", "integral", "integral",
                          "roller"] + ["roller"] * 7 + [NONE]


def test_integral_both_ways_is_what_makes_a_frame_continuous():
    frames, links = derive(EN_NAMES, EN_BACK, EN_AHEAD)
    by = dict(zip(EN_NAMES, frames))
    # A8 and A9 are monolithic, so they do not interrupt the deck: the three
    # gaps A7-A8-A9-A10 fuse into ONE frame
    assert by["A8"] == by["A9"]
    assert "," not in by["A8"]                     # one deck, named once
    cont = by["A8"]
    assert cont in by["A7"] and cont in by["A10"]  # bounded by A7 and A10
    assert cont not in by["A6"] and cont not in by["A11"]


def test_a_bearing_pier_names_the_deck_either_side_of_the_joint():
    frames, links = derive(EN_NAMES, EN_BACK, EN_AHEAD)
    by_f, by_l = dict(zip(EN_NAMES, frames)), dict(zip(EN_NAMES, links))
    assert by_f["A7"].count(",") == 1              # two decks
    assert by_l["A7"] == "bearing, bearing"        # roller to both
    assert by_l["A6"] == "bearing, pinned"         # roller behind, pin ahead
    assert by_l["A11"] == "pinned, bearing"


def test_the_ends_of_the_bridge_name_one_deck_only():
    frames, links = derive(EN_NAMES, EN_BACK, EN_AHEAD)
    assert "," not in frames[0] and "," not in frames[-1]
    assert links[0] == "pinned" and links[-1] == "pinned"


def test_every_gap_becomes_exactly_one_frame():
    frames, _ = derive(EN_NAMES, EN_BACK, EN_AHEAD)
    keys = {k for f in frames for k in f.split(", ") if k}
    assert len(keys) == 15          # 17 gaps, 3 of them fused into one


def test_a_run_of_simple_spans_gives_one_frame_per_gap():
    names = ["P1", "P2", "P3", "P4"]
    frames, links = derive(names, [NONE, "roller", "roller", "roller"],
                           ["pin", "pin", "pin", NONE])
    assert frames == ["F1", "F1, F2", "F2, F3", "F3"]
    assert links == ["pinned", "bearing, pinned", "bearing, pinned", "bearing"]


def test_an_all_integral_run_is_a_single_frame():
    names = ["P1", "P2", "P3"]
    frames, links = derive(names, [NONE, "integral", "integral"],
                           ["integral", "integral", NONE])
    assert frames == ["F1", "F1", "F1"]
    assert links == ["integral", "integral", "integral"]


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        derive(["P1", "P2"], ["pin"], ["pin", "pin"])


def test_describe_reads_a_row_back_as_pairs():
    assert describe("F6, F7", "bearing, bearing") == [("F6", "bearing"),
                                                      ("F7", "bearing")]
    assert describe("C1", "integral") == [("C1", "integral")]


def test_the_table_round_trips_through_derive():
    """Seeding the bearing table from existing cells and deriving again must
    give back the same frames -- otherwise opening a project would silently
    rewrite it."""
    frames, links = derive(EN_NAMES, EN_BACK, EN_AHEAD)
    back, ahead = to_table(EN_NAMES, frames, links)
    again, again_l = derive(EN_NAMES, back, ahead)
    assert again == frames
    assert again_l == links
