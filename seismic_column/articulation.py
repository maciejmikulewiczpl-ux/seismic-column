"""Derive the frame layout from the bearings, one bent at a time.

Entering ``frame`` and ``deck_link`` as two comma-lists that have to be zipped
together in your head is the least readable part of the input.  What an engineer
actually knows is much simpler: **at each bent, how does each deck sit on it?**

    bent          A6      A7      A8       A9      A10     A11
    deck behind   roller  roller  integral integral roller  pin
    deck ahead    pin     roller  integral integral roller  roller

Everything else follows.  A deck runs between the bearings that carry it, and
passes THROUGH a bent that is integral on both sides -- that, and only that, is
what makes a frame continuous::

    A6 --D5-- A7 =========== D6 =========== A10 --D7-- A11
       pin/roller   roller  integral  roller   roller/pin

So the two continuous bents A8 and A9 fuse the three gaps A7-A8, A8-A9, A9-A10
into one deck bounded by free bearings at A7 and A10, and every other gap is a
simply supported span of its own.

:func:`derive` turns the bearing table into the ``frame`` and ``deck_link``
cells; :func:`describe` reads them back for display.
"""
from __future__ import annotations

from .io_schema import deck_links, frame_keys

#: what the bearing table offers, and what each choice stores
CHOICES: dict[str, str] = {
    "pin": "pinned",             # fixed bearing: restrains both directions
    "roller": "bearing",         # expansion bearing, shear-keyed transversely
    "integral": "integral",      # monolithic with the deck
    "—": "",                     # no deck this side (bridge ends, or off-model)
}
NONE = "—"


def _decks(back: list[str], ahead: list[str]) -> list[int | None]:
    """Deck id carried in each gap between consecutive bents.

    Gap ``i`` sits between bent ``i`` and bent ``i+1``.  Two gaps are the SAME
    deck when the bent between them is integral on both sides -- a monolithic
    bent does not interrupt the superstructure.
    """
    n_gaps = max(len(back) - 1, 0)
    deck: list[int | None] = [None] * n_gaps
    nxt = 0
    for g in range(n_gaps):
        # continuous with the previous gap?
        if (g > 0 and ahead[g] == "integral" and back[g] == "integral"):
            deck[g] = deck[g - 1]
        else:
            deck[g] = nxt
            nxt += 1
    return deck


def derive(names: list[str], back: list[str], ahead: list[str],
           prefix: str = "F") -> tuple[list[str], list[str]]:
    """``(frame cell, deck_link cell)`` for every bent, from its two bearings.

    A bent touches the deck behind it and the deck ahead of it.  Where those are
    the same deck (it is integral both ways) it names that deck once; where they
    differ it names both, which is exactly the expansion-joint case -- it carries
    the last span of one frame and the first of the next.
    """
    if not (len(names) == len(back) == len(ahead)):
        raise ValueError("names, back and ahead must be the same length")
    deck = _decks(back, ahead)
    label: dict[int, str] = {}
    for d in deck:
        if d is not None and d not in label:
            label[d] = f"{prefix}{len(label) + 1}"

    frames: list[str] = []
    links: list[str] = []
    for i, _name in enumerate(names):
        pairs: list[tuple[str, str]] = []
        for gap, choice in ((i - 1, back[i]), (i, ahead[i])):
            if choice == NONE or choice not in CHOICES or not CHOICES[choice]:
                continue
            if not (0 <= gap < len(deck)) or deck[gap] is None:
                continue
            key = label[deck[gap]]
            if any(k == key for k, _ in pairs):     # integral both ways: one deck
                continue
            pairs.append((key, CHOICES[choice]))
        frames.append(", ".join(k for k, _ in pairs))
        links.append(", ".join(v for _, v in pairs) or "pinned")
    return frames, links


def describe(frame: object, link: object) -> list[tuple[str, str]]:
    """Read a ``frame``/``deck_link`` pair back as ``[(frame, link), ...]``."""
    keys = frame_keys(frame)
    return list(zip(keys, deck_links(link, len(keys) or 1)))


def to_table(names: list[str], frames: list[object],
             links: list[object]) -> tuple[list[str], list[str]]:
    """Best-effort inverse of :func:`derive`, to seed the bearing table.

    A bent naming two decks meets the earlier one behind and the later one
    ahead.  A bent naming one deck is ambiguous on its own -- it could be the
    start or the end of that deck -- so the neighbours decide: the side facing
    a bent that shares the deck is the side that carries it.
    """
    word = {v: k for k, v in CHOICES.items() if v}
    back = [NONE] * len(names)
    ahead = [NONE] * len(names)
    seen = [describe(f, l) for f, l in zip(frames, links)]
    for i, pairs in enumerate(seen):
        if len(pairs) >= 2:
            back[i] = word.get(pairs[0][1], NONE)
            ahead[i] = word.get(pairs[-1][1], NONE)
        elif len(pairs) == 1:
            key, lk = pairs[0]
            w = word.get(lk, NONE)
            prev_has = i > 0 and any(k == key for k, _ in seen[i - 1])
            next_has = i + 1 < len(seen) and any(k == key for k, _ in seen[i + 1])
            if prev_has:
                back[i] = w
            if next_has:
                ahead[i] = w
            if not prev_has and not next_has:       # alone in its own frame
                back[i] = ahead[i] = w
    return back, ahead
