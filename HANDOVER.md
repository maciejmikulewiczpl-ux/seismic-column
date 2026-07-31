# Handover — seismic column / balance tool

Written 2026-07-31. This is the part that is **not** obvious from the code: why
the frame model is shaped the way it is, which rules apply where, what the two
live structures currently look like, and the traps that have already cost real
time. Read this before changing anything in `balance.py`, `articulation.py` or
the silo search in `batch.py`.

Scope: Caltrans SDC 2.1 / AASHTO SGS 3rd Ed., SDC D only. Code differences are
switches on `CodeProvisions` (`provisions.py`), never replacements.

---

## 1. The frame model

**One idea underpins everything: bearings define decks, and a deck IS a frame.**
AASHTO SGS §5.1 — *"a single frame or a series of frames separated by expansion
joints, articulated construction joints, or both."*

A deck runs between the bearings that carry it and passes **through** a bent
only if that bent is `integral` on both sides. Integral both sides means the
deck does not stop; anything else is a joint.

### Input

`frame` takes one or more names — `"F1, F2"`. A pier under an expansion joint
carries the last span of one frame and the first span of the next, so it names
**both**.

`deck_link` takes **one entry per frame, in the same order**:

```
frame     = "F6, C1"
deck_link = "pinned, bearing"     # pinned to F6, free bearing to C1
```

A single value is broadcast to every frame, so older tables mean what they
always meant. Engineer's vocabulary is accepted and preserved —
`roller` / `expansion` / `pin` / `fixed` / `monolithic` (`DECK_LINK_ALIASES`).

`articulation.py` derives both cells from a **bearing table** — two dropdowns
per bent (`deck behind`, `deck ahead`) of pin / roller / integral / —. That is
the intended way to enter articulation; zipping two comma-lists by hand is not.
`derive()` builds the cells, `to_table()` seeds the editor back from them, and
the round trip is tested.

### Frames are rebuilt PER DIRECTION

- **Transversely** every bearing is shear-keyed, so `pinned` and `bearing` are
  **identical** — same participation, same tributary mass, same fixed-free
  head. Only `integral` differs. **No pin↔roller change can move a transverse
  number.** (Verified: flipping a bearing leaves transverse frames and masses
  byte-identical.)
- **Longitudinally** only `pinned` and `integral` resist. A bent that rollers
  both sides carries **no deck** — only its own column self weight — so it
  leaves the longitudinal model entirely. Its own seismic checks still run, on
  its own stand-alone period.

### The joint bent

Appears in **both** frames it carries, with **full stiffness in each** (each
frame is analysed alone, and a rigid deck leans on the whole bent) and its
tributary mass **divided between the frames it resists in that direction**.
The entered transverse mass is already the sum of two half-spans, so a 50/50
split is exact where the adjacent spans match.

### Validation (`io_schema.validate`)

- A **two-support** frame needs exactly one pin and one roller. Pin-pin locks
  the span against temperature; roller-roller leaves it unrestrained.
- Free bearings at **both** ends of a **continuous** frame are fine — its
  integral bents hold it. (That is C1 on structure A.)
- The **first and last** decks may roller both in-model supports: they are held
  at an abutment, which is never a table row. Recorded as a migration note.
- A missing `frame` column means one frame per bent (simply supported), not one
  giant continuous deck.

Abutments are **not modelled** — they carry passive soil resistance the tool
does not represent. Structure B's B1 and B23 were removed for this reason.

---

## 2. What gets checked, and where

Both rules are evaluated **in each direction** on the frames above.

### Balanced stiffness — INSIDE one frame

`κ = k/m` between bents sharing a deck. Adjacent **0.75**, any two **0.50**.
Caltrans SDC §7.1.2 says *shall*; AASHTO SGS §4.1.2 says *recommended*. Fires
only where a frame has two or more members **in that direction**.

**The simple-span exception** — `balance_simple_span_stiffness`, default
**OFF**. A frame that is `simply_supported` (**exactly two supports, neither
integral**) is matched on period alone unless the switch is on.

*Exactly two* matters: a girder continuous over three or more bents on bearings
is a continuous frame that merely is not integral, and its bents must balance.

The code basis is genuinely ambiguous, which is why this is a switch and not a
decision baked in:

| clause | says |
|---|---|
| SGS §5.1 | each simply supported span is a frame → its two supports are "two bents within a frame" → the rule applies |
| SGS §5.1.1 | multi-simple-span bridges **may** use ESA per bent; "global analysis requirements … need not be applied" → per-bent SDOF is explicitly sanctioned |
| SGS §4.2 | applies Eqs 4.1.2-1..4 "from span-to-span or from support-to-support" — but as a *regularity* screen selecting the analysis procedure, not a proportioning requirement |
| practice | no published worked example applies it across a run of simple spans; MoDOT EPG 751.9 specifies no stiffness-ratio check between adjacent bents at all |

### Balanced frame geometry — BETWEEN adjacent frames

`min(Ti,Tj) / max(Ti,Tj) ≥ 0.70`, everywhere, simple spans included.

### A geometry shortfall is REFERRED, not cleared

> **Caltrans C7.1.3: "The use of NTHA is not by itself a justification to waive
> the requirement of balanced frame geometry."**

The tool previously cited SDC as though time-history cleared it. That was
wrong. Accepting a shortfall is a **project-criteria decision for the owner**;
the tool now states the pair, the ratio and the magnitude so the decision is
made explicitly. **Within-frame stiffness is never referred** — it governs how
a frame distributes demand among its own members and must be proportioned.

### Silo is the tuning lever

Auto-silo softens stiff piers until both rules hold, then re-runs the **full**
seismic suite, because a longer free column changes `Lp`, the displacement
demand, `Vo`, `Vp` and P-Δ. It also sizes for a pier's own short-column shear
as a floor — that, not balance, is what drives structure B's depths.

---

## 3. Traps

Every one of these produced plausible-looking wrong numbers.

**Per-COLUMN vs per-BENT.** `BentStiffness.k` is stored per column;
`stiffness()` returns `n_columns × k`; `mass()` is per bent. The silo
predictor once returned per-column while the rule it planned against was
per-bent, so a 2-column bent beside a 3-column one *looked* balanced at 0.85
while the real check failed at 0.56 — and the search never softened anyone. It
hides because the two agree wherever neighbours have equal column counts.
Assert the predictor reproduces `b.stiffness(...)` at the bent's own silo.

**Stages that re-run rows must carry the frame basis.** `_frame_basis` gives
each pier its FRAME's `(K, W)` and end condition. The shear-silo stage ran
*after* the frame stage and re-ran rows bare, reverting them to stand-alone
cantilevers — one pier read `end=free` when it was fixed, `T` 1.618 s instead
of 0.870 s, and failed displacement capacity falsely. Anything calling
`run_row` after `_frame_demand_stage` must pass `demand_basis` and
`end_fixity`.

**The auto-silo search is greedy, monotone and path-dependent.** It never takes
silo back off, so where it starts changes where it lands — same articulation
and sections gave 50 ft / 2 THA from the saved silos and 77 ft / 3 THA from
zero. Always compare like-for-like starting points, and say which you used.

**The optimiser will overwrite a swept diameter.**
`DEFAULT_VARIABLE = ("longitudinal", "confinement", "diameter", "fc")`. Sweeping
diameters with `optimize=True` collapses every candidate to one answer. Strip
`"diameter"` and `"shaft_diameter"` from `cfg.variable` when the grid owns
them. This has cost hours more than once.

**A search must score against the incumbent.** A sweep comparing candidates only
with each other will save the best of a bad set over a better baseline — it
once wrote a 173 ft, three-infeasible design over a working 63 ft one.
Evaluate the existing design first and never write unless a candidate beats it.

**Budget: optimiser runs cost ~10-15 min, bare runs ~3.** Screen wide with the
cage as entered, then refine the best few with the optimiser. A 180-candidate
optimised sweep is 30 hours, not overnight.

**Test contracts encode the OLD behaviour.** Five tests broke while the frame
model changed — all asserting superseded behaviour, none finding a bug. Watch
for fixtures defaulting to `deck_link="pinned"` while *meaning* a continuous
frame (under the two-support rule that reads as a simply supported span and
silently loses its stiffness checks), and for tests targeting the FIRST frame
with a roller-roller error, which is the legitimate abutment-held case.
**Run the full suite before pushing** — the fast suite missed three of these.

**Two sources for one number.** Recurring: a mechanism shear computed in two
places with only one getting a correction; `_frame_basis` letting whichever
frame was processed last win for a shared bent. Derive once.

---

## 4. The two live structures

Project `.json` files are **gitignored** (`seismic_project_A*.json`,
`seismic_project_B*.json`) and live only on the engineer's disk. They are not
in the repo and not backed up by it.

| file | rule | rows | hard | THA | silo | col/shaft |
|---|---|---|---|---|---|---|
| `seismic_project_A_BEST_simplespan_OFF.json` | off | 18/18 | 0 | 1 | **46 ft** | 90/118, C1 78/150 |
| `seismic_project_A_BEST_simplespan_ON.json` | on | 18/18 | 0 | 1 | 64 ft | 90/144, C1 78/150 |
| `seismic_project_B_BEST_min_concrete.json` | n/a | 21/21 | 0 | **0** | 63 ft | 66/114 |
| `seismic_project_B_BEST_min_silo.json` | n/a | 21/21 | 0 | **0** | **45 ft** | 78/162 |

### A — 18 bents, 14 simple spans plus one continuous frame

Articulation, entered as bearings:

```
A1 pin | A2-A6 roller,pin | A7 roller,roller | A8-A9 integral
       | A10 roller,roller | A11-A17 pin,roller | A18 roller
```

C1 = F7 = A7…A10, free bearings at both ends, integral at A8/A9. Pins face
outward from C1 on both sides — that layout is what removed the longitudinal
referral. `W_long = 0` on A7, A10 and A18 (roller only). The last span F15 is
held at abutment **A19**, not a table row. A1 is a tie-in to an existing
structure.

**The surviving referral is structural.** Transverse `F6–F7` ≈ 0.62–0.65: C1 is
a four-bent frame carrying far more mass than the single spans beside it, so a
longer transverse period is intrinsic to it. Three levers were tried and none
closes it:

- siloing A6 fixes it transversely but breaks it longitudinally — A6 is F6's
  *only* longitudinal support but one of *two* transverse supports, so the two
  directions pull opposite ways on the same bent;
- stiffening C1's sections is self-defeating, because the auto-silo adds depth
  back as fast as the section adds stiffness (45 → 68 ft for 0.645 → 0.669);
- A7/A10 is the only transverse-only lever — both roller each side, so they sit
  in C1 transversely and are absent longitudinally — and it tops out at 0.664.

### B — 21 bents, continuous frames separated by roller bents

`cap_fixity = "pinned_trans"` throughout, adopted to remove the push/pull
tension question (the couple equals the sum of the column TOP moments and
saturates at ΣMo; only spacing and Mo move it). Silo is almost entirely
**short-column shear**, not balance.

**The simple-span switch is a no-op on B** — verified identical both ways.
Every B frame is continuous (3–4 members with integral bents), none is
`simply_supported`, and the roller bents are already stiffness-checked against
**both** frames they carry, across all 36 pairs.

### Open decisions

1. **Is the simple-span stiffness rule in scope for A?** Off 46 ft; on 64 ft
   *plus* all 14 simple-span shafts 118 → 144 in (~50% more shaft concrete).
   It does not improve the referral. Code basis ambiguous — see §2.
2. **B: silo depth against shaft concrete.** 45 ft on 162 in shafts, or 63 ft
   on 114 in — 18 ft of drilling against roughly double the shaft section on 21
   bents. Not priceable without site costs.
3. **B's bearing bents carry ~759 kip of `W_long`** (cap + column) while the
   run separately adds ⅓ of the column self weight — about 6% double-counted,
   conservative. A's equivalents are 0, which is the cleaner convention.
4. **Shaft diameter behaves oppositely in the two cases.** Irrelevant to A with
   the rule off, and to B (silo is shear-driven); but the dominant lever with
   the rule on — 118 → 168 in takes A from 104 ft to 67 ft — because it evens
   out `Df` between bents of differing height. Column diameter cannot do this:
   `κ = k/m`, and raising `EI` on both bents of a pair leaves their ratio
   exactly where it was.
