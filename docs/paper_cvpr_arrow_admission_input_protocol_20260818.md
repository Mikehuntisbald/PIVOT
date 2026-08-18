# ARROW Admission-input protocol

> **Status:** complete. Results and sealed hashes are reported in
> `paper_cvpr_arrow_admission_input_results_20260818.md`.

ARROW stands for **Admission, Ranking, and Rejection with Ownership-Separated
Weights for Reliable Visual Grounding**. This block changes only the Admission
input while retaining the sealed U2-v5 implementation lineage.

| Internal row | Paper label | Admission input |
|---|---|---|
| `AR_A_PATCH` | A | current support patch |
| `AR_B_TEXT` | B | explicit canonical category text |
| `AR_C_NULL` | C | learned category-agnostic null token |

All rows use the same legacy surface8 and training-only auxiliary8. Full
referring expressions continue to own B58 query geometry and frozen R100
ranking; detached confidence12 remains independent. B pools frozen raw BERT
phrase tokens into a 768-d source and tiles it to 7×7. C uses a deterministic,
SHA-bound dense sentinel; its projected 256-d representation is learned through
the same surface8. Neither B nor C consumes a support patch in the model.

A reuses the sealed three-seed ARROW main route. B/C each train seeds 17/42/73
for exactly U100 at physical B56 from the same clean initializer and dataset.
Gap3, loss weights, optimizer exposure, augmentations and ownership are fixed;
there is no milestone selection.

The primary endpoint is a fresh 512-image LVIS category-switch panel that is
image-disjoint from admission train, Ref8 and strict2031. Each category pair
uses one fixed joint geometry caption and query universe. Planned one-sided
contrasts are A>B and B>C with 5,000 pair-cluster bootstrap replicates and Holm
correction. Val3 is mechanistic. Test5 is a prospectively frozen post-release
no-collapse analysis at margin 0.005. No strict forward is permitted; clean D3
confidence is reused only after exact record parity.

Interpretation is fixed before training: A>B is required for a visual-exemplar
claim; B>C is required for an explicit-category-condition claim. If A≈B, the
support patch is one interchangeable Admission implementation. If C retains Ref
accuracy but cannot switch categories, it is a generic query prior rather than
a controllable category route. None of these outcomes changes the sealed ARROW
main model.
