# Historical pre-ARROW implementation archive

This directory is the navigation boundary for pre-ARROW experiments. These
artifacts explain how the final ownership-separated design was reached; they
do **not** define the current paper model.

The current public model is **ARROW-U2**. See the
[`ARROW-U2 model card`](../arrow_u2_model_card.md).

## Why most files retain their original paths

Many historical documents are inputs to sealed receipts or are linked by
immutable experiment manifests. Moving or rewriting them would break paths,
hashes, and checkpoint provenance. They therefore remain at their original
locations, while all public navigation is quarantined through this index.

The complete pre-isolation Stage-A/B manual is preserved locally as the
[`legacy Stage-A/B README snapshot`](legacy_stage_ab_readme.md). Its source is
also immutable in Git history at
[`b77c8fa:README.md`](https://github.com/Mikehuntisbald/PIVOT/blob/b77c8fa45c5c5eae1566f1928805c4fd138af73a/README.md#historical-grounding-dino--stage-ab-implementation-notes).

## Stage-A and patch exploration

- [Stage-A caliber evaluation](../stage_a_caliber_eval.md)
- [Stage-A v3/v4 LVIS negatives](../stage_a_v3_v4_lvis_neg.md)
- [Stage-A/B runbook](../stage_ab_runbook.md)
- [B58 + checkpoint0006 realignment](../stage_a_b58_patch0006_realign_20260814.md)
- [Stage-A → R100 → C100 sealed diagnostic](../stagea_b58_r100_c100_sealed_pipeline_20260815.md)

## Stage-B and dense-duty versions

- [Stage-B CVPR development ledger, including V55/V56](../paper_cvpr_stage_b_development_20260802.md)
- [Architecture relative to Grounding DINO and routed-v3](../stage_b_architecture_vs_gdino.md)
- [Decoupled scoring handoff](../stage_b_decoupled_scoring_handoff_20260716.md)
- [Fixed-baseline protocol](../stage_b_fixed_baseline_protocol.md)
- [FPR95 failure analysis](../stage_b_fpr95_failure_analysis_20260711.md)
- [Historical paper evaluation runbook](../stageb_paper_evaluation_runbook.md)
- [Historical serial matrix queue](../stageb_serial_matrix_queue.md)

## U2 development lineage

- [U2-v2 post-gate rank diagnostic](../stageb_u2v2_postgate_rank_diagnostic_20260816.md)
- [U2-v3 category-admission bridge](../stageb_u2v3_category_admission_bridge_20260816.md)
- [U2-v4 legacy-admission replay](../stageb_u2v4_legacy_admission_replay_20260816.md)
- [U2-v5 leakage-clean anchor](../stageb_u2v5_leakage_clean_anchor_20260817.md)
- [U2-v5 complete ablation design](../paper_cvpr_u2v5_complete_ablation_design_20260817.md)
- [U2-v5 ablation execution](../paper_cvpr_u2v5_ablation_execution_20260817.md)
- [U2-v5 ablation results](../paper_cvpr_u2v5_ablation_results_20260818.md)

## Naming rule

Historical text may contain `PIVOT`, `Stage-B`, version numbers, and serialized
`pivot.*` schemas. Preserve those strings inside historical artifacts. New
paper-facing prose must use:

- **ARROW** for the project and method;
- **ARROW-U2** for the complete sealed paper model;
- **ARROW-V/T/N** only for Admission-input interfaces or controls.
