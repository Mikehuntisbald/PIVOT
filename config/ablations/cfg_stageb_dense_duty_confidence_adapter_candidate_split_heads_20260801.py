from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_deployed_routing_20260801 import *  # noqa: F401,F403

# V44 keeps the complete v43 deployed-routing surface and separates only the
# token-veto and global absolute-confidence gradient owners.  The revision
# remains v43 so every data, loss, routing, and inference contract is unchanged.
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_global_absolute_v2"
)

# The TN rows and token roles are unchanged. This dedicated receipt rebinds
# their audit to the split-head code source closure without mutating v43's
# historical provenance.
stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_split_heads_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "1706f96560d59673d73674ff2486b8f9c811c70ff413d25905e7c10ea0d93038"
)
