"""Lane B — Science. Torch-only. NEVER imports FastAPI (contract #3).

Public surface that Lane A imports:
  sae.sae_topk(act, k=15) -> list[Feature]
  persona.score_all_trackers(act_last, act_resp) -> dict[tracker_id -> dict]
  concept_synth.synth_concept(name, desc) -> tracker_id
"""
