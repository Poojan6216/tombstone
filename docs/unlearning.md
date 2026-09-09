# Unlearning: methods, metrics, and why the metrics are fragile

The model is an artifact too. A fine-tuned adapter has no `delete()`; the subject's data is
diffused into weights. Tombstone treats the adapter as a store with a suppress step, a reclaim
step and probes, and it measures what each unlearning method actually achieves. The numbers below
come from committed runs (`bench/results/unlearn-latest.json`, `bench/results/phase4-smoke.json`,
`bench/results/composition-latest.json`); this page explains what they mean and what they do not.

## Setup

- Base model `Qwen/Qwen2.5-0.5B` on CPU (the integration is the claim, not the scale).
- Dataset: one example per synthetic subject carrying its canary sentence, sharded by a stable
  hash of the subject HMAC into 16 shards, so every subject's data lives in exactly one shard.
- One LoRA adapter per shard (rank 16, all projection matrices), plus one *unsharded* adapter
  trained on everything for the approximate-unlearning comparison.
- Canaries: a natural sentence with a unique 12-character token bound to the subject. Ground
  truth for extraction: greedy decoding from the sentence's prefix must not reproduce the token.
- Membership inference: loss-based and Min-K% (Shi et al. 2024) over the subject's examples vs
  a held-out reference set of matched length; AUC with a bootstrap 95% CI. An AUC whose CI
  includes 0.5 passes the MIA probe.
- Utility: perplexity on 60 held-out AG News passages.

## The serving composition, and a measurement that changed the design

The spec allowed an averaged or sequential *weight* merge of the shard adapters. We measured
before choosing (`bench/results/composition-*.json`): three shard adapters each extracted their
own canaries (2/2, 2/2, 1/2) but every weight-level merge lost them — 0/6 for a linear sum, 0/6
for concatenation, 1/6 for a linear average. Merged LoRA deltas do not preserve memorised facts.

The serving model is therefore what SISA (Bourtoule et al. 2021) actually prescribes: an
ensemble of the shard models at prediction time. Each active shard scores the next token and the
ensemble is a mixture weighted by the likelihood each shard assigns to the context so far (the
shard that memorised a document recognises its own prefix; on unrelated text the weights are
near-uniform). A first attempt with per-token max-confidence gating was also measured and
rejected: an over-fitted shard is "confident" everywhere and won the gate on unrelated text
(held-out perplexity 1.7×10⁵) and on other shards' canaries (2/6 extracted).

Cost: one forward pass per active shard per token. Exact unlearning cost: one shard's retrain.

## Methods

| method | what it does | what it can promise |
|---|---|---|
| exact (M3) | drop the subject's rows from its shard, retrain that shard's adapter from the base model, recompose the ensemble | the data was never in the retrained weights; immune to relearning by construction |
| NPO (M1) | negative preference optimisation (Zhang et al. 2024) on the unsharded adapter: push the forget set below the reference model's likelihood, bounded, plus a retain loss | lowers extractability; leaves residue; reversible by light continued training |
| gradient difference (M2) | gradient ascent on the forget set plus descent on a retain set (Liu et al. 2022; the GD baseline in TOFU, Maini et al. 2024) | same class of promise as NPO, usually less stable |
| full retrain (M4) | retrain the unsharded adapter from scratch without the subject | the oracle; costs the whole dataset every time |

Hyperparameters for M1/M2 come from a grid (steps × learning rate) committed in full in
`RESULTS.md`; the chosen configuration minimises canary extraction subject to held-out
perplexity staying within 25% of the unsharded baseline. The full grid is reported, not just the
winner.

## What the numbers say

See `RESULTS.md` § "Unlearning matrix" for the committed rows: canary extraction, both MIA AUCs
with CIs, held-out perplexity delta, wall-clock and CPU-minutes for M0–M4, plus the relearning
attack (Phase 7.6): after M1/M2, a handful of fine-tuning steps on *unrelated* AG News text
re-measures extraction; after M3 the same procedure is run on the retrained shard adapters.

The two statements this page exists to make plainly:

1. **Exact shard retraining is the only method here whose "forgotten" is structural.** The
   subject's rows are gone from the shard, the shard's weights are retrained without them, and
   nothing can be relearned from weights that never saw the data.
2. **Approximate methods leave residual extractability at some measured rate, and MIA after
   them may still separate members.** Whatever the rate is in the current run, it is reported
   next to M3 and M4, and the relearning rows show how quickly it comes back.

## Why the metrics are fragile

*Verification of Machine Unlearning is Fragile* (arXiv 2408.00929) shows verification can be
fooled; the *Dual-View Inference Attack* (arXiv 2512.16126) and SMI (arXiv 2602.01150) show
stronger membership signals than loss alone. Our probes are two attacks and one extraction
protocol. A `VERIFIED(model)` in a receipt means: canaries not extractable by greedy decoding
from their prefixes, and neither of these two MIAs separates the subject's examples from a
held-out set at the 95% level. It does not mean the weights carry no trace; it means these
measurements found none. That is why the receipt carries the measurement, and why Phase 7
attacks the measurement rather than trusting it.
