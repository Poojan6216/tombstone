"""A LoRA adapter (sharded or not) as an ``ErasableStore``.

Layout under ``path``::

    shard-NN/            one adapter per shard (SISA-style), trained from the base model
    serving/             the composition served to users (linear average of shard adapters)
    unsharded/           one adapter trained on everything (the approximate-unlearning case)
    tombstone-state.json excluded shards, original adapter hashes, per-saga forget snapshots

suppress  = recompose ``serving`` without the affected shards (immediate, reversible). The
            unsharded adapter cannot be hidden; suppression is recorded only.
reclaim   = exact: retrain the shard on its remaining examples (the dataset store has already
            dropped the subject's rows) and recompose. approximate: NPO / gradient difference on
            the unsharded adapter with the subject's examples as the forget set.
probe_logical  = extraction: greedy decoding from the subject's prefixes must not reproduce the
            continuation (canary tokens when the app supplies them, else a 70% text split).
probe_physical = the shard adapter's weights hash differs from the original and no excluded
            copy of the original remains on disk.
probe_model    = extraction rate + membership inference AUC with CI (reference set required).
capabilities = {LOGICAL, MODEL} for sharded adapters, {LOGICAL} for an unsharded one, which is
            RESIDUAL-prone by construction.

The subject's example texts are snapshotted at suppress time (before the dataset reclaims them)
into ``tombstone-state.json`` under the app's adapter directory — never under ``.tombstone/`` —
and dropped once the saga has probed the model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tombstone.errors import NotSupported
from tombstone.lineage.capture import Capture
from tombstone.model.artifacts import ArtifactKind, ArtifactRef
from tombstone.model.lineage import Edge, Node, Trace
from tombstone.model.pins import ModelPin
from tombstone.model.status import VerifyLevel
from tombstone.stores.base import (
    LogicalProbeResult,
    PhysicalProbeResult,
    ProbeSet,
    ReclaimResult,
)
from tombstone.train.dataset import DatasetStore
from tombstone.train.finetune import TrainConfig, adapter_hash
from tombstone.train.ops import ModelOps, prompts_from_texts, read_texts_jsonl
from tombstone.train.unlearn import UnlearnConfig
from tombstone.util import atomic_write_text, derived_ulid, sha256_hex

STATE = "tombstone-state.json"


def _read_json(p: Path) -> dict[str, Any]:
    return dict(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else {}


class AdapterStore:
    kind = "adapter"

    def __init__(
        self,
        name: str,
        path: str | Path,
        shards: int = 1,
        base_model: str = "",
        runtime_model_cfg: Any = None,
        dataset: DatasetStore | None = None,
        ops: ModelOps | None = None,
        mia_reference: Path | None = None,
        train_cfg: TrainConfig | None = None,
    ) -> None:
        self.name = name
        self.path = Path(path)
        self.shards = shards
        self.base_model = base_model
        self.model_cfg = runtime_model_cfg
        self.dataset = dataset
        self._ops = ops
        self.mia_reference = mia_reference
        self.train_cfg = train_cfg or TrainConfig(base_model=base_model)
        self.unlearn_method = str(getattr(runtime_model_cfg, "unlearn", "exact"))
        self.unlearn_cfg = UnlearnConfig(
            method=self.unlearn_method if self.unlearn_method != "exact" else "npo",
            steps=int(getattr(runtime_model_cfg, "unlearn_steps", 40)),
            lr=float(getattr(runtime_model_cfg, "unlearn_lr", 1e-4)),
        )
        self.context: Trace | None = None
        self.probe_prompts: list[tuple[str, str]] | None = None  # app-supplied (e.g. canaries)
        self.capabilities: frozenset[VerifyLevel] = self.detect_capabilities()

    # --- layout ------------------------------------------------------------------------------------

    @property
    def serving_dir(self) -> Path:
        return self.path / "serving"

    @property
    def unsharded_dir(self) -> Path:
        return self.path / "unsharded"

    def shard_dir(self, shard: int) -> Path:
        return self.path / f"shard-{shard:02d}"

    @property
    def state_path(self) -> Path:
        return self.path / STATE

    def state(self) -> dict[str, Any]:
        return _read_json(self.state_path)

    def _write_state(self, st: dict[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        # durable erasure state: never leave it truncated behind a crash or a concurrent reader
        atomic_write_text(self.state_path, json.dumps(st, indent=1, sort_keys=True))

    def shard_dirs(self) -> list[Path]:
        return sorted(p for p in self.path.glob("shard-*") if (p / "adapter_config.json").is_file())

    def is_sharded(self) -> bool:
        return self.shards > 1 and bool(self.shard_dirs())

    def detect_capabilities(self) -> frozenset[VerifyLevel]:
        caps = {VerifyLevel.LOGICAL}
        if self.is_sharded():
            caps.add(VerifyLevel.MODEL)
            caps.add(VerifyLevel.PHYSICAL)
        elif (self.unsharded_dir / "adapter_config.json").is_file():
            caps.add(VerifyLevel.PHYSICAL)
        return frozenset(caps)

    @property
    def ops(self) -> ModelOps:
        if self._ops is None:
            from tombstone.train.ops import TorchOps

            self._ops = TorchOps()
        return self._ops

    def version(self) -> str:
        return f"peft adapters: {len(self.shard_dirs())} shard(s), base {self.base_model}"

    def close(self) -> None:
        return None

    def pin(self) -> ModelPin:
        hashes = [adapter_hash(d) for d in self.shard_dirs()]
        if (self.unsharded_dir / "adapter_config.json").is_file():
            hashes.append(adapter_hash(self.unsharded_dir))
        return ModelPin(
            self.name,
            self.base_model,
            sha256_hex("|".join(hashes)),
            len(self.shard_dirs()) or self.shards,
        )

    # --- lineage registration --------------------------------------------------------------------

    def register_lineage(self, capture: Capture, dataset: DatasetStore) -> list[Node]:
        """ADAPTER nodes: one per shard adapter (edges from that shard's TRAIN examples) and one
        for the unsharded adapter (edges from every example). Idempotent."""
        capture.register(self.name, self.kind)
        lineage = capture.lineage
        nodes: list[Node] = []
        manifest = dataset.manifest()
        by_shard: dict[int, list[dict[str, Any]]] = {}
        for e in manifest["examples"]:
            by_shard.setdefault(int(e["shard"]), []).append(e)
        targets: list[tuple[str, Path, list[dict[str, Any]]]] = []
        for d in self.shard_dirs():
            shard = int(d.name.split("-")[1])
            targets.append((d.name, d, by_shard.get(shard, [])))
        if (self.unsharded_dir / "adapter_config.json").is_file():
            targets.append(
                ("unsharded", self.unsharded_dir, [e for es in by_shard.values() for e in es])
            )
        with lineage.tx():
            for key, d, examples in targets:
                if not examples:
                    continue
                aid = derived_ulid("adapter", self.name, key)
                subjects = sorted({str(e["subject"]) for e in examples})
                node = lineage.node(aid)
                if node is None:
                    node = Node(
                        artifact_id=aid,
                        kind=ArtifactKind.ADAPTER,
                        store=self.name,
                        store_key=key,
                        scope=capture.scope,
                        content_hash=adapter_hash(d),
                        embedding_fingerprint=None,
                        subject_hmac=subjects[0] if len(subjects) == 1 else "shared",
                        created_seq=lineage.next_seq(),
                    )
                    lineage.add_node(node)
                for e in examples:
                    lineage.add_edge(Edge(str(e["id"]), aid, f"adapter:{key}"))
                nodes.append(node)
        return nodes

    # --- erasure context -----------------------------------------------------------------------------

    def set_context(self, trace: Trace) -> None:
        """The saga hands the trace over so the store knows which examples are the subject's."""
        self.context = trace

    def _subject_example_ids(self) -> list[str]:
        if self.context is None:
            return []
        return [a.artifact_id for a in self.context.artifacts if a.kind is ArtifactKind.TRAIN]

    def _snapshot_forget_texts(self) -> list[str]:
        """Read the subject's example texts from the dataset *now* (before it reclaims them)."""
        if self.dataset is None:
            return []
        ids = set(self._subject_example_ids())
        texts: list[str] = []
        for shard in range(self.dataset.shards()):
            for ex in self.dataset.shard_examples(shard, include_suppressed=True):
                if ex.example_id in ids:
                    texts.append(ex.text)
        return texts

    def _shard_of(self, ref: ArtifactRef) -> int | None:
        if ref.store_key.startswith("shard-"):
            return int(ref.store_key.split("-")[1])
        return None

    # --- ErasableStore ---------------------------------------------------------------------------------

    def suppress(self, refs: Sequence[ArtifactRef]) -> None:
        st = self.state()
        excluded = set(int(x) for x in st.get("excluded", []))
        originals = dict(st.get("original_hashes", {}))
        for r in refs:
            originals.setdefault(r.store_key, adapter_hash(self.path / r.store_key))
            shard = self._shard_of(r)
            if shard is not None:
                excluded.add(shard)
        forget = st.get("forget", {})
        key = self.context.trace_id if self.context else "unknown"
        if key not in forget:
            forget[key] = self._snapshot_forget_texts()
        st.update(
            {
                "excluded": sorted(excluded),
                "original_hashes": originals,
                "forget": forget,
                "suppressed": sorted(set(st.get("suppressed", [])) | {r.store_key for r in refs}),
            }
        )
        self._write_state(st)
        if excluded and self.is_sharded():
            self.ops.compose(self.path, self.serving_dir, self.base_model, sorted(excluded))

    def reclaim(self, refs: Sequence[ArtifactRef]) -> ReclaimResult:
        st = self.state()
        key = self.context.trace_id if self.context else "unknown"
        forget = list(st.get("forget", {}).get(key, []))
        results: list[dict[str, Any]] = []
        noop = True
        method = ""
        for r in refs:
            shard = self._shard_of(r)
            before = adapter_hash(self.path / r.store_key)
            if shard is not None and self.unlearn_method == "exact" and self.dataset is not None:
                res = self.ops.exact_unlearn(
                    self.dataset, self.path, self.serving_dir, shard, self.train_cfg
                )
                method = f"exact retrain: shard {shard} of {self.shards}"
                st["excluded"] = sorted(set(int(x) for x in st.get("excluded", [])) - {shard})
            elif shard is not None and self.unlearn_method == "exact" and self.dataset is None:
                raise NotSupported(
                    f"adapter store {self.name!r}: exact unlearning needs the dataset store "
                    "(set `dataset:` on the adapter entry in tombstone.yaml)"
                )
            else:
                target = self.path / r.store_key
                if not forget:
                    raise NotSupported(
                        f"adapter store {self.name!r}: approximate unlearning needs the subject's "
                        "examples (snapshotted at suppress time) — none were found"
                    )
                retain = self._retain_sample(exclude=set(forget))
                res = self.ops.approximate_unlearn(
                    self.base_model, target, forget, retain, self.unlearn_cfg
                )
                method = f"approximate unlearn ({self.unlearn_cfg.method}, {self.unlearn_cfg.steps} steps)"
            after = adapter_hash(self.path / r.store_key)
            if after != before:
                noop = False
            res["adapter_hash_before"] = before
            res["adapter_hash_after"] = after
            results.append(res)
        st["reclaimed"] = sorted(set(st.get("reclaimed", [])) | {r.store_key for r in refs})
        self._write_state(st)
        wall = sum(float(x.get("wall_clock_s", 0.0)) for x in results)
        return ReclaimResult(
            noop=noop,
            method=method or "adapter reclaim",
            measurement={"wall_clock_s": wall, "adapters": float(len(results))},
        )

    def _retain_sample(self, exclude: set[str], n: int = 64) -> list[str]:
        if self.dataset is None:
            return []
        out: list[str] = []
        for shard in range(self.dataset.shards()):
            for ex in self.dataset.shard_examples(shard):
                if ex.text not in exclude:
                    out.append(ex.text)
                if len(out) >= n:
                    return out
        return out

    def _prompts(self) -> list[tuple[str, str]]:
        if self.probe_prompts:
            return list(self.probe_prompts)
        key = self.context.trace_id if self.context else "unknown"
        forget = list(self.state().get("forget", {}).get(key, []))
        return prompts_from_texts(forget)

    def _model_dir_for(self, ref: ArtifactRef) -> Path:
        shard = self._shard_of(ref)
        if shard is not None:
            return (
                self.serving_dir
                if (self.serving_dir / "adapter_config.json").is_file()
                else self.shard_dir(shard)
            )
        return self.path / ref.store_key

    def probe_logical(self, ref: ArtifactRef, probes: ProbeSet) -> LogicalProbeResult:
        prompts = self._prompts()
        if not prompts:
            return LogicalProbeResult(
                found=False, found_by=(), probes_run=0, detail="no extraction prompts"
            )
        hits, total = self.ops.extraction(self.base_model, self._model_dir_for(ref), prompts)
        return LogicalProbeResult(
            found=hits > 0,
            found_by=tuple(f"extract:{i}" for i in range(hits)),
            probes_run=total,
            detail=f"canary {hits}/{total} extracted (greedy)",
        )

    def probe_physical(self, ref: ArtifactRef) -> PhysicalProbeResult:
        if VerifyLevel.PHYSICAL not in self.capabilities:
            raise NotSupported(f"adapter store {self.name!r}: no adapter weights on disk to check")
        st = self.state()
        original = str(st.get("original_hashes", {}).get(ref.store_key, ""))
        current = adapter_hash(self.path / ref.store_key)
        unchanged = bool(original) and original == current
        stale_copies = [p.name for p in self.path.glob(f"{ref.store_key}*.bak*")]
        found = unchanged or bool(stale_copies)
        return PhysicalProbeResult(
            found=found,
            method="adapter weights hash",
            locations=tuple(([ref.store_key] if unchanged else []) + stale_copies),
            measurement={"matches": float(found), "matches_artifact_id": float(unchanged)},
            detail="weights unchanged since suppression"
            if unchanged
            else (
                "stale copy on disk"
                if stale_copies
                else "weights rewritten, no original copy remains"
            ),
        )

    def probe_model(self, ref: ArtifactRef) -> dict[str, Any]:
        prompts = self._prompts()
        model_dir = self._model_dir_for(ref)
        hits, total = (
            self.ops.extraction(self.base_model, model_dir, prompts) if prompts else (0, 0)
        )
        rate = hits / total if total else 0.0
        measurement: dict[str, float] = {
            "canary_rate": rate,
            "canary_extracted": float(hits),
            "canary_total": float(total),
        }
        detail = f"canary {hits}/{total}"
        key = self.context.trace_id if self.context else "unknown"
        members = list(self.state().get("forget", {}).get(key, []))
        reference = read_texts_jsonl(self.mia_reference) if self.mia_reference else []
        if members and reference:
            mia = self.ops.mia(self.base_model, model_dir, members, reference)
            loss = mia.get("loss", {})
            measurement.update(
                {
                    "mia_auc": float(loss.get("auc", 0.5)),
                    "mia_ci_low": float(loss.get("ci_low", 0.0)),
                    "mia_ci_high": float(loss.get("ci_high", 1.0)),
                    "mink_auc": float(mia.get("mink", {}).get("auc", 0.5)),
                }
            )
            detail += f", MIA AUC {measurement['mia_auc']:.2f} [CI {measurement['mia_ci_low']:.2f},{measurement['mia_ci_high']:.2f}]"
        else:
            detail += ", MIA not run (no reference set)"
        found = hits > 0 or (
            "mia_ci_low" in measurement
            and not (measurement["mia_ci_low"] <= 0.5 <= measurement["mia_ci_high"])
        )
        return {
            "found": found,
            "method": "canary extraction + MIA",
            "measurement": measurement,
            "detail": detail,
        }

    def finalize(self) -> None:
        """Drop the forget snapshot once the saga is done with it."""
        st = self.state()
        if "forget" in st:
            st["forget"] = {}
            self._write_state(st)

    def count(self) -> int:
        return len(self.shard_dirs()) + (
            1 if (self.unsharded_dir / "adapter_config.json").is_file() else 0
        )

    def sample_keys(self, n: int) -> list[str]:
        keys = [d.name for d in self.shard_dirs()]
        if (self.unsharded_dir / "adapter_config.json").is_file():
            keys.append("unsharded")
        return keys[:n]
