"""Degree-preserving directed double-edge swaps for topology comparisons."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from scipy import sparse
from .assets import artifact_manifest, canonical_hash, home, locked, write_json
from .connectome import normalize_counts, resolve_graph, io_reachability


def directed_swaps(counts, seed=0, swaps_per_edge=1):
    if not np.isfinite(swaps_per_edge) or swaps_per_edge <= 0:
        raise ValueError("Positive swaps per edge required")
    original = counts.tocoo()
    post = original.row.copy(); pre = original.col.copy(); weights = original.data.copy()
    n = counts.shape[0]
    occupied = set((post.astype(np.int64) * n + pre).tolist())
    rng = np.random.default_rng(seed)
    accepted = 0; wanted = int(len(pre) * swaps_per_edge)
    attempts = 0
    while accepted < wanted and attempts < max(100, wanted * 20):
        a, b = rng.integers(0, len(pre), size=2)
        attempts += 1
        # Exchange destinations while preserving both in/out degree sequences.
        if a == b or post[a] == post[b] or pre[a] == pre[b] or post[b] == pre[a] or post[a] == pre[b]:
            continue
        next_a = int(post[b]) * n + int(pre[a]); next_b = int(post[a]) * n + int(pre[b])
        if next_a in occupied or next_b in occupied:
            continue
        occupied.remove(int(post[a]) * n + int(pre[a])); occupied.remove(int(post[b]) * n + int(pre[b]))
        occupied.add(next_a); occupied.add(next_b)
        post[a], post[b] = post[b], post[a]
        accepted += 1
    result = sparse.coo_matrix((weights, (post, pre)), shape=counts.shape).tocsr()
    if not np.array_equal(np.diff(result.indptr), np.diff(counts.tocsr().indptr)):
        raise AssertionError("In-degree sequence changed")
    if not np.array_equal(np.diff(result.tocsc().indptr), np.diff(counts.tocsc().indptr)):
        raise AssertionError("Out-degree sequence changed")
    return result, {"seed": seed, "requested_swaps": wanted, "accepted_swaps": accepted, "attempts": attempts,
                    "preserved": ["directed in-degree", "directed out-degree", "weight multiset", "outgoing weighted strength"],
                    "not_preserved": ["incoming weighted strength", "biological cell-pair relationships"]}


def rewire(root=None, seed=0, swaps_per_edge=1):
    root = home(root)
    source, original = resolve_graph(root)
    recipe = {"method": "directed-double-edge-swap-v1", "parent_graph_id": original["graph_id"],
              "seed": seed, "swaps_per_edge": swaps_per_edge}
    graph_id = canonical_hash(recipe)
    target = root / "prepared" / graph_id
    with locked(root / ".locks" / graph_id):
        if target.exists():
            return {"directory": str(target), "graph_id": graph_id, "reused": True}
        temp = Path(tempfile.mkdtemp(prefix=".rewire-", dir=target.parent))
        try:
            counts, stats = directed_swaps(sparse.load_npz(source / "contact-counts.npz"), seed, swaps_per_edge)
            sparse.save_npz(temp / "contact-counts.npz", counts)
            graph = normalize_counts(counts)
            sparse.save_npz(temp / "graph-csr.npz", graph)
            for name in ("node-ids.npy", "neuron-features.npy", "neuron-features.parquet", "feature-schema.json", "ATTRIBUTION.md"):
                shutil.copy2(source / name, temp / name)
            populations = json.loads((source / "io-populations.json").read_text())
            reachable, total = io_reachability(graph, populations["inputs"], populations["outputs"])
            populations["reachable_outputs"] = int(reachable.sum()); populations["reached_neurons"] = total
            ids = np.load(temp / "node-ids.npy", allow_pickle=False)
            populations["unreachable_output_body_ids"] = ids[np.array(populations["outputs"])[~reachable]].tolist()
            write_json(temp / "io-populations.json", populations)
            metadata = {k: v for k, v in original.items() if k not in ("files", "graph_id", "recipe")}
            artifact_manifest(temp, {**metadata, "graph_id": graph_id, "recipe": recipe, "rewiring": stats,
                                     "reachable_outputs": int(reachable.sum())})
            os.rename(temp, target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return {"directory": str(target), "graph_id": graph_id, **stats}
