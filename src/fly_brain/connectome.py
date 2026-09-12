"""Streaming MaleCNS import. Matrices are always rows=post, columns=pre."""
import collections
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order
from .assets import (ATTRIBUTION, artifact_manifest, canonical_hash, checked_files,
                     ensure_assets, home, locked, write_json)

RECIPE = {
    "version": 1,
    "nodes": "status=Traced OR (status=null AND superclass!=null)",
    "excluded_statuses": ["Glia", "Unimportant", "Orphan", "Assign", "Anchor"],
    "edge_selection": "both endpoints in selected neurons; all positive contact counts",
    "orientation": "row=post,column=pre",
    "normalization": "each postsynaptic row sums to one; zero rows remain zero",
    "input_superclasses": ["cb_sensory", "ol_sensory", "vnc_sensory", "sensory_ascending", "sensory_descending"],
    "output_superclasses": ["descending_neuron", "cb_motor", "vnc_motor"],
}


def normalize_counts(counts):
    graph = counts.astype(np.float32)
    totals = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    return sparse.csr_matrix(sparse.diags(inverse) @ graph, dtype=np.float32)


def map_ids(ids, values):
    positions = np.searchsorted(ids, values)
    valid = positions < len(ids)
    valid &= ids[np.minimum(positions, len(ids) - 1)] == values
    return positions, valid


def io_reachability(graph, inputs, outputs):
    n = graph.shape[0]
    # A temporary virtual source performs one directed BFS from all sensory cells.
    outgoing = graph.T.tocsr()
    augmented = sparse.vstack([
        sparse.hstack([outgoing, sparse.csr_matrix((n, 1))]),
        sparse.csr_matrix((np.ones(len(inputs)), (np.zeros(len(inputs)), inputs)), shape=(1, n + 1)),
    ], format="csr")
    reached = breadth_first_order(augmented, n, directed=True, return_predecessors=False)
    mask = np.zeros(n + 1, dtype=bool)
    mask[reached] = True
    return mask[outputs], int(mask[:n].sum())


def import_graph(root=None, progress=print):
    root = home(root)
    sources = ensure_assets(root)
    graph_id = canonical_hash({"sources": {a["name"]: a["sha256"] for a in sources["files"]}, "recipe": RECIPE})
    target = root / "prepared" / graph_id
    with locked(root / ".locks" / "import"):
        if target.exists():
            manifest = json.loads((target / "manifest.json").read_text())
            checked_files(target, manifest)
            write_json(root / "prepared" / "latest.json", {"graph_id": graph_id})
            return target, manifest
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=target.parent))
        try:
            manifest = _import(root, temporary, graph_id, sources, progress)
            os.rename(temporary, target)
            write_json(root / "prepared" / "latest.json", {"graph_id": graph_id})
            return target, manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _import(root, destination, graph_id, sources, progress):
    started = time.monotonic()
    annotations = feather.read_table(root / "v1.0" / "annotations.feather")
    statuses = annotations["status"].to_pylist()
    classes = annotations["superclass"].to_pylist()
    keep = np.array([s == "Traced" or (s is None and c is not None) for s, c in zip(statuses, classes)])
    selected = annotations.filter(pa.array(keep)).sort_by("bodyId")
    ids = selected["bodyId"].to_numpy()
    if len(np.unique(ids)) != len(ids) or len(ids) == 0:
        raise ValueError("Neuron IDs must be unique and nonempty")
    np.save(destination / "node-ids.npy", ids, allow_pickle=False)
    selected_classes = selected["superclass"].to_pylist()
    inputs = np.array([i for i, c in enumerate(selected_classes) if c in RECIPE["input_superclasses"]], dtype=np.int32)
    outputs = np.array([i for i, c in enumerate(selected_classes) if c in RECIPE["output_superclasses"]], dtype=np.int32)
    if not len(inputs) or not len(outputs) or np.intersect1d(inputs, outputs).size:
        raise ValueError("Input/output populations must be nonempty and disjoint")
    progress(f"Selected {len(ids):,} neurons; {len(inputs):,} sensory inputs, {len(outputs):,} outputs", flush=True)

    # Stream the 152M raw rows: never materialize the segment graph or N x N dense arrays.
    pre_parts, post_parts, weight_parts = [], [], []
    raw_rows = excluded_rows = excluded_contacts = 0
    with pa.memory_map(str(root / "v1.0" / "edges.feather"), "r") as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pre, valid_pre = map_ids(ids, batch["body_pre"].to_numpy())
            post, valid_post = map_ids(ids, batch["body_post"].to_numpy())
            weight = batch["weight"].to_numpy()
            if np.any(weight <= 0):
                raise ValueError("Nonpositive synapse count in official edges")
            retain = valid_pre & valid_post
            raw_rows += len(weight)
            excluded_rows += int((~retain).sum())
            excluded_contacts += int(weight[~retain].sum())
            pre_parts.append(pre[retain].astype(np.int32))
            post_parts.append(post[retain].astype(np.int32))
            weight_parts.append(weight[retain].astype(np.int64))
    counts = sparse.coo_matrix((np.concatenate(weight_parts),
                               (np.concatenate(post_parts), np.concatenate(pre_parts))), shape=(len(ids), len(ids))).tocsr()
    del pre_parts, post_parts, weight_parts
    counts.sum_duplicates()
    counts.sort_indices()
    sparse.save_npz(destination / "contact-counts.npz", counts)
    graph = normalize_counts(counts)
    sparse.save_npz(destination / "graph-csr.npz", graph)
    progress(f"Retained {counts.nnz:,} directed weighted edges from {raw_rows:,} raw rows", flush=True)
    reachable, reached_count = io_reachability(graph, inputs, outputs)
    if not reachable.any():
        raise ValueError("No directed path from sensory inputs to outputs")
    write_json(destination / "io-populations.json", {
        "inputs": inputs.tolist(), "outputs": outputs.tolist(),
        "input_body_ids": ids[inputs].tolist(), "output_body_ids": ids[outputs].tolist(),
        "unreachable_output_body_ids": ids[outputs[~reachable]].tolist(),
        "reached_neurons": reached_count, "reachable_outputs": int(reachable.sum()),
    })

    nt = feather.read_table(root / "v1.0" / "neurotransmitters.feather")
    nt = nt.filter(pa.compute.is_in(nt["body"], value_set=pa.array(ids))).sort_by("body")
    nt_rows = {row["body"]: row for row in nt.to_pylist()}
    if len(nt_rows) != len(nt):
        raise ValueError("Duplicate transmitter body IDs")
    nt_names = sorted({row["consensus_nt"] for row in nt_rows.values() if row["consensus_nt"] is not None}) + ["unknown"]
    superclasses = sorted({c for c in selected_classes if c is not None}) + ["unknown"]
    feature_names = ["nt:" + x for x in nt_names] + ["class:" + x for x in superclasses] + ["side:L", "side:R", "side:M", "nt_confidence"]
    features = np.zeros((len(ids), len(feature_names)), dtype=np.float32)
    nt_labels, confidence, predictions, ground_truth, body_prediction = [], [], [], [], []
    sides = selected["somaSide"].to_pylist()
    for i, body_id in enumerate(ids):
        row = nt_rows.get(int(body_id), {})
        label = row.get("consensus_nt") or "unknown"
        conf = row.get("predicted_nt_confidence")
        conf = float(conf) if conf is not None and np.isfinite(conf) else 0.0
        nt_labels.append(label); confidence.append(conf)
        predictions.append(row.get("total_nt_predictions", 0))
        ground_truth.append(row.get("ground_truth"))
        body_prediction.append(row.get("predicted_nt"))
        features[i, nt_names.index(label)] = 1
        features[i, len(nt_names) + superclasses.index(selected_classes[i] or "unknown")] = 1
        if sides[i] in ("L", "R", "M"):
            features[i, len(nt_names) + len(superclasses) + ["L", "R", "M"].index(sides[i])] = 1
        features[i, -1] = conf
    np.save(destination / "neuron-features.npy", features, allow_pickle=False)
    for name, values in {"consensus_nt": nt_labels, "predicted_nt_confidence": confidence,
                         "total_nt_predictions": predictions, "ground_truth_nt": ground_truth,
                         "body_predicted_nt": body_prediction}.items():
        selected = selected.append_column(name, pa.array(values))
    pq.write_table(selected, destination / "neuron-features.parquet", compression="zstd")
    write_json(destination / "feature-schema.json", {"names": feature_names, "transmitter_signs_assumed": False})
    (destination / "ATTRIBUTION.md").write_text(ATTRIBUTION)
    excluded = collections.Counter((s or "null") for s, retained in zip(statuses, keep) if not retained)
    return artifact_manifest(destination, {
        "schema": "malecns-graph-v1", "graph_id": graph_id, "recipe": RECIPE,
        "sources": sources["files"], "license": "CC-BY-4.0",
        "nodes": len(ids), "edges": counts.nnz, "synaptic_contacts": int(counts.sum()),
        "raw_annotation_rows": len(annotations), "raw_edge_rows": raw_rows,
        "excluded_annotation_rows_by_status": dict(excluded),
        "excluded_edge_rows": excluded_rows, "excluded_synaptic_contacts": excluded_contacts,
        "aggregated_duplicate_edge_rows": raw_rows - excluded_rows - counts.nnz,
        "inputs": len(inputs), "outputs": len(outputs), "reachable_outputs": int(reachable.sum()),
        "missing_transmitter_rows": int(len(ids) - len(nt_rows)),
        "sparse_bytes": graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes,
        "build_seconds": time.monotonic() - started,
    })


def resolve_graph(root=None, graph_id=None, verify=True):
    root = home(root)
    if graph_id is None:
        graph_id = json.loads((root / "prepared" / "latest.json").read_text())["graph_id"]
    if not graph_id or any(c not in "0123456789abcdef" for c in graph_id):
        raise ValueError("Invalid graph asset ID")
    directory = root / "prepared" / graph_id
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["graph_id"] != graph_id:
        raise ValueError("Graph identity mismatch")
    if verify:
        checked_files(directory, manifest)
    return directory, manifest


class Graph:
    def __init__(self, directory, verify=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if verify:
            checked_files(self.directory, self.manifest)
        self.matrix = sparse.load_npz(self.directory / "graph-csr.npz").tocsr()
        self.ids = np.load(self.directory / "node-ids.npy", allow_pickle=False)
        self.features = np.load(self.directory / "neuron-features.npy", allow_pickle=False)
        populations = json.loads((self.directory / "io-populations.json").read_text())
        self.inputs = np.array(populations["inputs"], dtype=np.int64)
        self.outputs = np.array(populations["outputs"], dtype=np.int64)
        if self.matrix.shape != (len(self.ids), len(self.ids)):
            raise ValueError("Graph shape and neuron IDs disagree")
