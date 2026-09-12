"""Immutable MaleCNS circuit assets with explicit biological modelling assumptions."""
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from ..assets import home, canonical_hash, locked, artifact_manifest, checked_files, write_json, ATTRIBUTION
from ..connectome import resolve_graph

RECIPE = {
    "schema": "malecns-visual-dopamine-circuit-v1", "version": 1,
    "photoreceptors": "R1-R6 with a dominant outgoing L1/L2 column assignment",
    "visual_kc": "KCg-d", "reward_dan": "PAM01, consensus dopamine",
    "dopamine_targets": "MBONs receiving >=10 synapses from the selected PAM01 pool",
    "motor": "annotated fl/ml/hl VNC motor neurons with direct selected DN input",
    "fast_signs": {"acetylcholine": 1, "histamine": -1, "gaba": -1, "glutamate": -1,
                   "dopamine": 0, "octopamine": 0, "serotonin": 0, "unclear": 1, "unknown": 1},
    "assumptions": [
        "Central glutamatergic effects are assumed inhibitory; receptor-resolved signs are not provided by the connectome.",
        "Uncertain fast transmitters default excitatory. Sensitivity controls are required before biological interpretation.",
        "R1-R6 drive uses camera luminance as a proxy, not a calibrated fly spectral response.",
        "PAM01 is an aggregate gamma5 reward candidate; subtype-specific molecular dynamics are not reconstructed.",
        "Visual KC to MBON edges are plastic; other recorded connections in the selected route stay fixed.",
        "The sensory core excludes MBON/descending/motor output feedback; the output route uses direct MBON-DN-MN edges.",
        "Rate states are deviations from a tonic operating point; steps are abstract, not validated biological milliseconds.",
    ],
    "sources": ["https://male-cns.janelia.org/download/", "https://elifesciences.org/articles/14009",
                "https://elifesciences.org/articles/62576", "https://pmc.ncbi.nlm.nih.gov/articles/PMC8105414/"],
}


def normalized(matrix):
    matrix = matrix.astype(np.float32).tocsr()
    total = np.asarray(abs(matrix).sum(axis=1)).ravel()
    return (sparse.diags(np.divide(1., total, out=np.zeros_like(total), where=total > 0)) @ matrix).tocsr()


def prepare(root=None, graph_id=None, pam_min_synapses=10):
    if pam_min_synapses not in (1,10):raise ValueError("Supported PAM innervation thresholds are 1 and 10")
    recipe={**RECIPE}
    if pam_min_synapses==1:
        recipe={**recipe,"version":2,"dopamine_targets":"MBONs receiving >=1 recorded synapse from the selected PAM01 pool"}
    root = home(root)
    graph, metadata = resolve_graph(root, graph_id, verify=False)
    circuit_id = canonical_hash({"graph_id": metadata["graph_id"], "recipe": recipe})
    destination = root / "circuits" / circuit_id
    with locked(root / ".locks" / ("visual-dopamine-" + circuit_id)):
        if (destination / "manifest.json").exists():
            record = json.loads((destination / "manifest.json").read_text())
            checked_files(destination, record)
            return destination, record
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".visual-dopamine-", dir=destination.parent))
        try:
            t = pq.read_table(graph / "neuron-features.parquet").to_pydict()
            ids = np.load(graph / "node-ids.npy", allow_pickle=False)
            if t["bodyId"] != ids.tolist():
                raise ValueError("Neuron annotations are not aligned with the graph IDs")
            counts = sparse.load_npz(graph / "contact-counts.npz").tocsr()
            def indices(condition): return np.asarray([i for i in range(len(ids)) if condition(i)], dtype=np.int64)
            kc = indices(lambda i: t["type"][i] == "KCg-d")
            dan = indices(lambda i: t["type"][i] == "PAM01" and t["consensus_nt"][i] == "dopamine")
            all_mbon = indices(lambda i: t["class"][i] == "MBON")
            mbon = all_mbon[np.asarray(counts[all_mbon][:, dan].sum(axis=1)).ravel() >= pam_min_synapses]
            mbon = mbon[np.asarray(counts[mbon][:, kc].sum(axis=1)).ravel() > 0]
            all_dn = indices(lambda i: t["superclass"][i] == "descending_neuron")
            dn = all_dn[np.asarray(counts[all_dn][:, mbon].sum(axis=1)).ravel() > 0]
            all_mn = indices(lambda i: t["superclass"][i] == "vnc_motor" and t["subclass"][i] in ["fl", "ml", "hl"])
            mn = all_mn[np.asarray(counts[all_mn][:, dn].sum(axis=1)).ravel() > 0]
            anchors = indices(lambda i: t["type"][i] in ["L1", "L2"] and
                t["assignedOlHex1"][i] is not None and t["assignedOlHex2"][i] is not None and t["somaSide"][i] in ["L", "R"])
            photos = indices(lambda i: t["type"][i] == "R1-R6")
            contacts = counts[anchors][:, photos].tocsc()
            retina, hexes, sides, confidence = [], [], [], []
            for column, photo in enumerate(photos):
                a,b = contacts.indptr[column:column+2]
                if a == b: continue
                rows = contacts.indices[a:b]; weights = contacts.data[a:b]
                votes = {}
                for row, weight in zip(rows, weights):
                    anchor = anchors[row]
                    key = (t["somaSide"][anchor], t["assignedOlHex1"][anchor], t["assignedOlHex2"][anchor])
                    votes[key] = votes.get(key, 0) + int(weight)
                best = max(votes, key=votes.get)
                retina.append(photo); sides.append(0 if best[0] == "L" else 1); hexes.append(best[1:])
                confidence.append(votes[best] / sum(votes.values()))
            retina = np.asarray(retina, dtype=np.int64); hexes = np.asarray(hexes, dtype=np.float32)
            uv = np.empty_like(hexes)
            for side in (0,1):
                mask = np.asarray(sides) == side
                # An explicit engineering registration, not a claim of calibrated visual-field angles.
                xy = np.column_stack([hexes[mask,0] + .5*hexes[mask,1], np.sqrt(3)/2*hexes[mask,1]])
                uv[mask] = (xy - xy.min(axis=0)) / np.maximum(np.ptp(xy, axis=0), 1)
            if min(len(retina),len(kc),len(mbon),len(dan),len(dn),len(mn)) == 0:
                raise ValueError("An anatomical population is empty")
            signs = np.asarray([RECIPE["fast_signs"].get(nt,1) for nt in t["consensus_nt"]], dtype=np.float32)
            sensory = normalized(counts).tocsc()
            sensory.data *= np.repeat(signs, np.diff(sensory.indptr))
            sensory = sensory.tocsr()
            # No downstream shortcut or repeated inclusion of the plastic edge block.
            blocked = np.unique(np.r_[all_mbon, all_dn, all_mn, dan])
            row_mask = np.ones(len(ids),dtype=np.float32); row_mask[blocked] = 0
            sensory = (sparse.diags(row_mask) @ sensory).tocsr(); sensory.eliminate_zeros()
            plastic = normalized(counts[mbon][:, kc]).toarray()
            dn_weights = normalized(counts[dn][:, mbon]).toarray() * signs[mbon][None,:]
            mn_weights = normalized(counts[mn][:, dn]).toarray() * signs[dn][None,:]
            dopamine = normalized(counts[mbon][:, dan]).toarray()
            distance = np.full(len(ids), -1, dtype=np.int16); distance[retina] = 0
            frontier = np.zeros(len(ids),dtype=np.int8); frontier[retina] = 1
            for depth in range(1,17):
                found = (counts @ frontier > 0) & (distance < 0)
                distance[found] = depth; frontier[:] = found
                if (distance[kc] >= 0).all(): break
            if (distance[kc] < 0).any(): raise ValueError("A selected visual KC is unreachable from the retinal input")
            populations = {name: a for name,a in [("retina",retina),("kc",kc),("mbon",mbon),("dan",dan),("dn",dn),("motor",mn)]}
            np.savez(tmp / "populations.npz", ids=ids, retina_uv=uv, retina_side=np.asarray(sides,dtype=np.int8),
                     retina_mapping_confidence=np.asarray(confidence,dtype=np.float32), **populations)
            np.savez(tmp / "routes.npz", plastic_base=plastic, dn_weights=dn_weights, motor_weights=mn_weights, dopamine_weights=dopamine)
            sparse.save_npz(tmp / "sensory-signed.npz", sensory)
            details = {name: [{"body_id":int(ids[i]),"type":t["type"][i],"side":t["somaSide"][i],"subclass":t["subclass"][i],"nt":t["consensus_nt"][i]} for i in inds] for name,inds in populations.items()}
            write_json(tmp / "neurons.json",details)
            (tmp / "ATTRIBUTION.md").write_text(ATTRIBUTION)
            record = artifact_manifest(tmp,{"schema":RECIPE["schema"],"circuit_id":circuit_id,"graph_id":metadata["graph_id"],
                "recipe":recipe,"populations":{k:len(v) for k,v in populations.items()},
                "plastic_edges":int(np.count_nonzero(plastic)),"retina_to_kc_max_hops":int(distance[kc].max()),
                "median_retina_assignment_confidence":float(np.median(confidence)),"biologically_validated":False})
            os.rename(tmp,destination)
        finally:
            if tmp.exists(): shutil.rmtree(tmp)
    return destination,record


class Circuit:
    def __init__(self,directory,verify=True):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/"manifest.json").read_text())
        if verify: checked_files(self.directory,self.manifest)
        for key,value in np.load(self.directory/"populations.npz",allow_pickle=False).items(): setattr(self,key,value)
        for key,value in np.load(self.directory/"routes.npz",allow_pickle=False).items(): setattr(self,key,value)
        self.sensory=sparse.load_npz(self.directory/"sensory-signed.npz").tocsr()
        self.neurons=json.loads((self.directory/"neurons.json").read_text())
