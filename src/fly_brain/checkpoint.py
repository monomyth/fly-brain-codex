"""Portable dense weights reference immutable shared graph assets by content ID."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from safetensors.torch import load_file, save_file
from .assets import artifact_manifest, checked_files, write_json
from .connectome import Graph, resolve_graph
from .neural_core import make_model


def save_weights(model, directory):
    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(weights, str(Path(directory) / "model.safetensors"))


def load_checkpoint(directory, root=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "fly-brain-checkpoint-v1":
        raise ValueError("Unsupported checkpoint schema")
    checked_files(directory, manifest)
    graph = None
    if manifest["architecture"]["kind"] == "malecns":
        graph = Graph(resolve_graph(root, manifest["graph_id"])[0], verify=False)
    model = make_model(manifest["architecture"], graph)
    model.load_state_dict(load_file(str(directory / "model.safetensors"), device="cpu"), strict=True)
    return model, manifest


def finish_checkpoint(directory, metadata):
    architecture_note = ("This checkpoint uses the fixed MaleCNS graph. Sensory projection, shared neuron-response parameters, and motor decoding can be learned; frozen-reservoir runs learn only the readout." if metadata["architecture"]["kind"] == "malecns" else "This is a conventional GRU baseline. It contains no MaleCNS graph or fly-neuron dynamics.")
    text = f"""# {metadata.get('name', 'MaleCNS robot policy')}

Training method: {metadata['training_method']}.
Observation mode: {metadata['observation_schema']['mode']}.
Graph: {metadata.get('graph_id', 'none (conventional baseline)')}.

This is an experimental learned robot adapter, not a validated biological fly
simulation. Training loss does not establish closed-loop robot task success.
See the evaluation reports for measured autonomous outcomes. This checkpoint is
not automatically promoted to the 90/100 held-out-success target.

Graph data: MaleCNS v1.0, CC BY 4.0, https://male-cns.janelia.org/download/.
{architecture_note}
A different robot requires new adapters.
No pretrained vision or language model is bundled.

Portable files: dense weights in safetensors, normalization and schemas in
manifest.json. Graph arrays are shared separately, referenced by content ID.
optimizer.pt is a local training-resume artifact, not required for inference.
"""
    (Path(directory) / "MODEL_CARD.md").write_text(text)
    return artifact_manifest(directory, {"schema": "fly-brain-checkpoint-v1", **metadata})


def export_checkpoint(source, destination, root=None, include_graph=False):
    source, destination = Path(source), Path(destination)
    metadata = json.loads((source / "manifest.json").read_text())
    checked_files(source, metadata)
    if metadata.get("schema") in ("visual-dopamine-checkpoint-v1", "seven-motor-checkpoint-v1"):
        from .assets import home
        for key in (("circuit_id", "graph_id", "motor_circuit_id") if metadata.get("schema")=="seven-motor-checkpoint-v1" else ("circuit_id", "graph_id")):
            identity=metadata.get(key, "")
            if len(identity)!=64 or any(c not in "0123456789abcdef" for c in identity): raise ValueError("Invalid visual asset identity")
        if "feedback_circuit_id" in metadata:
            identity=metadata["feedback_circuit_id"]
            if len(identity)!=64 or any(c not in "0123456789abcdef" for c in identity):raise ValueError("Invalid feedback asset identity")
        if destination.exists(): raise FileExistsError("Exports never overwrite an existing directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".export-visual-", dir=destination.parent))
        try:
            shutil.copytree(source, temporary / "checkpoint")
            if include_graph:
                shared = home(root)
                circuit = shared / "circuits" / metadata["circuit_id"]
                checked_files(circuit, json.loads((circuit / "manifest.json").read_text()))
                shutil.copytree(circuit, temporary / "circuits" / metadata["circuit_id"])
                if metadata.get("motor_circuit_id"):
                    motor=shared/"motor-circuits"/metadata["motor_circuit_id"]
                    checked_files(motor,json.loads((motor/"manifest.json").read_text()))
                    shutil.copytree(motor,temporary/"motor-circuits"/metadata["motor_circuit_id"])
                if metadata.get("feedback_circuit_id"):
                    feedback=shared/"feedback-circuits"/metadata["feedback_circuit_id"]
                    checked_files(feedback,json.loads((feedback/"manifest.json").read_text()))
                    shutil.copytree(feedback,temporary/"feedback-circuits"/metadata["feedback_circuit_id"])
                graph, _ = resolve_graph(root, metadata["graph_id"])
                shutil.copytree(graph, temporary / "prepared" / metadata["graph_id"])
                write_json(temporary / "prepared" / "latest.json", {"graph_id": metadata["graph_id"]})
            loader="fly_brain.visual_dopamine.motor_policy.MotorPolicy" if metadata.get("schema")=="seven-motor-checkpoint-v1" else "fly_brain.visual_dopamine.policy.VisualDopaminePolicy"
            (temporary / "README.txt").write_text("Load checkpoint/ with "+loader+". "
                "Set MALECNS_HOME to this bundle when graph assets are included, or to your existing shared MaleCNS root. "
                "See the checkpoint task scope and independent evaluation; exporting does not qualify a pickup skill.\n")
            os.rename(temporary, destination)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
        return {"export":str(destination),"included_graph":include_graph,"schema":metadata["schema"]}
    if destination.exists():
        raise FileExistsError("Exports never overwrite an existing directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    try:
        checkpoint = temporary / "checkpoint"
        shutil.copytree(source, checkpoint, ignore=shutil.ignore_patterns("optimizer.pt"))
        metadata.pop("files", None)
        finish_checkpoint(checkpoint, metadata)
        if include_graph and metadata.get("graph_id"):
            graph, _ = resolve_graph(root, metadata["graph_id"])
            shutil.copytree(graph, temporary / "prepared" / metadata["graph_id"])
            write_json(temporary / "prepared" / "latest.json", {"graph_id": metadata["graph_id"]})
        (temporary / "README.txt").write_text(
            "Install the fly-brain package. Load checkpoint/ with MALECNS_HOME pointing to this bundle\n"
            "if prepared/ is included, otherwise point it at your existing verified shared data root.\n")
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"export": str(destination), "included_graph": include_graph and bool(metadata.get("graph_id"))}


def export_core(source, destination):
    """Export learned neuron dynamics separately from robot-specific adapters."""
    source, destination = Path(source), Path(destination)
    metadata = json.loads((source / 'manifest.json').read_text())
    checked_files(source, metadata)
    if metadata['architecture']['kind'] != 'malecns':
        raise ValueError('A GRU baseline has no MaleCNS core to export')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.core-', dir=destination.parent))
    try:
        weights = load_file(str(source / 'model.safetensors'))
        keys = ['response.weight', 'leak_logit', 'gain_logit']
        save_file({key: weights[key] for key in keys}, str(temporary / 'core.safetensors'))
        manifest = artifact_manifest(temporary, {
            'schema': 'malecns-core-v1', 'graph_id': metadata['graph_id'],
            'channels': metadata['architecture']['channels'], 'updates': metadata['architecture']['updates'],
            'source_checkpoint': metadata['name'], 'training_method': metadata['training_method'],
            'robot_adapters_included': False, 'data_license': 'CC-BY-4.0',
            'source': 'https://male-cns.janelia.org/download/',
        })
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'core': str(destination), **manifest}


def apply_core(model, directory, graph_id):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    checked_files(directory, manifest)
    if (manifest['schema'] != 'malecns-core-v1' or manifest['graph_id'] != graph_id
            or manifest['channels'] != model.config['channels'] or manifest['updates'] != model.config['updates']):
        raise ValueError('Core graph/dynamics configuration is incompatible')
    weights = load_file(str(directory / 'core.safetensors'))
    if set(weights) != {'response.weight', 'leak_logit', 'gain_logit'}:
        raise ValueError('Unexpected core weight keys')
    model.load_state_dict(weights, strict=False)
