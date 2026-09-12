"""Continue constrained neural-head fitting from a verified portable bundle."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'src'))
os.environ['CUDA_CACHE_PATH'] = str(root / '.cache/cuda')
os.environ['TORCH_HOME'] = str(root / '.cache/torch')
import numpy as np
import torch
from scipy.optimize import minimize
from fly_brain.assets import checked_files, write_json, artifact_manifest, sha256_file
from fly_brain.visual_dopamine.feedback_training import validation_score
from head_training_runtime import load_bundle, loss, statistics


def fidelity_eligible(statistics,recipe):
    limits=recipe.get('selection_fidelity_limits')
    if not limits:return True
    if recipe.get('retention_selection_group')=='legacy':statistics=statistics.get('groups',{}).get('legacy',{})
    fidelity=statistics.get('nonheld_fidelity')
    return fidelity is not None and all(fidelity.get(key,float('inf'))<=value for key,value in limits.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=8000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.output.exists() or not args.output.resolve().is_relative_to(root):
        raise ValueError('Use positive iterations and a new directory inside the experiment')
    checked_files(root, json.loads((root / 'manifest.json').read_text()))
    if not json.loads((root / 'benchmark.json').read_text())['passed']:
        raise ValueError('CUDA parity benchmark must pass first')
    torch.set_num_threads(2)
    model, fk, data, recipe = load_bundle(root, 'cuda')
    post, pre = torch.nonzero(model.mask, as_tuple=True)
    n, mb = len(post), model.excitability.numel()
    def vector(gradient=False):
        def array(parameter):
            return parameter.grad if gradient else parameter.detach()
        return torch.cat((array(model.weights)[post, pre], array(model.excitability),
                          array(model.motor_strength))).detach().cpu().numpy().copy()
    def assign(value):
        value = torch.as_tensor(value, dtype=torch.float64, device='cuda')
        with torch.no_grad():
            model.weights.zero_()
            model.weights[post, pre] = value[:n]
            model.excitability.copy_(value[n:n+mb])
            model.motor_strength.copy_(value[n+mb:])
    start = vector()
    low = np.r_[np.zeros(n), np.full(mb, -2.), np.zeros(len(start)-n-mb)]
    high = np.r_[recipe['weight_multiple'] * model.base[post, pre].cpu().numpy(),
                 np.full(mb, 2.), recipe['motor_weight_multiple'] * model.motor_base.cpu().numpy()]
    if np.any(start < low-1e-7) or np.any(start > high+1e-7):
        raise ValueError('Source parameters violate anatomical bounds')
    scale = np.load(root / 'parameter-scaling.npy', allow_pickle=False)
    if scale.shape != start.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid parameter scaling')
    scope = data['held_rows'] if recipe.get('training_scope') == 'held' else ~data['held_rows'] if recipe.get('training_scope') == 'unheld' else torch.ones_like(data['split'], dtype=torch.bool)
    splits = {name: torch.nonzero((data['split'] == index) & scope).flatten()
              for index, name in enumerate(['train', 'validation', 'test'])}
    if any(len(rows) == 0 for rows in splits.values()):
        raise ValueError('Independent training, validation and test rows required')
    def selection():
        rows = splits['validation']
        result = statistics(model, fk, data, rows, recipe)
        groups = {name: rows[data['recovery'][rows] == value]
                  for name, value in [('legacy', False), ('recovery', True)]}
        if recipe['validation_aggregation'] == 'recovery_balanced':
            result['groups'] = {name: statistics(model, fk, data, selected, recipe)
                                for name, selected in groups.items() if len(selected)}
        return result
    def selection_score(stats):
        if stats.get('groups'):
            return float(np.mean([selection_score(group) for group in stats['groups'].values()]))
        score = validation_score(stats, True)
        critical = stats.get('grasp_precision')
        if critical:
            score += recipe.get('precision_validation_weight', 0.) * (critical['xy_rmse_mm']/3 + critical['height_rmse_mm']/3 + critical['orientation_rmse_deg']/5 + 3*(1-critical['gripper_accuracy']))
        return score
    args.output.mkdir(parents=True)
    start_time = time.perf_counter()
    initial = selection()
    best_score = selection_score(initial)
    best = start.copy()
    best_call = 0
    calls = 0
    history = []
    def record(value, training_loss):
        nonlocal best_score, best, best_call
        stats = selection()
        score = selection_score(stats)
        eligible=fidelity_eligible(stats,recipe)
        if eligible and score < best_score:
            best_score, best, best_call = score, value.copy(), calls
            temporary = args.output / '.best-head.tmp.npz'
            np.savez(temporary, weights=model.weights.detach().cpu().numpy(), excitability=model.excitability.detach().cpu().numpy(), motor_synapses=(model.motor_strength*model.motor_signs).detach().cpu().numpy().astype(np.float32))
            temporary.replace(args.output / 'best-head.npz')
        row = {'calls': calls, 'elapsed_seconds': time.perf_counter()-start_time,
               'loss': training_loss, 'selection_score': score, 'selection': stats,
               'best_call': best_call,'fidelity_eligible':eligible}
        history.append(row)
        write_json(args.output / 'progress.json', row)
        print(json.dumps(row), flush=True)
    def objective(scaled):
        nonlocal calls
        value = scaled / scale
        assign(value)
        model.zero_grad(set_to_none=True)
        objective_loss = loss(model, fk, data, splits['train'], recipe)
        objective_loss.backward()
        gradient = vector(gradient=True)
        measured_loss = float(objective_loss.detach().cpu())
        if not np.isfinite(measured_loss) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite loss or gradient')
        calls += 1
        if calls == 1 or calls % 100 == 0:
            record(value, measured_loss)
        return measured_loss, gradient / scale
    result = minimize(objective, start*scale, jac=True, bounds=list(zip(low*scale, high*scale)),
                      method='L-BFGS-B', options={'maxiter': args.iterations, 'maxfun': args.iterations*3,
                      'ftol': 1e-11, 'gtol': 1e-7, 'maxls': 40, 'maxcor': 50})
    assign(result.x/scale)
    record(result.x/scale, float(result.fun))
    assign(best)
    # Only fitted parameters may change; original sparse wiring, signs, and readout stay fixed.
    initial_state = torch.load(root / 'head-state.pt', map_location='cpu', weights_only=True)
    for name, value in model.state_dict().items():
        if name in recipe['parameter_names']:
            continue
        original = initial_state[name].double() if value.is_floating_point() else initial_state[name]
        actual = value.detach().cpu()
        if actual.layout != torch.strided:
            actual, original = actual.to_dense(), original.to_dense()
        if not torch.equal(actual, original):
            raise AssertionError('Fixed buffer changed: '+name)
    final_vector = vector()
    if np.any(final_vector < low-1e-10) or np.any(final_vector > high+1e-10):
        raise AssertionError('Fitted parameters violate anatomical bounds')
    if torch.count_nonzero(model.weights[~model.mask]):
        raise AssertionError('New connections were introduced')
    np.savez(args.output / 'learned-head.npz', weights=model.weights.detach().cpu().numpy(),
             excitability=model.excitability.detach().cpu().numpy(),
             motor_synapses=(model.motor_strength*model.motor_signs).detach().cpu().numpy().astype(np.float32))
    metrics = {name: statistics(model, fk, data, rows, recipe) for name, rows in splits.items()}
    if recipe.get('pregrasp_retention_strength',0):
        np.savez(args.output/'retention-supervision.npz',reference_scores=data['reference_scores'].cpu().numpy(),anchor_rows=data['anchor_rows'].cpu().numpy())
    if recipe.get('held_orientation_objective')=='cube_up':
        np.savez(args.output/'orientation-supervision.npz',**{key:data[key].cpu().numpy() for key in ['held_rows','relative_cube_up','current_joints_deg','relative_cube_position_mm','relative_cube_rotation','cube_size_mm'] if key in data})
    report = {'loss_recipe':dict(recipe),'source_model_sha256': recipe['model_sha256'], 'source_bundle_manifest_sha256': sha256_file(root/'manifest.json'),
              'trainer_sha256': sha256_file(__file__), 'benchmark': json.loads((root/'benchmark.json').read_text()),
              'iterations_requested': args.iterations, 'optimizer_iterations': int(result.nit), 'objective_calls': calls,
              'optimizer_message': str(result.message), 'elapsed_seconds': time.perf_counter()-start_time,
              'initial_selection': initial, 'selection_statistics': selection(), 'best_call': best_call,
              'best_selection_score': best_score, 'metrics': metrics, 'fixed_buffers_unchanged': True,
              'parameter_bounds_passed': True, 'training_device': 'cuda', 'torch_version': torch.__version__,
              'gpu': torch.cuda.get_device_name(0), 'closed_loop_validated': False,
              'selection': 'Equal legacy/recovery validation groups with declared grasp precision weighting and training scope; test rows scored once after selection'}
    write_json(args.output/'result.json', report)
    write_json(args.output/'training-history.json', history)
    artifact_manifest(args.output, {'schema': 'continued-neural-head-v1', 'source_model_sha256': recipe['model_sha256']})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
