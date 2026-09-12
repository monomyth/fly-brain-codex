import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from head_training_runtime import retention_loss
from continue_head_training import fidelity_eligible


def test_retention_is_zero_for_the_baseline_outputs():
    scores=torch.tensor([[.01]*7,[-.01]*7],dtype=torch.float64)
    assert abs(retention_loss(scores,scores,torch.tensor([True,True]),torch.ones(2),1440.).item())<1e-12


def test_held_outputs_are_not_anchored():
    scores=torch.zeros((2,7),dtype=torch.float64,requires_grad=True)
    reference=torch.ones((2,7),dtype=torch.float64)*.001
    value=retention_loss(scores,reference,torch.tensor([True,False]),torch.ones(2),1440.)
    value.backward()
    assert torch.count_nonzero(scores.grad[0])>0
    assert torch.count_nonzero(scores.grad[1])==0


def test_selection_requires_preserved_pregrasp_behavior():
    recipe={'selection_fidelity_limits':{'joint_rmse_deg':.1,'joint_max_abs_deg':.5,'gripper_aperture_mae_mm':1.}}
    assert fidelity_eligible({'nonheld_fidelity':{'joint_rmse_deg':.02,'joint_max_abs_deg':.3,'gripper_aperture_mae_mm':.1}},recipe)
    assert not fidelity_eligible({'nonheld_fidelity':{'joint_rmse_deg':.2,'joint_max_abs_deg':.3,'gripper_aperture_mae_mm':.1}},recipe)
    assert not fidelity_eligible({},recipe)
    assert fidelity_eligible({}, {})


def test_recovery_states_can_change_without_loosening_legacy_limits():
    recipe={'retention_selection_group':'legacy','selection_fidelity_limits':{'joint_max_abs_deg':.5}}
    stats={'nonheld_fidelity':{'joint_max_abs_deg':2.},'groups':{'legacy':{'nonheld_fidelity':{'joint_max_abs_deg':.3}}}}
    assert fidelity_eligible(stats,recipe)
    stats['groups']['legacy']['nonheld_fidelity']['joint_max_abs_deg']=.6
    assert not fidelity_eligible(stats,recipe)


def test_legacy_retention_leaves_new_recovery_targets_free():
    from head_training_runtime import retention_anchor_rows
    import numpy as np
    # A new recovery frame must not be trained toward the failed old response.
    result=retention_anchor_rows([True,True,True,False],[False,False,True,False],[False,True,False,False],'legacy')
    assert np.array_equal(result,[True,False,False,False])
    assert np.array_equal(retention_anchor_rows([True,True],[False,False],[False,True]),[True,True])
