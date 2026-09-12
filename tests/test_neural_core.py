import numpy as np
import torch
from fly_brain.neural_core import FixedSparseMultiply, FlyController, SparseOperator
from fly_brain.train_bc import imitation_loss


def test_sparse_gradient_matches_numerical_derivative(graph):
    matrix = graph.matrix.astype(np.float64)
    x = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: FixedSparseMultiply.apply(value, matrix, matrix.T.tocsr()), (x,))


def test_torch_and_scipy_backends_agree(graph):
    state = torch.randn(2, 5, 3, requires_grad=True)
    a = SparseOperator(graph.matrix, "scipy")(state)
    b = SparseOperator(graph.matrix, "torch")(state)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a.square().sum(), state)[0]
    gb = torch.autograd.grad(b.square().sum(), state)[0]
    torch.testing.assert_close(ga, gb)


def test_outputs_depend_on_graph_and_episode_reset(graph):
    model = FlyController(graph, 4, updates=2)
    x = torch.tensor([[1., -1., 2., .5]])
    with torch.no_grad():
        state = None
        for _ in range(8):
            out, state = model.step(x, state)
        _, other = model.step(-x, model.initial_state())
        assert not torch.allclose(state, other)
        a, sa = model.step(x, model.initial_state())
        b, sb = model.step(x, model.initial_state())
        torch.testing.assert_close(a, b); torch.testing.assert_close(sa, sb)
        model.graph_enabled = False
        state = None
        for _ in range(8):
            disabled, state = model.step(x, state)
        assert torch.count_nonzero(model.readout(state)) == 0
        assert not torch.allclose(out, disabled)


def test_training_changes_adapters_but_not_anatomy(graph):
    model = FlyController(graph, 4, seed=31)
    before = graph.matrix.data.copy()
    sensory = model.sensory.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    state = None
    for _ in range(5):
        output, state = model.step(torch.randn(1, 4), state)
    imitation_loss(output, torch.ones(1, 7) * .5).backward(); optimizer.step()
    assert not torch.equal(sensory, model.sensory.weight)
    np.testing.assert_array_equal(before, graph.matrix.data)
    assert all(not key.startswith("operator.") for key in model.state_dict())


def test_reservoir_freezes_only_core(graph):
    model = FlyController(graph, 4); model.freeze_reservoir()
    assert not model.sensory.weight.requires_grad
    assert not model.leak_logit.requires_grad
    assert model.head[0].weight.requires_grad


def test_reservoir_cache_invalidates_core_but_reuses_head_updates(graph, tmp_path):
    from fly_brain.train_bc import cache_reservoir
    model = FlyController(graph, 4, channels=2); model.freeze_reservoir()
    x = torch.randn(5, 4)
    first = {'x': x}
    assert cache_reservoir(model, [first], tmp_path, 'graph')['built'] == 1
    with torch.no_grad(): model.head[0].weight.add_(1)
    second = {'x': x.clone()}
    assert cache_reservoir(model, [second], tmp_path, 'graph')['hits'] == 1
    torch.testing.assert_close(first['reservoir'], second['reservoir'])
    with torch.no_grad(): model.sensory.weight.add_(.1)
    third = {'x': x.clone()}
    assert cache_reservoir(model, [third], tmp_path, 'graph')['built'] == 1
    assert not torch.allclose(first['reservoir'], third['reservoir'])
