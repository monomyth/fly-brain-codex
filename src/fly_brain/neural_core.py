"""Frozen anatomical operator with learned sensory projection and motor decoding."""
import math
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import torch
from torch import nn


_SPARSE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fly-csr")


def sparse_product(matrix, values):
    # SciPy's generic few-column SpMM is much slower than its vector kernel on
    # Apple Silicon. Independent vector products release the GIL and keep the
    # exact same fixed operator and gradients; larger batches use normal SpMM.
    if 1 < values.shape[1] <= 4:
        columns = [np.ascontiguousarray(values[:, i]) for i in range(values.shape[1])]
        return np.column_stack(list(_SPARSE_POOL.map(matrix.dot, columns)))
    return np.asarray(matrix @ values)


class FixedSparseMultiply(torch.autograd.Function):
    """CPU CSR SpMM with an explicit transpose gradient, avoiding sparse-weight grads.

    SciPy is used for the fixed graph on CPU. PyTorch tensors and all trainable
    adapters stay differentiable; no gradient is claimed for anatomical weights.
    """
    @staticmethod
    def forward(ctx, states, operator, transpose):
        if states.device.type != "cpu":
            raise ValueError("SciPy graph backend requires CPU tensors")
        ctx.transpose = transpose
        return torch.from_numpy(sparse_product(operator, states.detach().numpy()))

    @staticmethod
    def backward(ctx, gradient):
        return torch.from_numpy(sparse_product(ctx.transpose, gradient.contiguous().numpy())), None, None


class SparseOperator(nn.Module):
    def __init__(self, matrix, backend="scipy"):
        super().__init__()
        self.backend = backend
        self.n = matrix.shape[0]
        if backend == "scipy":
            self.matrix = matrix.astype(np.float32).tocsr()
            self.transpose = self.matrix.T.tocsr()
        elif backend == "torch":
            tensor = torch.sparse_csr_tensor(torch.from_numpy(matrix.indptr.astype(np.int64)),
                                             torch.from_numpy(matrix.indices.astype(np.int64)),
                                             torch.from_numpy(matrix.data.astype(np.float32)), size=matrix.shape, check_invariants=True)
            self.register_buffer("matrix", tensor, persistent=False)
        else:
            raise ValueError("Graph backend must be scipy or torch")

    def forward(self, state):
        # [batch, neurons, channels] -> [neurons, batch*channels]. No dense adjacency.
        batch, neurons, channels = state.shape
        columns = state.permute(1, 0, 2).reshape(neurons, batch * channels).contiguous()
        if self.backend == "scipy":
            result = FixedSparseMultiply.apply(columns, self.matrix, self.transpose)
        else:
            result = torch.sparse.mm(self.matrix, columns)
        return result.reshape(neurons, batch, channels).permute(1, 0, 2)


class FlyController(nn.Module):
    def __init__(self, graph, input_dim, channels=1, updates=2, hidden_dim=64, backend="scipy", seed=0):
        super().__init__()
        if channels < 1 or not 1 <= updates <= 4:
            raise ValueError("Channels must be positive and graph updates must be 1–4")
        self.config = dict(kind="malecns", input_dim=input_dim, channels=channels, updates=updates,
                           hidden_dim=hidden_dim, backend=backend, seed=seed)
        self.n = graph.matrix.shape[0]
        self.channels = channels
        self.updates = updates
        self.operator = SparseOperator(graph.matrix, backend)
        self.register_buffer("inputs", torch.as_tensor(graph.inputs.copy(), dtype=torch.long), persistent=False)
        self.register_buffer("outputs", torch.as_tensor(graph.outputs.copy(), dtype=torch.long), persistent=False)
        self.register_buffer("neuron_features", torch.as_tensor(graph.features.copy()), persistent=False)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.sensory = nn.Linear(input_dim, len(graph.inputs) * channels)
            self.response = nn.Linear(graph.features.shape[1], channels, bias=False)
            nn.init.zeros_(self.response.weight)
            self.read_norm = nn.LayerNorm(len(graph.outputs) * channels)
            self.head = nn.Sequential(nn.Linear(len(graph.outputs) * channels, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 7))
        # A one-channel baseline retains slow contact history. Multiple channels
        # span fast sensory feedback and slow memory, still through the same graph.
        initial_leak = torch.full((1,), -3.0) if channels == 1 else torch.linspace(2.2, -3.0, channels)
        self.leak_logit = nn.Parameter(initial_leak)
        self.gain_logit = nn.Parameter(torch.full((channels,), 0.0))
        self.graph_enabled = True
        self.inputs_enabled = True

    def initial_state(self, batch=1):
        return self.sensory.weight.new_zeros(batch, self.n, self.channels)

    def advance_state(self, features, state=None):
        if state is None:
            state = self.initial_state(len(features))
        drive = self.sensory(features).reshape(len(features), len(self.inputs), self.channels)
        if not self.inputs_enabled:
            drive = drive * 0
        injection = torch.zeros_like(state).index_copy(1, self.inputs, drive)
        bias = self.response(self.neuron_features).unsqueeze(0)
        leak = torch.sigmoid(self.leak_logit)
        gain = 1.8 * torch.sigmoid(self.gain_logit)
        for _ in range(self.updates):
            message = self.operator(state) if self.graph_enabled else state * 0
            state = (1 - leak) * state + leak * torch.tanh(gain * message + injection + bias)
        return state

    def readout(self, state):
        return state.index_select(1, self.outputs).flatten(1)

    def step(self, features, state=None):
        state = self.advance_state(features, state)
        return self.head(self.read_norm(self.readout(state))), state

    def freeze_reservoir(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("head.") or name.startswith("read_norm."))


class RecurrentBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, seed=0):
        super().__init__()
        self.config = dict(kind="gru", input_dim=input_dim, hidden_dim=hidden_dim, seed=seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.cell = nn.GRUCell(input_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, 7)
        self.hidden_dim = hidden_dim

    def initial_state(self, batch=1):
        return self.head.weight.new_zeros(batch, self.hidden_dim)

    def step(self, features, state=None):
        state = self.cell(features, self.initial_state(len(features)) if state is None else state)
        return self.head(state), state


def make_model(config, graph=None):
    config = dict(config)
    kind = config.pop("kind")
    if kind == "malecns":
        if graph is None:
            raise ValueError("MaleCNS policy requires its pinned graph")
        return FlyController(graph, **config)
    if kind == "gru":
        return RecurrentBaseline(**config)
    raise ValueError(f"Unknown architecture {kind}")


def benchmark(model, steps=30):
    torch.set_num_threads(4)
    model.eval()
    rng = torch.Generator().manual_seed(817)
    samples = torch.randn(steps + 5, 1, model.config["input_dim"], generator=rng)
    state = model.initial_state()
    timings = []
    with torch.no_grad():
        for i, observation in enumerate(samples):
            started = time.perf_counter()
            output, state = model.step(observation, state)
            if not torch.isfinite(output).all() or not torch.isfinite(state).all():
                raise ValueError("Non-finite full-graph activity")
            if i >= 5:
                timings.append(1000 * (time.perf_counter() - started))
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    state = model.initial_state()
    for step in range(4):
        output, state = model.step(samples[step], state)
    output.square().mean().backward()
    backward_ms = 1000 * (time.perf_counter() - started)
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    result = {"steps": steps, "p50_ms": float(np.percentile(timings, 50)),
              "p95_ms": float(np.percentile(timings, 95)), "maximum_ms": max(timings),
              "four_step_forward_backward_ms": backward_ms,
              "finite_gradients": all(torch.isfinite(g).all().item() for g in gradients),
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "backend": model.config.get("backend", "torch"), "torch_threads": torch.get_num_threads(),
              "meets_inference_only_200ms_target": float(np.percentile(timings, 95)) < 200,
              "note": "Inference only; live camera, IPC, and full-loop timing are measured separately."}
    model.zero_grad(set_to_none=True)
    return result
