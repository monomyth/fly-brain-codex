"""Optional recurrent PPO fine-tuning after an independently evaluated BC policy."""
import json
import os
import sys
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal
from .assets import write_json, canonical_hash, sha256_file
from .checkpoint import finish_checkpoint, save_weights
from .dataset import EpisodeWriter
from .evaluate import released
from .policy import Policy
from .simulator import configure_episode, floor_limited_action, release_for_cleanup


def gae(rewards, values, terminals, bootstrap=0., gamma=.99, lam=.95):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype)
    next_value = torch.as_tensor(bootstrap, dtype=rewards.dtype)
    for i in reversed(range(len(rewards))):
        alive = 1 - terminals[i]
        delta = rewards[i] + gamma * next_value * alive - values[i]
        carry = delta + gamma * lam * alive * carry
        advantages[i] = carry
        next_value = values[i]
    return advantages, advantages + values


def action_log_probability(output, latent, opened, log_std):
    # The invertible tanh action transform cancels from the old/new PPO ratio.
    return Normal(output[..., :6], log_std.exp()).log_prob(latent).sum(-1) + Bernoulli(logits=output[..., 6]).log_prob(opened)


def ppo_update(model, critic, log_std, rollout, optimizer, passes=2, chunk_length=32, clip=.2):
    advantages, returns = gae(rollout["rewards"], rollout["values"], rollout["terminals"], rollout["bootstrap"])
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    losses = []
    for _ in range(passes):
        # Recompute the recurrent history under the current parameters on every pass.
        state = model.initial_state()
        for start in range(0, len(advantages), chunk_length):
            stop = min(len(advantages), start + chunk_length)
            outputs = []
            for x in rollout["features"][start:stop]:
                output, state = model.step(x.unsqueeze(0), state)
                outputs.append(output.squeeze(0))
            output = torch.stack(outputs)
            log_probability = action_log_probability(output, rollout["latent"][start:stop], rollout["opened"][start:stop], log_std)
            ratio = (log_probability - rollout["log_probabilities"][start:stop]).exp()
            adv = advantages[start:stop]
            actor_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            value_loss = (critic(output).squeeze(-1) - returns[start:stop]).square().mean()
            entropy = (Normal(output[:, :6], log_std.exp()).entropy().sum(-1) + Bernoulli(logits=output[:, 6]).entropy()).mean()
            loss = actor_loss + .5 * value_loss - .001 * entropy
            if not torch.isfinite(loss):
                raise ValueError("Non-finite PPO objective")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(critic.parameters()) + [log_std], 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                log_std.clamp_(-4., -.5)
            state = state.detach()
            losses.append(float(loss.detach()))
    return {"ppo_loss": float(np.mean(losses)), "return": float(rollout["rewards"].sum())}


def reward(previous, current, action, previous_action):
    value = .03 * (current.get("clearance_mm", 0) - previous.get("clearance_mm", 0))
    value += 2 * (current.get("hold_seconds", 0) - previous.get("hold_seconds", 0))
    value += .02 if current.get("held") else 0
    if current.get("success") and not previous.get("success"):
        value += 30
    if current.get("success") and released(current) and not released(previous):
        value += 10
    if current.get("dropped") and not current.get("success"):
        value -= 10
    if previous_action is not None:
        value -= .001 * float(np.square(np.array(action["joints_deg"]) - previous_action["joints_deg"]).sum())
    return float(value)


def rollout_episode(client, policy, critic, log_std, task_spec, dataset, max_steps=256, floor_projection=False):
    task = configure_episode(client, task_spec)
    first = client.observe(images=policy.mode == "vision")
    policy.reset(first)
    policy.act(first); policy.reset(first)
    rows = {key: [] for key in ("features", "latent", "opened", "log_probabilities", "values", "rewards", "terminals")}
    writer = EpisodeWriter(dataset, task, "malecns", policy.hz, {"training": "on_policy_ppo"})
    previous_action = None
    bootstrap = 0.
    try:
        client.call("rebot_experiment_control", action="start")
        client.connect(provenance="malecns", model_id="fly-brain-ppo-training")
        previous = client.call("rebot_get_evaluation")
        state = policy.model.initial_state()
        for step in range(max_steps):
            began = time.monotonic()
            observation = client.observe(images=policy.mode == "vision")
            encoded = policy.encoder.encode(observation)
            features = torch.from_numpy((encoded - policy.mean) / policy.std)
            with torch.no_grad():
                output, state = policy.model.step(features.unsqueeze(0), state)
                output = output.squeeze(0)
                latent = Normal(output[:6], log_std.exp()).sample()
                opened = Bernoulli(logits=output[6]).sample()
                lp = action_log_probability(output, latent, opened, log_std)
                value = critic(output).squeeze()
            action = policy.decoder.decode(observation, latent.tanh().numpy(), opened.item())
            projection = None
            if floor_projection:
                _, action, projection = floor_limited_action(client, observation, action)
            else:
                client.action(observation, action)
            time.sleep(max(0, 1 / policy.hz - (time.monotonic() - began)))
            current = client.call("rebot_get_evaluation")
            terminal = (current.get("success") and released(current)) or current["phase"] not in ("running", "completed")
            for key, entry in {"features": features, "latent": latent, "opened": opened, "log_probabilities": lp,
                               "values": value, "rewards": torch.tensor(reward(previous, current, action, previous_action)),
                               "terminals": torch.tensor(float(terminal))}.items():
                rows[key].append(entry)
            writer.append(observation, action, evaluation=current, intervention=projection)
            previous_action = action; previous = current
            if terminal:
                break
        if rows["terminals"] and not rows["terminals"][-1].item():
            observation = client.observe(images=policy.mode == "vision")
            features = torch.from_numpy((policy.encoder.encode(observation) - policy.mean) / policy.std)
            with torch.no_grad():
                output, _ = policy.model.step(features.unsqueeze(0), state)
                bootstrap = critic(output).squeeze().item()
        writer.finish({"success": bool(previous.get("success") and released(previous)), "evaluation": previous})
    except BaseException as error:
        writer.finish({"success": False, "error": str(error)}, complete=False)
        raise
    finally:
        primary_error = sys.exception()
        cleanup_error = release_for_cleanup(client)
        try:
            client.call("rebot_experiment_control", action="pause")
        except Exception:
            if primary_error is None:
                raise
    return {**{key: torch.stack(value) for key, value in rows.items()}, "bootstrap": bootstrap}


def train_rl(client, args, root, tasks):
    gate = json.loads(args.gate_report.read_text())
    policy = Policy(args.checkpoint, root)
    if (gate.get("checkpoint") != policy.metadata["name"] or gate.get("ablation") != "none"
            or gate.get("teacher_verified_attempts", 0) < 100
            or gate.get("teacher_verified_successes", 0) / gate["teacher_verified_attempts"] < .9):
        raise ValueError("RL requires a matching independent report with >=90% success over >=100 teacher-verified held-out trials")
    if policy.metadata["architecture"]["kind"] != "malecns":
        raise ValueError("This fine-tuning entry point requires a MaleCNS checkpoint")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    working = output.with_name(output.name + ".partial"); working.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    critic = nn.Linear(7, 1)
    log_std = nn.Parameter(torch.full((6,), -2.))
    optimizer = torch.optim.Adam(list(policy.model.parameters()) + list(critic.parameters()) + [log_std], lr=args.learning_rate)
    history = []
    for i, task in enumerate(tasks):
        rollout = rollout_episode(client, policy, critic, log_std, {**task, "input_mode": policy.mode}, working / "rollouts", args.max_steps, bool(gate.get("floor_projection")))
        result = ppo_update(policy.model, critic, log_std, rollout, optimizer)
        history.append(result); write_json(working / "training.json", history)
        print(json.dumps({"rollout": i + 1, **result}), flush=True)
    save_weights(policy.model, working)
    from safetensors.torch import save_file
    save_file({"critic.weight": critic.weight.detach(), "critic.bias": critic.bias.detach(), "log_std": log_std.detach()}, str(working / "training-heads.safetensors"))
    torch.save(optimizer.state_dict(), working / "optimizer.pt")
    metadata = {key: value for key, value in policy.metadata.items() if key not in ("files", "schema", "best_validation_loss")}
    metadata.update(name=output.name, training_method="recurrent_ppo", closed_loop_validated=False,
                    parent_checkpoint=policy.metadata["name"], seed=args.seed, floor_projection=bool(gate.get("floor_projection")), gate_report_sha256=sha256_file(args.gate_report))
    finish_checkpoint(working, metadata); os.rename(working, output)
    return {"checkpoint": str(output), "rollouts": len(history), "autonomous_evaluation_required": True}
