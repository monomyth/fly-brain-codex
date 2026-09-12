"""Command line entry points; optional expensive stages require explicit commands."""
import argparse
import json
import sys
from pathlib import Path
from .assets import home, write_json


def parser():
    result = argparse.ArgumentParser(prog="fly-brain")
    result.add_argument("--home", help="Project artifact root; defaults to this fly-brain checkout/data")
    commands = result.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("assets", help="Verify the three pinned data files")
    assets.add_argument("--download", action="store_true")
    commands.add_parser("import", help="Build or reuse the complete selected neuronal graph")
    overlay = commands.add_parser("prepare-overlay", help="Build project-local 3D soma geometry from existing annotations; no downloads")
    overlay.add_argument("--graph-id")
    bench = commands.add_parser("benchmark", help="Measure full graph forward and backward operations")
    bench.add_argument("--steps", type=int, default=30)
    bench.add_argument("--updates", type=int, default=2)
    bench.add_argument("--channels", type=int, default=1)
    bench.add_argument("--backend", choices=["scipy", "torch"], default="scipy")
    bench.add_argument("--output", type=Path)
    for name in ("collect", "dagger"):
        p = commands.add_parser(name)
        p.add_argument("--dataset", required=True, type=Path)
        p.add_argument("--episodes", type=int, default=50)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--sizes", type=float, nargs="+", default=[50])
        p.add_argument("--xy-radius", type=float, default=25)
        p.add_argument("--yaw-range", type=float, default=5)
        p.add_argument("--hz", type=float, default=5)
        p.add_argument("--no-images", action="store_true")
        p.add_argument("--perturbation-degrees",type=float,default=0.,help="Teacher-collection joint perturbations during approach only")
        p.add_argument("--tasks", type=Path, help="Explicit JSON array of task overrides")
        p.add_argument("--control-directory", type=Path)
        p.add_argument("--binary", help="Native simulator binary; omitted starts the standard Codex clone")
        if name == "dagger":
            p.add_argument("--checkpoint", required=True, type=Path)
            p.add_argument("--teacher-probability", type=float, default=.5)
            p.add_argument("--sensory-host",help="SSH/CUDA host, or local to override runtime.json")
    split = commands.add_parser("split")
    split.add_argument("dataset", type=Path)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--include-dagger-failures", action="store_true")
    train = commands.add_parser("train-bc")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--mode", choices=["state", "vision"], default="state")
    train.add_argument("--kind", choices=["malecns", "gru"], default="malecns")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--sequence-length", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--reservoir", action="store_true")
    train.add_argument("--updates", type=int, default=2)
    train.add_argument("--channels", type=int, default=1)
    train.add_argument("--resume", type=Path, help="Warm start from checkpoint weights")
    train.add_argument("--resume-optimizer", action="store_true")
    train.add_argument("--graph-id")
    for name in ("evaluate", "run"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--tasks", type=Path)
        p.add_argument("--episodes", type=int, default=1)
        p.add_argument("--seed", type=int, default=10000)
        p.add_argument("--sizes", type=float, nargs="+", default=[50])
        p.add_argument("--control-directory", type=Path)
        p.add_argument("--binary")
        p.add_argument("--no-release", action="store_true")
        p.add_argument("--record-images", action="store_true")
        if name == "run":
            p.add_argument("--current-episode", action="store_true", help="Use the cube already placed in the connected app; do not reset")
            p.add_argument("--episode-id", help="Refuse to run if the placed episode changed while loading")
        p.add_argument("--floor-projection", action="store_true", help="Scale floor-rejected actions toward the current pose; log every adjustment")
        p.add_argument("--ablation", choices=["none", "disconnect-inputs", "disable-graph"], default="none")
    export = commands.add_parser("export")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--include-graph", action="store_true")
    rewire = commands.add_parser("rewire")
    rewire.add_argument("--seed", type=int, default=0)
    rewire.add_argument("--swaps-per-edge", type=float, default=1)
    rl = commands.add_parser("train-rl", help="Optional PPO, gated by a successful independent BC evaluation")
    rl.add_argument("--checkpoint", required=True, type=Path)
    rl.add_argument("--gate-report", required=True, type=Path)
    rl.add_argument("--output", required=True, type=Path)
    rl.add_argument("--episodes", type=int, default=5)
    rl.add_argument("--max-steps", type=int, default=256)
    rl.add_argument("--seed", type=int, default=0)
    rl.add_argument("--learning-rate", type=float, default=.0001)
    rl.add_argument("--control-directory", type=Path)
    rl.add_argument("--binary")
    tasks = commands.add_parser("tasks", help="Export teacher-verified task configurations from an episode split")
    tasks.add_argument("dataset", type=Path)
    tasks.add_argument("--split", choices=["train", "validation", "test"], default="test")
    tasks.add_argument("--output", required=True, type=Path)
    audit = commands.add_parser("audit-dataset")
    audit.add_argument("dataset", type=Path)
    audit.add_argument("--output", type=Path)
    core = commands.add_parser("export-core")
    core.add_argument("checkpoint", type=Path)
    core.add_argument("destination", type=Path)
    visual = commands.add_parser("visual-prepare", help="Prepare annotated retinal, dopamine-plastic and leg-motor routes")
    visual.add_argument("--graph-id")
    visual = commands.add_parser("visual-collect", help="Record camera images and actual base-response outcomes")
    visual.add_argument("--output", required=True, type=Path)
    visual.add_argument("--limit", type=int)
    visual.add_argument("--base-prefixes", type=float, nargs="+", default=[0.])
    visual = commands.add_parser("visual-train", help="Train PAM01-gated plasticity and compare dopamine-disabled controls")
    visual.add_argument("--dataset", required=True, type=Path)
    visual.add_argument("--output", required=True, type=Path)
    visual.add_argument("--trials", type=int, default=8000)
    visual.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    visual.add_argument("--updates", type=int, default=12)
    visual.add_argument("--local-contrast", action="store_true", help="Experimental per-receptor contrast calibration")
    visual.add_argument("--learning-rate", type=float, default=.002)
    visual.add_argument("--exploration-sigma", type=float, default=.35)
    visual = commands.add_parser("visual-run", help="Evaluate a retinal dopamine checkpoint on live base-turn responses")
    visual.add_argument("--checkpoint", required=True, type=Path)
    visual.add_argument("--output", required=True, type=Path)
    visual.add_argument("--tasks", type=Path)
    visual.add_argument("--episodes", type=int, default=1)
    visual.add_argument("--control-directory", type=Path)
    visual.add_argument("--current-episode", action="store_true")
    visual.add_argument("--episode-id")
    visual.add_argument("--max-steps", type=int, default=1)
    visual.add_argument("--learn", action="store_true", help="Explore, apply dopamine-gated updates, and save a new checkpoint")
    visual.add_argument("--record-images", action="store_true", help="Images are always recorded for this controller")
    visual.add_argument("--floor-projection", action="store_true", help="Native floor limiting is always active")
    commands.add_parser("motor-prepare",help="Map named opposing leg-motor pools to all six joints and the gripper")
    motor=commands.add_parser("motor-calibrate",help="Physically test direct stimulation of all fourteen motor pools")
    motor.add_argument("--output",required=True,type=Path)
    motor=commands.add_parser("motor-collect",help="Record camera scenes and measured seven-axis action rewards")
    motor.add_argument("--output",required=True,type=Path)
    motor.add_argument("--limit",type=int)
    motor.add_argument("--resume",action="store_true")
    motor=commands.add_parser("motor-train",help="Train PAM-gated synapses for seven motor response channels")
    motor.add_argument("--dataset",required=True,type=Path)
    motor.add_argument("--output",required=True,type=Path)
    motor.add_argument("--trials",type=int,default=40000)
    motor.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    motor.add_argument("--learning-rate",type=float,default=.0002)
    motor.add_argument("--exploration-sigma",type=float,default=.15)
    motor.add_argument("--uniform-contrast",action="store_true")
    feedback=commands.add_parser("feedback-train",help="Train a camera/body neural controller from complete pickup demonstrations")
    feedback.add_argument("--dataset",required=True,type=Path,nargs="+")
    feedback.add_argument("--configurations",required=True,type=Path)
    feedback.add_argument("--output",required=True,type=Path)
    feedback.add_argument("--iterations",type=int,default=5000)
    feedback.add_argument("--seed",type=int,default=0)
    feedback.add_argument("--sample-hz",type=float,help="Thin training frames to this rate while retaining stage transitions")
    feedback.add_argument("--task-space-loss",action="store_true",help="Supervise tool position and orientation using reference robot kinematics")
    feedback.add_argument("--lift-goal-offset-mm",type=float,default=0.,help="Training-only extra lift margin, with task-space supervision")
    feedback.add_argument("--desired-lift-clearance-mm",type=float,help="Use a common lift target across demonstrations collected at different heights")
    feedback.add_argument("--grasp-goal-height-mm",type=float,help="Training-only world Z of the grasp goal, 22-40 mm")
    feedback.add_argument("--hold-center-mm",type=float,nargs=2,help="Training-only XY target for stable transport/holding")
    feedback.add_argument("--approach-center-mm",type=float,nargs=2,help="Training-only XY target for a common raised approach")
    feedback.add_argument("--feedback-circuit",type=Path,help="Explicit prepared feedback circuit for a controlled readout comparison")
    feedback.add_argument("--transfer-readout",action="store_true",help="Transfer shared weights only after verifying identical sensory and arm pathways")
    feedback.add_argument("--warm-start",type=Path)
    feedback.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    feedback.add_argument("--output-mode",choices=["joint_targets","joint_deltas"],default="joint_targets")
    feedback.add_argument("--learn-motor-synapses",action="store_true")
    feedback.add_argument("--no-dopamine",action="store_true")
    feedback.add_argument("--precondition",action="store_true")
    feedback.add_argument("--refit-adaptation",action="store_true")
    feedback.add_argument("--refit-feature-adaptation",action="store_true",help="Refit neural feature normalization while retaining retinal calibration")
    feedback.add_argument("--sample-weight-policy",choices=["recorded_stage","goal_role"],default="recorded_stage")
    feedback.add_argument("--policy-visited-weight",type=float,default=1.)
    feedback.add_argument("--validation-aggregation",choices=["frames","recovery_balanced"],default="frames")
    feedback.add_argument("--recovery-target-policy",choices=["recorded","reactive"],default="recorded",help="Use recorded recovery waypoints or consistent observable goals with local escape corrections")
    feedback.add_argument("--temporal-sensory",action="store_true",help="Retain timed sensory activity between observations, reset at episode boundaries")
    feedback.add_argument("--retinal-profile",type=Path)
    feedback.add_argument("--initial-frame-weight",type=float,default=1.)
    feedback.add_argument("--first-frame-weight",type=float,default=1.)
    feedback.add_argument("--grasp-frame-weight",type=float,default=1.,help="Training-only emphasis on descent and closure observations")
    feedback.add_argument("--goal-supervision",choices=["recorded","observable","aligned"],default="recorded")
    motor=commands.add_parser("motor-run",help="Run camera-driven six-joint/gripper control in the native simulator")
    motor.add_argument("--checkpoint",required=True,type=Path)
    motor.add_argument("--output",required=True,type=Path)
    motor.add_argument("--tasks",type=Path)
    motor.add_argument("--episodes",type=int,default=1)
    motor.add_argument("--control-directory",type=Path)
    motor.add_argument("--current-episode",action="store_true")
    motor.add_argument("--episode-id")
    motor.add_argument("--stop-after-failures",type=int,help="Optional early rejection after this many failed trials")
    motor.add_argument("--max-steps",type=int,default=120)
    motor.add_argument("--axis",type=int,choices=range(1,8),help="Optional single-axis test: joints1–6, gripper7")
    motor.add_argument("--selection",choices=["all","strongest"],default="all")
    motor.add_argument("--step-degrees",type=float,default=2.)
    motor.add_argument("--step-mm",type=float,default=4.)
    motor.add_argument("--learn",action="store_true")
    motor.add_argument("--settle-seconds",type=float,help="Override post-command wait, 0.05-0.5 seconds; native controller safeguards are unchanged")
    motor.add_argument("--sensory-host",help="SSH/CUDA host, or local to override runtime.json")
    motor.add_argument("--record-images",action="store_true",help="Images are always recorded")
    motor.add_argument("--floor-projection",action="store_true",help="Native floor limiting is always enabled")
    return result


def _main():
    command_parser = parser()
    args = command_parser.parse_args()
    from .assets import local_artifact
    for key,value in vars(args).items():
        if isinstance(value,Path) and key!="home":setattr(args,key,local_artifact(value))
        elif isinstance(value,list) and value and all(isinstance(x,Path) for x in value):setattr(args,key,[local_artifact(x) for x in value])
    if getattr(args, "current_episode", False):
        if not args.control_directory or args.tasks or args.episodes != 1:
            command_parser.error("--current-episode requires --control-directory, one episode, and no --tasks")
    if getattr(args, "episode_id", None) and not getattr(args, "current_episode", False):
        command_parser.error("--episode-id requires --current-episode")
    root = home(args.home)
    if args.command == "motor-prepare":
        from .visual_dopamine.motor_circuit import prepare
        directory,meta=prepare(root);report={"directory":str(directory),**meta}
    elif args.command == "motor-calibrate":
        from .visual_dopamine.motor_experiment import calibrate
        report=calibrate(args.output,root)
    elif args.command == "motor-collect":
        from .visual_dopamine.motor_experiment import collect_responses
        report=collect_responses(args.output,root,args.limit,args.resume)
    elif args.command == "motor-train":
        from .visual_dopamine.motor_training import train
        report=train(args.dataset,args.output,root,args.trials,args.seeds,args.learning_rate,args.exploration_sigma,not args.uniform_contrast)
    elif args.command == "feedback-train":
        from .visual_dopamine.feedback_training import train
        report=train(args.dataset,args.output,json.loads(args.configurations.read_text()),root,args.iterations,args.seed,
                     dopamine_enabled=not args.no_dopamine,warm_start=args.warm_start,device=args.device,train_motor=args.learn_motor_synapses,output_mode=args.output_mode,goal_supervision=args.goal_supervision,precondition=args.precondition,initial_weight=args.initial_frame_weight,first_frame_weight=args.first_frame_weight,refit_adaptation=args.refit_adaptation,retinal_profile=json.loads(args.retinal_profile.read_text()) if args.retinal_profile else None,sample_hz=args.sample_hz,task_space_loss=args.task_space_loss,lift_goal_offset_mm=args.lift_goal_offset_mm,desired_lift_clearance_mm=args.desired_lift_clearance_mm,grasp_frame_weight=args.grasp_frame_weight,grasp_goal_height_mm=args.grasp_goal_height_mm,hold_center_mm=args.hold_center_mm,approach_center_mm=args.approach_center_mm,temporal_sensory=args.temporal_sensory,refit_feature_adaptation=args.refit_feature_adaptation,feedback_circuit=args.feedback_circuit,transfer_readout=args.transfer_readout,recovery_target_policy=args.recovery_target_policy,sample_weight_policy=args.sample_weight_policy,policy_visited_weight=args.policy_visited_weight,validation_aggregation=args.validation_aggregation)
    elif args.command == "motor-run":
        from .visual_dopamine.motor_policy import MotorPolicy
        from .visual_dopamine.motor_run import run
        from .visual_dopamine.motor_experiment import task
        from .simulator import Simulator,EpisodeSimulator
        if args.episodes<1:command_parser.error("--episodes must be positive")
        policy=MotorPolicy(args.checkpoint,root,axis=None if args.axis is None else args.axis-1,selection=args.selection,step_degrees=args.step_degrees,step_mm=args.step_mm)
        tasks=[{}] if args.current_episode else (json.loads(args.tasks.read_text()) if args.tasks else [task(350,12 if i%2==0 else -12,9500+i) for i in range(args.episodes)])
        environment=Simulator(args.control_directory) if args.control_directory else EpisodeSimulator()
        from .visual_dopamine.remote_features import configure_runtime,close_runtime
        configure_runtime(policy,args.sensory_host)
        try:
            with environment as client:report=run(client,policy,tasks,args.output,args.current_episode,args.episode_id,args.max_steps,args.learn,args.settle_seconds,args.stop_after_failures)
        finally:close_runtime(policy)
    elif args.command == "visual-prepare":
        from .visual_dopamine.circuit import prepare
        directory, report = prepare(root,args.graph_id)
        report = {"directory":str(directory),**report}
    elif args.command == "visual-collect":
        from .visual_dopamine.experiment import collect_responses
        report = collect_responses(args.output,root,args.limit,args.base_prefixes)
    elif args.command == "visual-train":
        from .visual_dopamine.experiment import train_responses
        report = train_responses(args.dataset,args.output,root,args.trials,args.seeds,args.updates,args.local_contrast,args.learning_rate,args.exploration_sigma)
    elif args.command == "visual-run":
        from .visual_dopamine.policy import VisualDopaminePolicy
        from .visual_dopamine.run import run_responses
        from .simulator import Simulator, EpisodeSimulator
        if args.max_steps < 1 or args.max_steps > 12: command_parser.error("--max-steps must be 1–12")
        if args.episodes < 1: command_parser.error("--episodes must be positive")
        policy = VisualDopaminePolicy(args.checkpoint,root)
        tasks = [{}] if args.current_episode else (json.loads(args.tasks.read_text()) if args.tasks else
            [{"cube_size_mm":20,"cube_xy_mm":[350,12 if i%2==0 else -12],"initial_joints_deg":[0]*6,
              "initial_gripper_mm":0,"input_mode":"vision","hold_seconds":5,"timeout_seconds":60,"seed":8100+i} for i in range(args.episodes)])
        environment = Simulator(args.control_directory) if args.control_directory else EpisodeSimulator()
        with environment as client:
            report = run_responses(client,policy,tasks,args.output,args.current_episode,args.episode_id,args.max_steps,args.learn)
    elif args.command == "assets":
        from .assets import ensure_assets
        report = ensure_assets(root, args.download)
    elif args.command == "import":
        from .connectome import import_graph
        path, metadata = import_graph(root)
        report = {"directory": str(path), **metadata}
    elif args.command == "prepare-overlay":
        from .activity import prepare_layout
        directory, layout = prepare_layout(root, args.graph_id)
        report = {"directory": str(directory), **layout}
    elif args.command == "benchmark":
        from .connectome import Graph, resolve_graph
        from .neural_core import FlyController, benchmark
        from .adapters import feature_schema
        graph = Graph(resolve_graph(root)[0], verify=False)
        model = FlyController(graph, len(feature_schema("state")["names"]), updates=args.updates, channels=args.channels, backend=args.backend)
        report = benchmark(model, args.steps)
        report["graph_id"] = graph.manifest["graph_id"]
        if args.output:
            write_json(args.output, report)
    elif args.command in ("collect", "dagger"):
        from .collect import collect
        from .simulator import Simulator, EpisodeSimulator
        from .teacher import placement_tasks
        tasks = json.loads(args.tasks.read_text()) if args.tasks else placement_tasks(args.episodes, args.seed, args.sizes, args.xy_radius, args.yaw_range)
        policy = None
        if args.command == "dagger":
            metadata=json.loads((args.checkpoint/"manifest.json").read_text())
            if metadata.get("schema")=="seven-motor-checkpoint-v1":
                from .visual_dopamine.motor_policy import MotorPolicy
                policy=MotorPolicy(args.checkpoint,root)
            else:
                from .policy import Policy
                policy=Policy(args.checkpoint,root)
            if policy.mode=="vision" and args.no_images:raise ValueError("Camera-policy DAgger requires images")
        kwargs = {"binary": args.binary} if args.binary else {}
        args.dataset.mkdir(parents=True, exist_ok=True)
        environment = EpisodeSimulator(log=args.dataset / "native.log", **kwargs) if args.command == "dagger" and args.control_directory is None else Simulator(args.control_directory, log=args.dataset / "native.log", **kwargs)
        from .visual_dopamine.remote_features import configure_runtime,close_runtime
        if policy is not None:configure_runtime(policy,getattr(args,'sensory_host',None))
        try:
            with environment as client:
                report=collect(client,tasks,args.dataset,args.hz,not args.no_images,policy,
                               getattr(args,'teacher_probability',1.0),args.seed,perturbation_degrees=args.perturbation_degrees)
        finally:
            if policy is not None:close_runtime(policy)
    elif args.command == "split":
        from .dataset import split_dataset
        report = split_dataset(args.dataset, args.seed, args.include_dagger_failures)
    elif args.command == "train-bc":
        from .train_bc import train
        report = train(args, root)
    elif args.command in ("evaluate", "run"):
        from .evaluate import evaluate
        from .teacher import placement_tasks
        from .simulator import Simulator, EpisodeSimulator
        from .policy import Policy
        policy = Policy(args.checkpoint, root, ablation=args.ablation)
        tasks = [{}] if getattr(args, "current_episode", False) else (json.loads(args.tasks.read_text()) if args.tasks else placement_tasks(args.episodes, args.seed, args.sizes))
        tasks = [{**spec, "task": {**spec["task"], "input_mode": policy.mode}} if "task" in spec else {**spec, "input_mode": policy.mode} for spec in tasks]
        kwargs = {"binary": args.binary} if args.binary else {}
        args.output.mkdir(parents=True, exist_ok=True)
        environment = EpisodeSimulator(log=args.output / "native.log", **kwargs) if args.control_directory is None else Simulator(args.control_directory, log=args.output / "native.log", **kwargs)
        with environment as client:
            report = evaluate(client, policy, tasks, args.output, release=not args.no_release, record_images=args.record_images, floor_projection=args.floor_projection, current_episode=getattr(args, "current_episode", False), expected_episode_id=getattr(args, "episode_id", None))
    elif args.command == "export":
        from .checkpoint import export_checkpoint
        report = export_checkpoint(args.checkpoint, args.destination, root, args.include_graph)
    elif args.command == "rewire":
        from .ablations import rewire
        report = rewire(root, args.seed, args.swaps_per_edge)
    elif args.command == "tasks":
        from .dataset import export_tasks
        report = export_tasks(args.dataset, args.split)
        write_json(args.output, report)
    elif args.command == "train-rl":
        from .train_rl import train_rl
        from .teacher import placement_tasks
        from .simulator import Simulator, EpisodeSimulator
        kwargs = {"binary": args.binary} if args.binary else {}
        environment = EpisodeSimulator(**kwargs) if args.control_directory is None else Simulator(args.control_directory, **kwargs)
        with environment as client:
            report = train_rl(client, args, root, placement_tasks(args.episodes, args.seed))
    elif args.command == "audit-dataset":
        from .audit import audit_dataset
        report = audit_dataset(args.dataset, args.output)
    elif args.command == "export-core":
        from .checkpoint import export_core
        report = export_core(args.checkpoint, args.destination)
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.command in ("run", "evaluate", "visual-run", "motor-run") and any(episode.get("interrupted") for episode in report["episodes"]):
        raise SystemExit(130)


def main():
    try:
        return _main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
