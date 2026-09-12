"""Recorded-image visual-response curriculum and independently scored motor trials."""
import base64
from copy import deepcopy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from ..assets import write_json, sha256_file, canonical_hash, home, artifact_manifest, checked_files
from ..simulator import Simulator, configure_episode, release_for_cleanup
from .circuit import prepare, Circuit
from .core import VisualCore, DopamineLearner


def alignment_error(observation, evaluation):
    # Privileged state is confined to the evaluator, never the retinal controller.
    cube=evaluation["cube_pose"]["position_mm"]
    grasp=observation["grasp_pose"]["position_mm"]
    goal=math.atan2(cube[1],cube[0]);actual=math.atan2(grasp[1],grasp[0])
    return abs(math.atan2(math.sin(goal-actual),math.cos(goal-actual))) * 180/math.pi


def save_observation(directory,observation):
    # Only RGB and proprioceptive/control timing fields belong in the policy record.
    allowed=["images","joints_deg","gripper_mm","finger_contacts","simulation_time","frame_id","episode_id"]
    result=deepcopy({key:observation[key] for key in allowed})
    for camera in result["images"]:
        payload=base64.b64decode(camera.pop("jpeg_base64"),validate=True)
        name=camera["name"]+".jpg"
        (directory/name).write_bytes(payload)
        camera["file"]=name;camera["sha256"]=hashlib.sha256(payload).hexdigest()
    write_json(directory/"observation.json",result)
    return result


def response_action(observation,choice,step_deg=1.):
    if choice not in (0,1):raise ValueError("Response choice must be left or right")
    q=list(observation["joints_deg"])
    from ..adapters import JOINT_LOWER, JOINT_UPPER
    q[0]=float(np.clip(q[0]+(-step_deg if choice==0 else step_deg),JOINT_LOWER[0],JOINT_UPPER[0]))
    return {"joints_deg":q,"gripper_mm":observation["gripper_mm"]}


def collect_responses(output,root=None,limit=None,base_prefixes=(0.,)):
    output=Path(output)
    if limit is not None and limit < 1:raise ValueError("Response count must be positive")
    if not base_prefixes or any(not np.isfinite(v) or abs(v)>5 for v in base_prefixes):raise ValueError("Exploratory prefixes must be finite and within ±5 degrees")
    if output.exists():raise FileExistsError("Choose a new response dataset directory")
    output.mkdir(parents=True)
    cases=[]
    for x in (338.,346.,354.):
        for y in (-14.,-12.,-10.,-8.,8.,10.,12.,14.):
            for prefix in base_prefixes:
                if abs(math.degrees(math.atan2(y,x))-prefix)<.65: continue
                cases.append(({"cube_xy_mm":[x,y],"cube_size_mm":20.,"initial_joints_deg":[0.]*6,
                          "initial_gripper_mm":0.,"input_mode":"vision","hold_seconds":5.,"timeout_seconds":60.,
                          "cube_yaw_deg":0.,"seed":7000+len(cases)},float(prefix)))
    if limit is not None:cases=cases[:limit]
    records=[]
    with Simulator(log=output/"native.log") as client:
        for number,(task,prefix) in enumerate(cases):
            directory=output/f"case-{number:03d}";directory.mkdir()
            configure_episode(client,task)
            outcomes=[]
            for choice in (0,1):
                if choice:
                    client.release();client.call("rebot_reset_episode");client.wait_ready()
                folded=client.observe()
                assert folded["joints_deg"]==[0]*6 and folded["gripper_mm"]==0
                client.call("rebot_experiment_control",action="start")
                client.connect(provenance="conventional",model_id="visual-response-body-calibration")
                if prefix:
                    source=client.observe()
                    prefix_action={"joints_deg":[prefix,0,0,0,0,0],"gripper_mm":0}
                    client.action(source,prefix_action);time.sleep(.2)
                before=client.observe(images=choice==0)
                assert abs(before["joints_deg"][0]-prefix)<.01
                if choice==0:save_observation(directory,before)
                evaluation=client.call("rebot_get_evaluation")
                old_error=alignment_error(before,evaluation)
                start=client.observe()
                action=response_action(start,choice)
                client.action(start,action)
                time.sleep(.25)
                after=client.observe()
                final=client.call("rebot_get_evaluation")
                new_error=alignment_error(after,final)
                reward=float(new_error<old_error-.1)
                outcomes.append({"choice":choice,"action":action,"reward":reward,"before_error_deg":old_error,
                                 "after_error_deg":new_error,"observed_joints_deg":after["joints_deg"],
                                 "episode_id":start["episode_id"],"delay_s":after["simulation_time"]-start["simulation_time"]})
                client.release()
            if sum(v["reward"] for v in outcomes)!=1:raise RuntimeError("The response calibration did not produce one successful direction")
            record={"id":directory.name,"split":"test" if task["cube_xy_mm"][0]==346 else "train",
                    "task_for_evaluator_only":task,"exploratory_prefix_deg":prefix,"starts_folded":True,"outcomes":outcomes}
            write_json(directory/"outcomes.json",record);records.append(record)
            write_json(output/"collection.json",{"schema":"retinal-response-dataset-v1","task":"turn_toward_cube_once",
                       "task_scope":"One-degree base rotation reducing angular error; this is not a pickup task.",
                       "cases":records,"simulator":client.provenance,"images_recorded":True})
            print(f"Recorded visual response {number+1}/{len(cases)}",flush=True)
    return {"dataset":str(output),"cases":len(records),"images":2*len(records)}


def load_cases(dataset):
    dataset=Path(dataset)
    manifest=json.loads((dataset/"collection.json").read_text())
    cases=[]
    for record in manifest["cases"]:
        directory=dataset/record["id"]
        observation=json.loads((directory/"observation.json").read_text())
        for image in observation["images"]:
            if sha256_file(directory/image["file"])!=image["sha256"]:raise ValueError("Response image checksum mismatch")
        cases.append({**record,"observation":observation,"directory":directory})
    return cases


def evaluate_features(learner,features,cases):
    outcomes=[]
    for x,case in zip(features,cases):
        choice,score=learner.choose(x,explore=False)
        outcomes.append({"case":case["id"],"choice":choice,"score":score,"reward":case["outcomes"][choice]["reward"]})
    return {"successes":sum(x["reward"] for x in outcomes),"attempts":len(outcomes),"outcomes":outcomes}


def train_responses(dataset,output,root=None,trials=8000,seeds=(0,1,2),updates=12,local_contrast=False,learning_rate=.002,noise=.35):
    output=Path(output);root=home(root)
    if trials < 1 or not seeds:raise ValueError("Positive training trials and at least one seed are required")
    if output.exists():raise FileExistsError("Choose a new visual-dopamine output directory")
    output.mkdir(parents=True)
    path,meta=prepare(root);circuit=Circuit(path)
    cases=load_cases(dataset);train=[c for c in cases if c["split"]=="train"];test=[c for c in cases if c["split"]=="test"]
    if not train or not test:raise ValueError("Disjoint image-scene training and test sets are required")
    core=VisualCore(circuit,updates=updates)
    retinal=[core.encoder.sample(c["observation"]["images"],c["directory"]) for c in cases]
    train_retinal=[r for r,c in zip(retinal,cases) if c["split"]=="train"]
    x_train=core.fit_adaptation(train_retinal,local_contrast=local_contrast)
    x_test=np.asarray([core.encode(r) for r,c in zip(retinal,cases) if c["split"]=="test"])
    np.savez(output/"calibration.npz",retinal_mean=core.retinal_mean,retinal_std=core.retinal_std,kc_mean=core.kc_mean,kc_std=core.kc_std,
             train_features=x_train,test_features=x_test)
    results=[]
    for seed in seeds:
        for enabled in (True,False):
            learner=DopamineLearner(circuit,seed=seed,learning_rate=learning_rate,noise=noise)
            initial=evaluate_features(learner,x_test,test)
            before=learner.weight_digest();history=[];events=[];rng=np.random.default_rng(seed)
            for step in range(trials):
                i=int(rng.integers(len(train)))
                choice,_=learner.choose(x_train[i],explore=True)
                outcome=train[i]["outcomes"][choice]
                feedback=learner.reinforce(outcome["reward"],delay_s=outcome["delay_s"],dopamine_enabled=enabled)
                events.append({"trial":step+1,"case":train[i]["id"],"choice":choice,**feedback,"reward_delay_s":outcome["delay_s"]})
                if (step+1)%250==0:
                    training=evaluate_features(learner,x_train,train)
                    history.append({"trial":step+1,"training_successes":training["successes"],"training_attempts":len(train),
                                    "last_dopamine":feedback["dopamine_mean"],"last_weight_change_l1":feedback["weight_change_l1"]})
            final=evaluate_features(learner,x_test,test)
            blocked=evaluate_features(learner,np.zeros_like(x_test),test)
            row={"seed":seed,"dopamine_enabled":enabled,"initial":initial,"final":final,"visual_input_blocked":blocked,
                 "weights_before":before,"weights_after":learner.weight_digest(),"weight_change_l1":float(abs(learner.weights-learner.base).sum())}
            if not enabled and before!=learner.weight_digest():raise AssertionError("Weights changed without dopamine")
            run=output/(f"seed-{seed}" if enabled else f"seed-{seed}-no-dopamine");run.mkdir()
            np.savez(run/"model.npz",weights=learner.weights,retinal_mean=core.retinal_mean,retinal_std=core.retinal_std,kc_mean=core.kc_mean,kc_std=core.kc_std,
                     baseline=np.array(learner.baseline),motor_readout=learner.motor_readout)
            write_json(run/"training.json",history)
            with (run/"dopamine-updates.jsonl").open("w") as stream:
                for event in events: stream.write(json.dumps(event,separators=(",",":"),allow_nan=False)+"\n")
            artifact_manifest(run,{"schema":"visual-dopamine-checkpoint-v1","name":run.name,"graph_id":meta["graph_id"],
                "circuit_id":meta["circuit_id"],"architecture":{"kind":"malecns_visual_dopamine","updates":updates},
                "observation_schema":{"mode":"vision","retinal_inputs":"R1-R6 luminance proxy","privileged_cube_pose":False},
                "task_kind":"visual_base_response","task_scope":"One base turn toward the cube; pickup is not qualified.",
                "training_method":"PAM01-gated node-perturbation eligibility on recorded KCg-d→MBON edges",
                "dopamine_enabled":enabled,"seed":seed,"trials":trials,"assumptions":meta["recipe"]["assumptions"],
                "learning_rule":{"name":"reward-gated node perturbation","learning_rate":learner.learning_rate,
                    "retinal_adaptation":"local" if local_contrast else "uniform",
                    "exploration_sigma":learner.noise,"eligibility_tau_seconds":learner.eligibility_tau,
                    "dopamine_signal":"tanh(reward - exponential reward expectation)","baseline_rate":.02,
                    "synapse_bounds":"0 to 4 times original normalized KCg-d input weight"},
                "dataset":str(Path(dataset).resolve()),"test_result":final,"biologically_validated":False,
                "package_source_hash":canonical_hash({str(p.relative_to(Path(__file__).parents[1])):sha256_file(p) for p in Path(__file__).parents[1].rglob("*.py")})})
            results.append(row);print(f"Seed {seed}, dopamine={enabled}: {final['successes']}/{final['attempts']} held-out responses",flush=True)
            write_json(output/"comparison.json",{"schema":"visual-dopamine-comparison-v1","task":"turn_toward_cube_once","trials_per_run":trials,"results":results})
    return {"output":str(output),"results":results}
