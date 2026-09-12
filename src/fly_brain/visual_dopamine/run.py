"""Live retinal-response evaluation; only the evaluator sees cube pose."""
import time
from pathlib import Path
from ..assets import write_json
from ..simulator import configure_episode, current_episode_task, release_for_cleanup
from ..activity import ActivityPublisher
from .experiment import alignment_error, save_observation


def run_responses(client,policy,tasks,output,current_episode=False,expected_episode_id=None,max_steps=1,learn=False):
    if not tasks or not 1 <= max_steps <= 12: raise ValueError("At least one task and 1–12 steps are required")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results=[]
    for number,task in enumerate(tasks):
        activity=None;row={"success":False,"steps":[]}
        directory=output/f"response-{number:03d}";directory.mkdir()
        try:
            actual=current_episode_task(client,"vision",expected_episode_id) if current_episode else configure_episode(client,{**task,"input_mode":"vision"})
            row["task_for_evaluator_only"]=actual
            observation=client.observe(images=True);policy.reset(observation)
            # Warm the computation before acquiring the controller lease.
            policy.act(observation);policy.reset(observation)
            activity=ActivityPublisher.for_policy(policy,client)
            initial=client.call("rebot_get_evaluation");initial_error=alignment_error(observation,initial)
            row["initial_error_deg"]=initial_error
            if initial_error<.65:
                row.update(success=max_steps>1,already_aligned=True,error="Already facing the cube; place it at Y=+/-8 to 15 mm to test a response")
            else:
                client.call("rebot_experiment_control",action="start")
                client.connect(provenance="malecns",model_id=policy.metadata["name"])
                for step in range(max_steps):
                    began=time.monotonic();observation=client.observe(images=True)
                    before=client.call("rebot_get_evaluation");old_error=alignment_error(observation,before)
                    action=policy.act(observation,explore=learn)
                    if activity:activity.publish(policy.state,observation)
                    frame_dir=directory/f"step-{step:03d}";frame_dir.mkdir()
                    save_observation(frame_dir,observation)
                    client.action(observation,action)
                    time.sleep(.25)
                    after=client.observe();evaluation=client.call("rebot_get_evaluation")
                    error=alignment_error(after,evaluation);reward=float(error<old_error-.1)
                    feedback=policy.feedback(reward,delay_s=after["simulation_time"]-observation["simulation_time"],learn=learn)
                    if activity:activity.publish(policy.state,after)
                    decision={"choice":policy.last_choice,"motor_score":policy.last_score,"action":action,
                              "before_error_deg":old_error,"after_error_deg":error,"feedback":feedback,
                              "actual_joints_deg":after["joints_deg"],"loop_ms":1000*(time.monotonic()-began)}
                    row["steps"].append(decision);write_json(frame_dir/"decision.json",decision)
                    row["success"]=bool(reward) if max_steps==1 else error<.65
                    row["final_error_deg"]=error
                    if row["success"] or not reward:break
                    time.sleep(max(0,1/policy.hz-(time.monotonic()-began)))
                # Keep a short visible reward pulse; heartbeat is maintained by observations/actions.
                for _ in range(3):
                    observation=client.observe();client.action(observation,{"joints_deg":observation["joints_deg"],"gripper_mm":observation["gripper_mm"]})
                    if activity:activity.publish(policy.state,observation)
                    time.sleep(.2)
        except KeyboardInterrupt:
            row.update(error="Interrupted",interrupted=True)
        except (OSError,ValueError,RuntimeError,TimeoutError) as error:
            row["error"]=str(error)
        finally:
            cleanup=release_for_cleanup(client)
            if activity:activity.close()
            if cleanup:row["cleanup_error"]=cleanup
            results.append(row)
            report={"task_kind":"visual_base_response" if max_steps==1 else "visual_base_alignment",
                    "task_scope":"Base turns driven by camera input; no pickup or hold is claimed.",
                    "checkpoint":policy.metadata["name"],"attempts":len(results),"successes":sum(r["success"] for r in results),
                    "learning_enabled":learn,"episodes":results}
            write_json(output/"evaluation.json",report)
        print(f"Visual response {number+1}: {'success' if row['success'] else row.get('error','failed')}",flush=True)
        if row.get("interrupted"):break
    if learn:
        report["learned_checkpoint"]=policy.save(output/"checkpoint",{"attempts":report["attempts"],"successes":report["successes"]})
        write_json(output/"evaluation.json",report)
    return report
