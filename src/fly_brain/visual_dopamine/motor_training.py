"""PAM-gated learning of seven directional responses from recorded RGB and rewards."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from ..assets import home,write_json,artifact_manifest,sha256_file,canonical_hash
from .core import VisualCore
from .motor_core import MotorLearner
from .motor_circuit import MotorCircuit
from .experiment import load_cases


def reward(case,axis,choice):
    # A no-command response is assigned the neutral progress reward by definition.
    return .5 if choice==2 else float(case['outcomes'][axis][choice]['reward'])


def evaluate_features(learner,features,cases):
    rows=[];correct=np.zeros(7,dtype=int);count=np.zeros(7,dtype=int)
    active_correct=np.zeros(7,dtype=int);active_count=np.zeros(7,dtype=int)
    for x,case in zip(features,cases):
        choices,scores=learner.choose(x)
        for axis,choice in enumerate(choices):
            rewards=np.array([reward(case,axis,c) for c in range(3)])
            achieved=reward(case,axis,int(choice));best=float(rewards.max());ok=achieved>=best-1e-8
            correct[axis]+=ok;count[axis]+=1
            if best>.5:active_correct[axis]+=ok;active_count[axis]+=1
            rows.append({'case':case['id'],'axis':axis,'choice':int(choice),'score':float(scores[axis]),'reward':achieved,'best_available_reward':best,'correct':bool(ok)})
    return {'correct':int(correct.sum()),'attempts':int(count.sum()),'per_axis_correct':correct.tolist(),'per_axis_attempts':count.tolist(),
            'active_correct':int(active_correct.sum()),'active_attempts':int(active_count.sum()),
            'per_axis_active_correct':active_correct.tolist(),'per_axis_active_attempts':active_count.tolist(),'responses':rows}


def save_checkpoint(directory,core,learner,metadata):
    metadata={k:v for k,v in metadata.items() if k not in ['name','files','learning_rule','rng_state','graph_id','circuit_id','motor_circuit_id','feedback_circuit_id','package_source_hash']}
    if getattr(core,'temporal_config',None):metadata['temporal_sensory']=dict(core.temporal_config)
    elif metadata.get('temporal_sensory'):raise ValueError('Temporal checkpoint requires a temporal sensory core')
    schema={"mode":"vision","privileged_cube_pose":False,"retinal_inputs":"R1-R6 luminance",**metadata.get("observation_schema",{})}
    schema["camera_names"]=list(core.encoder.camera_names)
    schema["camera_rig_revision"]=core.encoder.camera_rig_revision
    if core.encoder.profile is not None:schema["retinal_profile"]=core.encoder.profile
    else:schema.pop("retinal_profile",None)
    metadata["observation_schema"]=schema
    directory=Path(directory)
    if directory.exists():raise FileExistsError('Motor checkpoints never overwrite another model')
    directory.mkdir(parents=True)
    motor_values={'motor_synapses':core.circuit.motor_from_dn_pm.data} if metadata.get('motor_synapse_plasticity') else {}
    np.savez(directory/'model.npz',**motor_values,weights=learner.weights,retinal_mean=core.retinal_mean,retinal_std=core.retinal_std,
             kc_mean=core.kc_mean,kc_std=core.kc_std,baseline=np.array(learner.baseline),excitability=learner.excitability,readout=core.circuit.readout,gain=core.circuit.gain)
    identity=core.circuit.motor_manifest['motor_circuit_id']
    feedback={'feedback_circuit_id':core.circuit.feedback_manifest['feedback_circuit_id']} if hasattr(core.circuit,'feedback_manifest') else {}
    rule={'learning_rate':learner.learning_rate,'weight_multiple':learner.weight_multiple,'exploration_sigma':learner.noise,'eligibility_tau_seconds':learner.eligibility_tau,'deadband':learner.deadband,'kc_tonic':learner.kc_tonic,'kc_gain':learner.kc_gain,'balanced_tonic':learner.balanced_tonic,'response_mode':learner.response_mode,'motor_reference':'zero rate-deviation state' if learner.response_mode=='centered' else 'untrained synapses with tonic KC input','exploration':'motor-pool Gaussian babbling with chain-rule synaptic eligibility through fixed neural routes'}
    return artifact_manifest(directory,{'schema':'seven-motor-checkpoint-v1','name':directory.parent.name+'/'+directory.name,
        'graph_id':core.circuit.manifest['graph_id'],'circuit_id':core.circuit.manifest['circuit_id'],'motor_circuit_id':identity,
        'architecture':{'kind':'malecns_motor_dopamine','updates':core.updates,'output_channels':7},
        'observation_schema':{'mode':'vision','privileged_cube_pose':False,'retinal_inputs':'R1-R6 luminance'},
        'learning_rule':rule,'rng_state':deepcopy(learner.rng.bit_generator.state),
        'package_source_hash':canonical_hash({str(p.relative_to(Path(__file__).parents[1])):sha256_file(p) for p in Path(__file__).parents[1].rglob('*.py')}),
        'task_kind':'seven_motor_response','task_scope':'Learned single-joint progress responses; complete pickup requires separate live verification',
        'training_method':'PAM01-gated policy-gradient eligibility on existing visual KC to MBON edges; fixed annotated muscle-pool decoder',
        'modelling_assumptions':['Abstract rate model; neuron time constants and transmitter receptor signs are not fully reconstructed','KC tonic rate 0.2 plus 0.4 times normalized visual response, clipped to [0,1]','Fixed motor rest reference from untrained synapses','Nonlocal chain-rule synaptic credit is an engineering approximation, not demonstrated fly biochemistry','Motor-neuron babbling supplies exploration; only existing KC-MBON weights change'],
        'biologically_validated':False,**metadata,**feedback})


def train(dataset,output,root=None,trials=40000,seeds=(0,1,2),learning_rate=.0002,noise=.15,local_contrast=True):
    root=home(root);dataset=Path(dataset);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new motor training output')
    if trials<1 or not seeds:raise ValueError('Training trials and seeds are required')
    collection=json.loads((dataset/'collection.json').read_text());circuit=MotorCircuit(root/'motor-circuits'/collection['motor_circuit_id'],root)
    cases=load_cases(dataset);train_cases=[c for c in cases if c['split']=='train'];test_cases=[c for c in cases if c['split']=='test']
    if not train_cases or not test_cases:raise ValueError('Separate training and test scenes are required')
    output.mkdir(parents=True)
    core=VisualCore(circuit,updates=12)
    retinal=[core.encoder.sample(c['observation']['images'],c['directory']) for c in cases]
    x_train=core.fit_adaptation([x for x,c in zip(retinal,cases) if c['split']=='train'],local_contrast=local_contrast)
    x_test=np.asarray([core.encode(x) for x,c in zip(retinal,cases) if c['split']=='test'])
    np.savez(output/'calibration.npz',retinal_mean=core.retinal_mean,retinal_std=core.retinal_std,kc_mean=core.kc_mean,kc_std=core.kc_std,train_features=x_train,test_features=x_test)
    results=[]
    for seed in seeds:
        for dopamine_enabled in [True,False]:
            learner=MotorLearner(circuit,seed=seed,learning_rate=learning_rate,noise=noise)
            initial=evaluate_features(learner,x_test,test_cases);before=learner.weight_digest();rng=np.random.default_rng(seed)
            history=[];events=[];best_score=-1;selected_trial=0;snapshot=None
            for step in range(trials):
                case_index=int(rng.integers(len(train_cases)));axis=int(rng.integers(7))
                choices,_=learner.choose(x_train[case_index],explore=True,exploration_axis=axis);choice=int(choices[axis])
                case=train_cases[case_index];outcome_reward=reward(case,axis,choice)
                delay=float(case['outcomes'][axis][choice]['delay_s']) if choice<2 else .2
                feedback=learner.reinforce(outcome_reward,delay_s=delay,dopamine_enabled=dopamine_enabled)
                events.append({'trial':step+1,'case':case['id'],'axis':axis,'choice':choice,**feedback})
                if (step+1)%1000==0 or step+1==trials:
                    score=evaluate_features(learner,x_train,train_cases)
                    history.append({'trial':step+1,'correct':score['correct'],'attempts':score['attempts'],'active_correct':score['active_correct'],'active_attempts':score['active_attempts']})
                    if score['correct']>best_score:
                        best_score=score['correct'];selected_trial=step+1;snapshot=(learner.weights.copy(),learner.baseline,deepcopy(learner.rng.bit_generator.state))
            learner.weights[:],learner.baseline,learner.rng.bit_generator.state=snapshot
            final=evaluate_features(learner,x_test,test_cases);training=evaluate_features(learner,x_train,train_cases)
            blocked=evaluate_features(learner,np.zeros_like(x_test),test_cases)
            if not dopamine_enabled and before!=learner.weight_digest():raise AssertionError('Weights changed without dopamine')
            directory=output/(f'seed-{seed}' if dopamine_enabled else f'seed-{seed}-no-dopamine')
            save_checkpoint(directory,core,learner,{'seed':seed,'dopamine_enabled':dopamine_enabled,'trials':trials,'selected_trial':selected_trial,
                'selection':'highest training-scene response accuracy; test scenes never select weights','dataset':str(dataset.resolve()),'local_contrast':local_contrast,'test_result':final})
            write_json(directory/'training.json',history)
            with (directory/'dopamine-updates.jsonl').open('w') as stream:
                for event in events:stream.write(json.dumps(event,separators=(',',':'),allow_nan=False)+'\n')
            # Rehash after adding the complete training audit.
            meta=json.loads((directory/'manifest.json').read_text());meta.pop('files',None);artifact_manifest(directory,meta)
            row={'seed':seed,'dopamine_enabled':dopamine_enabled,'initial':initial,'training':training,'final':final,'visual_input_blocked':blocked,
                 'selected_trial':selected_trial,'weights_before':before,'weights_after':learner.weight_digest(),'weight_change_l1':float(np.abs(learner.weights-learner.base).sum())}
            results.append(row);write_json(output/'comparison.json',{'schema':'seven-motor-comparison-v1','results':results})
            print(f"Seed {seed}, dopamine={dopamine_enabled}: train {training['correct']}/{training['attempts']}, test {final['correct']}/{final['attempts']}; active {final['active_correct']}/{final['active_attempts']}",flush=True)
    return {'output':str(output),'results':results}
