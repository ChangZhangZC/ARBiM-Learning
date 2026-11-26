import numpy as np
from arbim.envs.duobench import RCS_NAMES, to_dataset_observation
from experiments.geometry_flow.eval import monitor_observation

def test_monitor_preserves_continuous_gripper():
    keys=["observation.images."+c for c in ("head","left_wrist","right_wrist")]
    mapping=dict(zip(keys,("head","left_wrist","right_wrist")))
    spec=dict(state_key="observation.state",state_dim=16,action_dim=16,
              state_shape=(16,),action_shape=(16,),state_names=RCS_NAMES,
              action_names=RCS_NAMES,camera_keys=keys,camera_shapes={k:(3,4,4) for k in keys})
    raw={s:{"joints":np.zeros(7,dtype=np.float32),"gripper":np.array([v],dtype=np.float32)}
         for s,v in (("left",.45),("right",.95))}
    raw["frames"]={c:{"rgb":{"data":np.zeros((4,4,3),dtype=np.uint8)}} for c in mapping.values()}
    monitor=monitor_observation(raw,spec,mapping)
    policy=to_dataset_observation(raw,spec,mapping,binarize_state_gripper=True)
    np.testing.assert_allclose(monitor["observation.state"][[7,15]],[.45,.95])
    np.testing.assert_allclose(policy["observation.state"][[7,15]],[0,1])
    for k in keys:
        np.testing.assert_array_equal(monitor[k],policy[k])
