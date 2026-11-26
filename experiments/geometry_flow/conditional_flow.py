"""Conditional RealNVP-style density using unmodified nflows components."""
import torch
from torch import nn

def build_flow(input_dim, condition_dim, cfg):
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.coupling import AffineCouplingTransform
    from nflows.transforms.permutations import RandomPermutation
    from nflows.nn.nets import ResidualNet
    from nflows.utils.torchutils import create_alternating_binary_mask
    if input_dim < 2 or condition_dim < 1 or cfg["layers"] < 1:
        raise ValueError("Invalid flow dimensions")
    transforms = []
    for i in range(cfg["layers"]):
        transforms.append(AffineCouplingTransform(
            mask=create_alternating_binary_mask(input_dim, even=(i%2==0)),
            transform_net_create_fn=lambda a,b: ResidualNet(
                in_features=a, out_features=b, hidden_features=cfg["hidden_dim"],
                context_features=condition_dim, num_blocks=cfg["residual_blocks"],
                dropout_probability=0.0, use_batch_norm=False)))
        transforms.append(RandomPermutation(features=input_dim))
    return Flow(CompositeTransform(transforms),StandardNormal([input_dim]))

class BimanualFlow(nn.Module):
    def __init__(self, t, feature_dim, state_dim, cfg):
        super().__init__()
        self.t,self.feature_dim,self.state_dim=t,feature_dim,state_dim
        self.flows=nn.ModuleDict({s:build_flow(t*feature_dim,t*state_dim,cfg)
                                  for s in ("left","right")})
    def scores(self,left,right,joint):
        if joint.ndim != 3 or joint.shape[1:] != (self.t,self.state_dim):
            raise ValueError("Unexpected joint window shape")
        context=joint.flatten(1)
        result={}
        for side,x in (("left",left),("right",right)):
            if x.shape != (len(joint),self.t,self.feature_dim):
                raise ValueError("Unexpected geometry window shape")
            result[side]=-self.flows[side].log_prob(x.flatten(1),context=context)
        return result
