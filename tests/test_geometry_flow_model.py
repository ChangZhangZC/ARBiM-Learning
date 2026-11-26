import pytest
import torch
pytest.importorskip("nflows")
from experiments.geometry_flow.conditional_flow import BimanualFlow

@pytest.mark.parametrize("t",[1,8,16])
def test_scores_and_inverse(t):
    model=BimanualFlow(t,4,6,dict(layers=2,hidden_dim=16,residual_blocks=1))
    x=torch.randn(3,t,4)
    q=torch.randn(3,t,6)
    scores=model.scores(x,x,q)
    assert scores["left"].shape==(3,)
    assert torch.isfinite(scores["left"]).all()
    scores["left"].mean().backward()
    assert all(p.grad is None for p in model.flows["right"].parameters())
    flow=model.flows["left"]
    z,ld=flow._transform(x.flatten(1),context=q.flatten(1))
    restored,ild=flow._transform.inverse(z,context=q.flatten(1))
    torch.testing.assert_close(restored,x.flatten(1),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(ld+ild,torch.zeros_like(ld),atol=1e-5,rtol=1e-5)
