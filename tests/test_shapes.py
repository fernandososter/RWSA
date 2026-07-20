import torch
from sleep_rswa import SleepStagingRSWASystem
def test_output_shapes():
    model=SleepStagingRSWASystem().eval(); b,t=1,4
    with torch.no_grad(): out=model(torch.randn(b,t,4,900),torch.randn(b,t,1,300),torch.ones(b,t,dtype=torch.bool))
    assert out["staging_logits"].shape==(b,t,5)
    assert out["tonic_logits"].shape==(b,t)
    assert out["phasic_logits"].shape==(b,t)
