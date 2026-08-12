import torch
from sleep_rswa import SleepStagingRSWASystem
from sleep_rswa.models import RSWADetectionNet
from sleep_rswa.training.engine import collect_rswa_predictions


def test_output_shapes():
    model=SleepStagingRSWASystem().eval(); b,t=1,4
    with torch.no_grad(): out=model(torch.randn(b,t,4,900),torch.randn(b,t,1,300),torch.ones(b,t,dtype=torch.bool))
    assert out["staging_logits"].shape==(b,t,5)
    assert out["stage_probs"].shape==(b,t,5)
    assert torch.allclose(out["stage_probs"].sum(dim=-1),torch.ones(b,t),atol=1e-5)
    assert out["tonic_logits"].shape==(b,t)
    assert out["phasic_logits"].shape==(b,t)
    assert out["any_logits"].shape==(b,t)


def test_rswa_head_accepts_optional_stage_probabilities():
    model=RSWADetectionNet().eval(); b,t=2,3; emg=torch.randn(b,t,1,300); stage_probs=torch.softmax(torch.randn(b,t,5),dim=-1)
    with torch.no_grad():
        out_without_stage=model(emg,torch.ones(b,t,dtype=torch.bool))
        out_with_stage=model(emg,torch.ones(b,t,dtype=torch.bool),stage_probs=stage_probs)
    for out in (out_without_stage,out_with_stage):
        assert out["tonic_logits"].shape==(b,t)
        assert out["phasic_logits"].shape==(b,t)
        assert out["any_logits"].shape==(b,t)


def test_collect_rswa_predictions_accepts_joint_system():
    model=SleepStagingRSWASystem().eval(); b,t=1,4
    batch={
        "signals": torch.randn(b,t,4,900),
        "emg_center": torch.randn(b,t,1,300),
        "padding_mask": torch.ones(b,t,dtype=torch.bool),
        "rswa_valid": torch.tensor([[True,True,False,True]]),
        "tonic_labels": torch.zeros(b,t),
        "phasic_labels": torch.zeros(b,t),
        "any_labels": torch.zeros(b,t),
        "subject_ids": ["synthetic"],
    }
    result=collect_rswa_predictions(model,[batch],torch.device("cpu"),amp=False,threshold=0.5)
    assert result["subject_id"].shape==(3,)
    assert result["mini_epoch_index"].tolist()==[0,1,3]
    assert result["tonic_probability"].shape==(3,)
    assert result["phasic_probability"].shape==(3,)
    assert result["any_probability"].shape==(3,)
