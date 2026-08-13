import torch
from sleep_rswa import SleepStagingRSWASystem
from sleep_rswa.config import ModelConfig
from sleep_rswa.models import MovementCNN, build_movement_model
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


def test_collect_rswa_predictions_supports_aasm_simple_postprocess():
    class _Fake(torch.nn.Module):
        def forward(self, signals, emg_center, mask=None):
            del signals, emg_center, mask
            tonic = torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0, -3.0, -3.0, -3.0, -3.0, -3.0]])
            phasic = torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0, -3.0, -3.0, -3.0, -3.0, -3.0]])
            any_logits = torch.full_like(tonic, -3.0)
            return {"tonic_logits": tonic, "phasic_logits": phasic, "any_logits": any_logits}

    batch={
        "signals": torch.randn(1,10,4,900),
        "emg_center": torch.randn(1,10,1,300),
        "padding_mask": torch.ones(1,10,dtype=torch.bool),
        "rswa_valid": torch.ones(1,10,dtype=torch.bool),
        "tonic_labels": torch.ones(1,10),
        "phasic_labels": torch.ones(1,10),
        "any_labels": torch.ones(1,10),
        "subject_ids": ["synthetic"],
    }
    result=collect_rswa_predictions(
        _Fake(), [batch], torch.device("cpu"), amp=False, threshold=0.5,
        postprocess_mode="aasm_simple",
    )
    assert result["tonic_prediction"].sum()==10
    assert result["phasic_prediction"].sum()==10
    assert result["any_prediction"].sum()==10


def test_rswa_standalone_defaults_to_no_stage_conditioning():
    model=RSWADetectionNet().eval()
    assert model.use_stage_conditioning is False
    assert model.stage_fusion is None


def test_joint_system_default_enables_stage_conditioning():
    model=SleepStagingRSWASystem().eval()
    assert model.use_stage_conditioning is True
    assert model.rswa_model.use_stage_conditioning is True


def test_build_movement_model_variants_match_multihead_contract():
    cfg=ModelConfig(rswa_stage_conditioning=True)
    for name in ("cnn","cnn_lstm","cnn_bilstm"):
        model=build_movement_model(name,config=cfg,stage_conditioning=True).eval()
        out=model(
            torch.randn(1,2,1,300),
            torch.ones(1,2,dtype=torch.bool),
            stage_probs=torch.softmax(torch.randn(1,2,5),dim=-1),
        )
        assert out["tonic_logits"].shape==(1,2)
        assert out["phasic_logits"].shape==(1,2)
        assert out["any_logits"].shape==(1,2)
        assert model.use_stage_conditioning is True


def test_rswa_model_accepts_baseline_relative_second_channel():
    cfg=ModelConfig(rswa_stage_conditioning=True,rswa_emg_in_channels=2,rswa_use_baseline_relative_channel=True)
    model=build_movement_model("cnn_bimamba",config=cfg,stage_conditioning=True).eval()
    out=model(
        torch.randn(1,2,2,300),
        torch.ones(1,2,dtype=torch.bool),
        stage_probs=torch.softmax(torch.randn(1,2,5),dim=-1),
    )
    assert out["tonic_logits"].shape==(1,2)
