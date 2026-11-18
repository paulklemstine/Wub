"""A failed training run must never partially replace the shipping bank."""
import json
from pathlib import Path
import pytest
from scripts import train_slice_pipeline as pipeline


def test_failed_staged_training_leaves_original_models_intact(tmp_path,monkeypatch):
    bank=tmp_path/'bank';bank.mkdir();(bank/'kits').mkdir()
    (bank/'kit_manifest.json').write_text('{"kits":[]}')
    names=['slice_flow.json','slice_flow_features.f32','motif_return.json','bar_tour_value.json','sidecars.json']
    expected={name:(json.dumps({'version':1}).encode() if name.endswith('.json') else b'original pcm') for name in names}
    for name,content in expected.items():(bank/name).write_bytes(content)
    def failing_run(script,*args):
        stage=Path(args[args.index('--kits')+1])
        (stage/'slice_flow.json').write_text('{"not":"published"}')
        assert (stage/'kits').resolve()==(bank/'kits').resolve()
        raise RuntimeError('held-out validation failed')
    monkeypatch.setattr(pipeline,'run',failing_run)
    work=tmp_path/'work'
    monkeypatch.setattr('sys.argv',['train_slice_pipeline','--kits',str(bank),'--work',str(work)])
    with pytest.raises(RuntimeError,match='held-out'):pipeline.main()
    assert all((bank/name).read_bytes()==content for name,content in expected.items())
    assert not list(work.glob('staged-*'))


def test_self_generated_preferences_cannot_promote_a_regression():
    baseline={'validation_mae':.012}
    assert not pipeline.auto_refinement_allowed(baseline,{'validation_mae':.0155})
    assert pipeline.auto_refinement_allowed(baseline,{'validation_mae':.0125})
    assert pipeline.auto_refinement_allowed(baseline,{'validation_mae':.011})
    assert not pipeline.auto_refinement_allowed(baseline,{'validation_mae':float('nan')})
