import pytest

from mbrs.metrics import get_metric, MetricChrF, MetricBLEU, MetricTER
from mbrs.selectors import Selector

from torch import Tensor
import torch

try:
    from .ensemble_mbr import DecoderEnsembleMBR
except (ImportError):
    from  mbrs.decoders import DecoderEnsembleMBR

HYPOTHESES = [
    ["another test", "this is a test", "this is an test", "this is a fest"],
]
REFERENCES = [
    ["another test", "this is a test", "this is an test", "this is a fest"],
]

BEST_INDICES = [1]
BEST_SENTENCES = [
    "this is a test",
]

RANKINGS = [
    {"bleu": [3,0,2,1]},
    {"chrf": [3,0,1,2]},
    {"ter": [2,0,1,3]},
]

SCORES = Tensor([8/3, 0, 4/3, 6/3]).sort().values


if __name__ == "__main__":
    bleu = MetricBLEU(MetricBLEU.Config())
    chrf = MetricChrF(MetricChrF.Config(word_order=2))
    ter =  MetricTER(MetricTER.Config())

    decoder = DecoderEnsembleMBR(DecoderEnsembleMBR.Config(), [bleu, chrf, ter])
    output = decoder.decode(HYPOTHESES[0], REFERENCES[0], nbest=4)
    print(output)
    torch.testing.assert_close(
        torch.tensor(output.score),
        SCORES,
        atol=0.0005,
        rtol=1e-4,
    )
    

"""class TestDecoderMBR:
    @pytest.mark.parametrize("metric_type", ["bleu", "ter"])
    @pytest.mark.parametrize("nbest", [1, 2])
    def test_decode(self, metric_type: str, nbest: int):
        metric_cls = get_metric(metric_type)
        decoder = DecoderMBR(DecoderMBR.Config(), metric_cls(metric_cls.Config()))
        for i, (hyps, refs) in enumerate(zip(HYPOTHESES, REFERENCES)):
            output = decoder.decode(hyps, refs, nbest=nbest, reference_lprobs=None)
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            assert len(output.sentence) == min(nbest, len(hyps))
            assert len(output.score) == min(nbest, len(hyps))
            torch.testing.assert_close(
                torch.tensor(output.score[0]),
                SCORES[metric_type][i],
                atol=0.0005,
                rtol=1e-4,
            )

            output = decoder.decode(
                hyps,
                refs,
                nbest=1,
                reference_lprobs=torch.Tensor([-2.000]).repeat(len(refs)),
            )
            assert output.idx[0] == BEST_INDICES[i]
            assert output.sentence[0] == BEST_SENTENCES[i]
            torch.testing.assert_close(
                torch.tensor(output.score[0]),
                SCORES[metric_type][i],
                atol=0.0005,
                rtol=1e-4,
            )"""