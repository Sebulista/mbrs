from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

try:
    from . import DecoderBase, DecoderReferenceBased, register
except (ImportError):
    from mbrs.decoders import DecoderBase, DecoderReferenceBased, register
from mbrs.selectors import Selector, SelectorNbest


@register("ensemble_mbr")
class DecoderEnsembleMBR(DecoderBase): # Why is it reference based?

    def __init__(
        self,
        cfg: DecoderBase.Config,
        metrics: list[MetricBase],
        selector: Selector = SelectorNbest(SelectorNbest.Config()),
    ) -> None:
        self.cfg = cfg
        self.metrics = metrics
        self.selector = selector

    def maximize(self, metric) -> bool:
        """Return `True` when maximizing the objective score."""
        return metric.HIGHER_IS_BETTER

    # I don't really see where the following three functions are used, however,
    # we set maximize invariably to False, as the selector is only dealing with rankings.
    def topk(self, x: Tensor, k: int = 1) -> tuple[list[float], list[int]]:
        """Return the top-k best elements and corresponding indices.

        Args:
            x (Tensor): Input 1-D array.
            k (int): Return the top-k values and indices.

        Returns:
            tuple[list[float], list[int]]
              - list[float]: The top-k values.
              - list[int]: The top-k indices.
        """
        return self.selector.topk(x, k=k, maximize=False)

    def argbest(self, x: Tensor) -> Tensor:
        """Return the index of the best element.

        Args:
            x (Tensor): Input 1-D array.

        Returns:
            Tensor: A scalar tensor of the best index.
        """
        return self.selector.argbest(x, maximize=False)

    def superior(self, a: float, b: float) -> bool:
        """Return whether the score `a` is superior to the score `b`.

        Args:
            a (float): A score.
            b (float): A score.

        Returns:
            bool: Return True when `a` is superior to `b`.
        """
        return self.selector.superior(a, b, maximize=False)
    
    def select(
        self,
        hypotheses: list[str],
        expected_scores: Tensor,
        nbest: int = 1,
        source: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Selector.Output:
        """Select the final output list.

        Args:
            hypotheses (list[str]): Hypotheses.
            expected_scores (Tensor): The expected scores for each hypothesis.
            nbest (int): Return the n-best hypotheses based on the selection rule.
            source (str, optional): A source.
            maximize (bool): Whether maximize the scores or not.

        Returns:
            Selector.Output: Outputs.
        """
        return self.selector.select(
            hypotheses,
            expected_scores,
            nbest=nbest,
            source=source,
            maximize=False, # We deal with rankings, lower is always better, this minimize
            *args,
            **kwargs,
        )

    
    
    def decode(
        self,
        hypotheses: list[str],
        references: list[str],
        source: Optional[str] = None,
        nbest: int = 1,
        reference_lprobs: Optional[Tensor] = None,
    ) -> DecoderMBR.Output:
        """Select the n-best hypotheses based on the strategy.

        Args:
            hypotheses (list[str]): Hypotheses.
            references (list[str]): References.
            source (str, optional): A source.
            nbest (int): Return the n-best hypotheses.
            reference_lprobs (Tensor, optional): Log-probabilities for each reference sample.
              The shape must be `(len(references),)`. See `https://arxiv.org/abs/2311.05263`.

        Returns:
            DecoderMBR.Output: The n-best hypotheses.
        """

        ranks = []
        
        for metric in self.metrics:
            expected_scores = metric.expected_scores(
                hypotheses, references, source, reference_lprobs=reference_lprobs
            ).to("cpu") # -> Expected values for each row of shape `(H,)`

            # Sort in descending order of metric score
            sorted_indices = expected_scores.argsort(descending = self.maximize(metric))
            current_ranks = torch.empty_like(sorted_indices)
            current_ranks[sorted_indices] = torch.arange(len(expected_scores))
            # current_ranks contains the rank of each element in expected_scores
            ranks.append(current_ranks)

        ranks = torch.vstack(ranks)
        # Need ranks to be float
        ranks = ranks.float()
        avg_ranks = ranks.mean(dim=0) # -> tensor of shape '(H,)' where each element is the average rank of the sent at index i
            
            
        selector_outputs = self.select(
            hypotheses, avg_ranks, nbest=nbest, source=source
        )
        return (
            self.Output(
                idx=selector_outputs.idx,
                sentence=selector_outputs.sentence,
                score=selector_outputs.score,
            )
            | selector_outputs
        )