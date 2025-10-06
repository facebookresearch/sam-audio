from abc import ABCMeta, abstractmethod
from typing import List

import torch


class Ranker(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, audio: list[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Args:
            audio: (list[torch.Tensor]) where each element in the list corresponds to
                the candidates for the i'th generation (num_candidates, num_frames)
        Returns:
            (torch.Tensor) of shape (batch_size, num_candidates) correspoding to the ranking scores
        """
        pass


class EnsembleRanker(Ranker):
    def __init__(self, rankers: List[Ranker], weights: List[float]):
        super().__init__()
        assert len(rankers) == len(weights)
        self.rankers = torch.nn.ModuleList(rankers)
        self.weights = weights

    def forward(self, **kwargs) -> torch.Tensor:
        result = None
        for weight, ranker in zip(self.weights, self.rankers):
            if result is None:
                result = weight * ranker(**kwargs)
            else:
                result += weight * ranker(**kwargs)
        return result
