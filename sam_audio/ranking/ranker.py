from abc import ABCMeta, abstractmethod
from typing import List

import torch


class Ranker(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            audio: (num_candidates, channels, num_frames)
        """
        pass

    def top_idx(self, audio: torch.Tensor, **kwargs) -> int:
        return self(audio, **kwargs).argmax()


class DualRanker(Ranker):
    def __init__(self, rankers: List[Ranker], weights: List[float]):
        super().__init__()
        self.rankers = torch.nn.ModuleList(rankers)
        self.weights = weights

    def forward(self, audio: torch.Tensor, **kwargs) -> int:
        pass
