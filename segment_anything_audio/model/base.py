from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseModel(torch.nn.Module, metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def get_configs(cls) -> Dict[str, Any]: ...

    def device(self):
        return next(self.parameters()).device

    def load_ckpt(self, checkpoint_path: str):
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(checkpoint_path), strict=False
        )
        # Filter out text encoder keys, they aren't in the checkpoint and instead are loaded from HF
        missing_keys = [k for k in missing_keys if not k.startswith("text_encoder")]
        assert len(missing_keys) == 0, f"Missing key(s): {missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected key(s): {unexpected_keys}"

    @classmethod
    def from_config(
        cls, name: str, pretrained: bool = False, checkpoint_path: Optional[str] = None
    ):
        configs = cls.get_configs()
        assert name in configs, (
            f"Unknown model name {name}.  Available configs: {configs.keys()}"
        )
        model = cls(configs[name])
        if pretrained:
            assert checkpoint_path is not None, (
                "No checkpoint downloading support yet..."
            )
            model.load_ckpt(checkpoint_path=checkpoint_path)
        return model
