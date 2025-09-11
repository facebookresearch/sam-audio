import json
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
from huggingface_hub import ModelHubMixin, snapshot_download

from sam_audio.model.config import deserialize_config


class BaseModel(torch.nn.Module, ModelHubMixin, metaclass=ABCMeta):
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
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",  # additional argument
        strict: bool = False,  # additional argument
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            cached_model_dir = model_id
        else:
            cached_model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        with open(os.path.join(cached_model_dir, "params.json")) as fin:
            config = json.load(fin)

        config = deserialize_config(config)
        model = cls(config)
        state_dict = torch.load(
            os.path.join(cached_model_dir, "checkpoint.pt"),
            weights_only=True,
            map_location=map_location,
        )
        model.load_state_dict(state_dict, strict=strict)
        return model

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
