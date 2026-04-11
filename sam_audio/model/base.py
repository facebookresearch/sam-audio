# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import json
import os
from typing import Callable, Dict, Optional, Union

import torch
from huggingface_hub import ModelHubMixin, snapshot_download


class BaseModel(torch.nn.Module, ModelHubMixin):
    config_cls: Callable

    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = True,
        revision: Optional[str] = None,
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            cached_model_dir = model_id
        else:
            cached_model_dir = snapshot_download(
                repo_id=model_id,
                revision=cls.revision,
                cache_dir=cache_dir,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
            )

        with open(os.path.join(cached_model_dir, "config.json")) as fin:
            config = json.load(fin)

        for key, value in model_kwargs.items():
            if key in config:
                config[key] = value

        config = cls.config_cls(**config)
        model = cls(config)
        state_dict = torch.load(
            os.path.join(cached_model_dir, "checkpoint.pt"),
            weights_only=True,
            map_location=map_location,
        )
        model.load_state_dict(state_dict, strict=strict)
        return model