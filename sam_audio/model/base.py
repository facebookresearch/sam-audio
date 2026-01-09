# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

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
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        low_cpu_mem_usage: bool = False,
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
                proxies=proxies,
                resume_download=resume_download,
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

        if low_cpu_mem_usage:
            model_state = model.state_dict()
            for k, v in list(state_dict.items()):
                if k in model_state:
                    try:
                        model_state[k].copy_(v)
                    except Exception:
                        # fallback to regular assignment if in-place copy fails
                        model_state[k] = v
                # free the tensor asap to keep peak memory low
                del state_dict[k]
            # load remaining keys (if any) normally
            if len(state_dict) > 0:
                model.load_state_dict(state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)
        return model
