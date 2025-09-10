from abc import ABCMeta, abstractmethod

import torch
import torchvision
from core.vision_encoder import pe
from open_clip import create_model_and_transforms
from torch.nn.utils.rnn import pad_sequence

from sam_audio.model.config import (
    MetaCLIPConfig,
    PerceptionEncoderConfig,
    VisionEncoderConfig,
)


class RescaleTransform(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, interpolation):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        self.interpolation = interpolation

    def __call__(self, sample):
        # sample: [T, C, H, W]
        sample = torch.nn.functional.interpolate(
            sample.float(), size=self.output_size, mode=self.interpolation.value
        )
        return sample


class VisionEncoder(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.dim = cfg.dim
        self.transform = self.get_transform()

    @torch.no_grad()
    def forward(self, videos: list[torch.Tensor]) -> torch.Tensor:
        """
        Encodes a list of input videos.  Each element of the list is a video represented
            as a tensor [T, C, H, W]
        Args:
            videos (list[torch.Tensor]): List of input image tensors to be processed.

        Returns:
            torch.Tensor: Encoded feature representations of the input tensors.
                The output is padded along the time dimension for variable length videos
        """
        result = []
        for video in videos:
            video = self.transform(video)
            if self.batch_size > 0 and video.size(0) > self.batch_size:
                res = []
                for i in range(0, video.size(0), self.batch_size):
                    res.append(self.encode(video[i : i + self.batch_size]))
                result.append(torch.cat(res, dim=0))
            result.append(self.encode(video))
        return pad_sequence(result, batch_first=True, padding_value=0.0)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_transform(self):
        pass


class MetaCLIPEncoder(VisionEncoder):
    def __init__(self, cfg: MetaCLIPConfig):
        self.model, _, self.preprocess = create_model_and_transforms(cfg.name)
        self.model = self.model.eval()
        self.resize_type = cfg.resize_type
        self.normalize_features = cfg.normalize_features
        super().__init__(cfg)

    def get_transform(self):
        T = torchvision.transforms
        if self.resize_type == "aspect_invariant":
            resize_transform = T.Resize(224, interpolation=T.InterpolationMode.BICUBIC)
        elif self.resize_type == "aspect_variant":
            resize_transform = RescaleTransform(
                224, interpolation=T.InterpolationMode.BICUBIC
            )
        else:
            raise NotImplementedError

        data_transform = T.Compose(
            [
                resize_transform,
                T.CenterCrop(224),
                T.Lambda(lambda x: x.float() / 255.0),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        return data_transform

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(x)
        if self.normalize_features:
            feats /= feats.norm(dim=-1, keepdim=True)
        return feats


class PerceptionEncoder(VisionEncoder):
    def __init__(self, cfg: PerceptionEncoderConfig):
        self.normalize_feature = cfg.normalize_feature
        self.interpolation_mode = cfg.interpolation_mode
        self.image_size = cfg.image_size
        super().__init__(cfg)
        self.model = pe.CLIP.from_config(cfg.name)

    def encode(self, x):
        image_features = self.model.encode_image(x, normalize=self.normalize_feature)
        return image_features

    def get_transform(self):
        T = torchvision.transforms
        try:
            interp = getattr(T.InterpolationMode, self.interpolation_mode.upper())
        except AttributeError as err:
            raise ValueError(
                f"Unsupported interpolation_mode: {self.interpolation_mode}"
            ) from err
        crop = [
            T.Resize(
                (self.image_size, self.image_size),
                interpolation=interp,
            )
        ]

        return T.Compose(
            crop
            + [
                T.Lambda(lambda x: x.float() / 255.0),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
            ]
        )
