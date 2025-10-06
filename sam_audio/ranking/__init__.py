from sam_audio.model.config import (
    ClapRankerConfig,
    EnsembleRankerConfig,
    ImageBindRankerConfig,
    JudgeRankerConfig,
)
from sam_audio.ranking.clap import ClapRanker
from sam_audio.ranking.imagebind import ImageBindRanker
from sam_audio.ranking.judge import JudgeRanker
from sam_audio.ranking.ranker import EnsembleRanker


def create_ranker(config):
    if isinstance(config, ImageBindRankerConfig):
        return ImageBindRanker(config)
    elif isinstance(config, ClapRankerConfig):
        return ClapRanker(config)
    elif isinstance(config, JudgeRankerConfig):
        return JudgeRanker(config)
    elif isinstance(config, EnsembleRankerConfig):
        return EnsembleRanker(
            rankers=[create_ranker(cfg) for cfg in config.rankers],
            weights=config.weights,
        )
    else:
        assert config is None
        return None
