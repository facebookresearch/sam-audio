<div align="center">

# SAM-Audio

[![arXiv](https://img.shields.io/badge/arXiv-2512.18099-b31b1b.svg)](https://arxiv.org/abs/2512.18099)
![CI](https://github.com/facebookresearch/sam-audio/actions/workflows/ci.yaml/badge.svg)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/facebook/sam-audio)

![model_image](assets/sam_audio_main_model.png)

</div>

Segment Anything Model for Audio [[**Blog**](https://ai.meta.com/blog/sam-audio/)] [[**Paper**](https://ai.meta.com/research/publications/sam-audio-segment-anything-in-audio/)] [[**Demo**](https://aidemos.meta.com/segment-anything/editor/segment-audio)]

SAM-Audio is a foundation model for isolating any sound in audio using text, visual, or temporal prompts. It can separate specific sounds from complex audio mixtures based on natural language descriptions, visual cues from video, or time spans.

SAM-Audio and the Judge model crucially rely on [Perception-Encoder Audio-Visual (PE-AV)](https://huggingface.co/facebook/pe-av-large), which you can read more about [here](https://ai.meta.com/research/publications/pushing-the-frontier-of-audiovisual-perception-with-large-scale-multimodal-correspondence-learning/)

## Setup

**Requirements:**
- Python >= 3.10
- CUDA-compatible GPU (recommended)

Install dependencies:

```bash
pip install .
```

## Usage

⚠️ Before using SAM Audio, please request access to the checkpoints on the SAM Audio
Hugging Face [repo](https://huggingface.co/facebook/sam-audio-large). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

### Basic Text Prompting

```python
from sam_audio import SAMAudio, SAMAudioProcessor
import torchaudio
import torch

model = SAMAudio.from_pretrained("facebook/sam-audio-large")
processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
model = model.eval().cuda()

file = "<audio file>" # audio file path or torch tensor
description = "<description>"

batch = processor(
    audios=[file],
    descriptions=[description],
).to("cuda")

with torch.inference_mode():
    # NOTE: `predict_spans` and `reranking_candidates` have a large impact on performance.
    # Setting `predict_span=True` and `reranking_candidates=8` will give you better results at the cost of
    # latency and memory. See the "Span Prediction" section below for more details
   result = model.separate(batch, predict_spans=False, reranking_candidates=1)

# Save separated audio
sample_rate = processor.audio_sampling_rate
torchaudio.save("target.wav", result.target.cpu(), sample_rate)      # The isolated sound
torchaudio.save("residual.wav", result.residual.cpu(), sample_rate)  # Everything else
```

### Prompting Methods

SAM-Audio supports three types of prompts:

1. **Text Prompting**: Describe the sound you want to isolate using natural language
   ```python
   processor(audios=[audio], descriptions=["A man speaking"])
   ```

2. **Visual Prompting**: Use video frames and masks to isolate sounds associated with visual objects
   ```python
   processor(audios=[video], descriptions=[""], masked_videos=processor.mask_videos([frames], [mask]))
   ```

3. **Span Prompting**: Specify time ranges where the target sound occurs
   ```python
   processor(audios=[audio], descriptions=["A horn honking"], anchors=[[["+", 6.3, 7.0]]])
   ```

See the [examples](examples) directory for more detailed examples

### Span Prediction (Optional for Text Prompting)

We also provide support for automatically predicting the spans based on the text description, which is especially helpful for separating non-ambience sound events.  You can enable this by adding `predict_spans=True` in your call to `separate`

```python
with torch.inference_mode()
   outputs = model.separate(batch, predict_spans=True)

# To further improve performance (at the expense of latency), you can add candidate re-ranking
with torch.inference_mode():
   outputs = model.separate(batch, predict_spans=True, reranking_candidates=8)
```

### Re-Ranking

We provide the following models to assess the quality of the separated audio:

- [CLAP](https://github.com/LAION-AI/CLAP): measures the similarity between the target audio and text description
- [Judge](https://huggingface.co/facebook/sam-audio-judge): measures the overall separation quality across 3 axes: precision, recall, and faithfulness (see the [model card](https://huggingface.co/facebook/sam-audio-judge#output-format) for more details)
- [ImageBind](https://github.com/facebookresearch/ImageBind): for visual prompting, we measure the imagebind embedding similarity between the separated audio and the masked input video

We provide support for generating multiple candidates (by setting `reranking_candidates=<k>` in your call to `separate`), which will generate `k` audios, and choose the best one based on the ranking models mentioned above

# Models

Below is a table of each of the models we released along with their overall subjective evaluation scores

| Model    | General SFX | Speech | Speaker | Music | Instr(wild) | Instr(pro) |
|----------|-------------|--------|---------|-------|-------------|------------|
| [`sam-audio-small`](https://huggingface.co/facebook/sam-audio-small) | 3.62        | 3.99   | 3.12    | 4.11  | 3.56        | 4.24       |
| [`sam-audio-base`](https://huggingface.co/facebook/sam-audio-base)   | 3.28        | 4.25   | 3.57    | 3.87  | 3.66        | 4.27       |
| [`sam-audio-large`](https://huggingface.co/facebook/sam-audio-large) | 3.50        | 4.03   | 3.60    | 4.22  | 3.66        | 4.49       |

We additional release another variant (in each size) that is better specifically on correctness of target sound as well as visual prompting:
- [`sam-audio-small-tv`](https://huggingface.co/facebook/sam-audio-small-tv)
- [`sam-audio-base-tv`](https://huggingface.co/facebook/sam-audio-base-tv)
- [`sam-audio-large-tv`](https://huggingface.co/facebook/sam-audio-large-tv)

## Evaluation

See the [eval](eval) directory for instructions and scripts to reproduce results from the paper

## Contributing

See [contributing](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for more information.

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file for details.

## Citing SAM Audio

If you use SAM Audio in your research, please use the following BibTex entry:

```bibtex
@article{shi2025samaudio,
    title={SAM Audio: Segment Anything in Audio},
    author={Bowen Shi and Andros Tjandra and John Hoffman and Helin Wang and Yi-Chiao Wu and Luya Gao and Julius Richter and Matt Le and Apoorv Vyas and Sanyuan Chen and Christoph Feichtenhofer and Piotr Doll{\'a}r and Wei-Ning Hsu and Ann Lee},
    year={2025},
    url={https://arxiv.org/abs/2512.18099}
}
```
