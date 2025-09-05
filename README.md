# SAM-Audio

Segment Anything Model for Audio

## Setup

Install dependencies

```
pip install .
```

## Usage

```python
from sam_audio.model import SAMAudio
import torchaudio

model = SAMAudio.from_config(
    "base", pretrained=True, checkpoint_path="<checkpoint path>",
)
model = model.eval().cuda()

file = ... # audio file
description = "Raindrops are falling heavily, splashing on the ground."

result_wav = model.separate(
    audio_paths=[file],
    descriptions=[description],
)

torchaudio.save("result.wav", result_wav[0].cpu(), 48_000)
```
