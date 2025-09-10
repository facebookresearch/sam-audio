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

transform = model.get_transform()
batch = transform(
    audio_paths=[file],
    descriptions=[description],
).to("cuda)

result = model.separate(batch)

torchaudio.save("target.wav", result.target.cpu(), 48_000)
torchaudio.save("residual.wav", result.residual.cpu(), 48_000)
```
