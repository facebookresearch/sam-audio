import argparse
import tempfile

import cv2
import decord
import numpy as np
import torch
import torchvision
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("video")
opt = parser.parse_args()

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast(
    "cuda", dtype=torch.bfloat16
), tempfile.TemporaryDirectory() as tdir:
    vr = decord.VideoReader(opt.video)
    print("Saving as images...")
    for i, frame in tqdm(enumerate(vr), total=len(vr)):
        file = f"{tdir}/{str(i).zfill(5)}.jpg"
        cv2.imwrite(file, frame.asnumpy())

    inference_state = predictor.init_state(video_path=tdir)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0
    ann_obj_id = 1
    points = np.array([[150, 60]])
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    masks = []
    video_segments = {}
    for _, out_obj_ids, out_mask_logits in tqdm(
        predictor.propagate_in_video(inference_state), total=len(vr)
    ):
        mask = None
        for i, _ in enumerate(out_obj_ids):
            if mask is None:
                mask = out_mask_logits[i] > 0.0
            else:
                mask = mask | (out_mask_logits[i] > 0.0)

        masks.append(mask.cpu())

    mask_frames = torch.cat(masks)
    mask_frames = mask_frames.unsqueeze(-1).expand(-1, -1, -1, 3)

    outfile = opt.video.replace(".mp4", "_mask.mp4")
    torchvision.io.write_video(
        outfile, mask_frames * 255, int(np.round(vr.get_avg_fps())), "h264"
    )
