"""Object localisation from a SAM mask instead of the detection box.

Development runs found that 7 of 8 false declarations were the right object at the wrong
place: the median depth over a Grounding DINO box mixes the object with
whatever is in front of or behind it (a table before a sofa, the wall behind a
toilet), and the track ends up ~1 m off. Here the box prompts SAM
(facebook/sam-vit-base, inference only), and the object is placed at the 3-D
median of the depth pixels inside the mask; its width is the lateral spread of
those points. Grounded-SAM style; no simulator truth is read.
"""
from __future__ import annotations

import numpy as np

SAM_ID = 'facebook/sam-vit-base'
MIN_MASK_PIXELS = 50


class MaskLocalizer:
    def __init__(self, device='cuda', half=True):
        import torch
        from transformers import SamModel, SamProcessor
        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if (half and device == 'cuda') else torch.float32
        self.processor = SamProcessor.from_pretrained(SAM_ID)
        self.model = SamModel.from_pretrained(SAM_ID, torch_dtype=self.dtype).to(device).eval()
        self.calls = 0

    def masks(self, rgb, boxes):
        """Best mask per box (bool HxW), one image encoding for all boxes."""
        torch = self.torch
        img = np.asarray(rgb)[..., :3]
        inputs = self.processor(images=img, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors='pt')
        with torch.no_grad():
            pv = inputs['pixel_values'].to(self.device, self.dtype)
            emb = self.model.get_image_embeddings(pv)
            out = self.model(image_embeddings=emb, input_boxes=inputs['input_boxes'].to(self.device, self.dtype),
                             multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.float().cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0]
        scores = out.iou_scores.float().cpu()[0]          # (boxes, 3)
        self.calls += 1
        best = scores.argmax(dim=-1)
        return [masks[i, int(best[i])].numpy().astype(bool) for i in range(len(boxes))]

    @staticmethod
    def place(mask, depth, eye, rotation, hfov_deg, box):
        """3-D median of the mask's depth pixels and the lateral width (m)."""
        d = np.asarray(depth, dtype=np.float32)
        h, w = d.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        keep = np.zeros_like(mask)
        keep[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True     # stay inside the prompt box
        m = mask & keep & np.isfinite(d) & (d > 0.05)
        if m.sum() < MIN_MASK_PIXELS:
            return None
        fx = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        vs, us = np.nonzero(m)
        z = d[vs, us]
        u = (us - w / 2.0) / fx
        v = (vs - h / 2.0) / fx
        cam = np.stack([u * z, -v * z, -z], axis=1)
        pts = cam @ np.asarray(rotation).T + np.asarray(eye)
        lateral = cam[:, 0]
        width = float(np.percentile(lateral, 95) - np.percentile(lateral, 5))
        return np.median(pts, axis=0), width, float(np.median(z))
