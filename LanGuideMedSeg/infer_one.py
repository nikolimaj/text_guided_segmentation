# infer_one.py
from __future__ import annotations

import numpy as np
import cv2
import torch
from transformers import AutoTokenizer

from LanGuideMedSeg.engine.wrapper import LanGuideMedSegWrapper
import utils.config as config


def load_text_guided_model(cfg_path: str, ckpt_path: str, device: str = "cpu"):
    """
    Load LanGuideMedSeg model + tokenizer from config and checkpoint.
    """
    cfg = config.load_cfg_from_cfg_file(cfg_path)

    model = LanGuideMedSegWrapper(cfg)

    # Load checkpoint (trusted source)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Remove key that can differ between transformers versions
    bad_key = "model.text_encoder.model.bert.embeddings.position_ids"
    if isinstance(state_dict, dict) and bad_key in state_dict:
        state_dict.pop(bad_key)

    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_type, trust_remote_code=True)
    return model, tokenizer, cfg


def _preprocess_image_gray01(img01: np.ndarray, image_size: int) -> torch.Tensor:
    """Convert grayscale image [0,1] to tensor [1,1,image_size,image_size]."""
    x = cv2.resize(img01, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    return x


@torch.no_grad()
def predict_mask_one(model, tokenizer, cfg, img01: np.ndarray, prompt: str, device: str = "cpu") -> np.ndarray:
    """
    Run single-image text-guided segmentation.

    Your model expects text length = 24 (based on the conv error).
    """
    H, W = img01.shape[:2]

    img_size = getattr(cfg, "image_size", [224, 224])
    if isinstance(img_size, (list, tuple)):
        img_size = int(img_size[0])
    else:
        img_size = int(img_size)

    # Image tensor
    x_img = _preprocess_image_gray01(img01, img_size).to(device)

    # IMPORTANT: fixed text length required by the model
    max_len = 24

    tok = tokenizer(
        prompt if prompt is not None else "",
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    # Pass ONLY these 2 keys
    text = {
        "input_ids": tok["input_ids"].to(device),
        "attention_mask": tok["attention_mask"].to(device),
    }

    # Model input: (image, text)
    x = (x_img, text)

    out = model(x)

    # Unwrap output
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, dict):
        for k in ["logits", "pred", "mask", "out"]:
            if k in out:
                out = out[k]
                break

    # Convert to probability map
    if torch.is_tensor(out):
        if out.dim() == 4:
            out = out[0, 0]
        elif out.dim() == 3:
            out = out[0]
        prob = torch.sigmoid(out).detach().cpu().numpy()
    else:
        prob = np.asarray(out)

    # Threshold + resize back
    mask_small = (prob > 0.5).astype(np.uint8)
    mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask
