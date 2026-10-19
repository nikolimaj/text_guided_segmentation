# app.py
import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import torch

from unet_seg.unet import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Lambdad, Resized, NormalizeIntensityd, ToTensord
import tempfile
import os
import imageio

ROOT = Path(__file__).parent.resolve()

LANGUIDE_DIR = ROOT / "LanGuideMedSeg"
sys.path.append(str(LANGUIDE_DIR))

from LanGuideMedSeg.infer_one import load_text_guided_model, predict_mask_one  # noqa: E402

dataset = "qata_models"

UNET_WEIGHTS = dataset + "/unet.pth"
LANGUIDE_CFG = str(LANGUIDE_DIR / "config" / "training_qata.yaml")
LANGUIDE_CKPT = dataset + "/text_guided.ckpt"

MODEL_IMAGE_SIZE = (224, 224)

IMG_WIDTH = 360
NAVY = "#0b1c2d"
NAVY_HOVER = "#142f4f"
ACCENT_TEXT_GUIDED = "#1f77b4"


def read_image(uploaded_file):
    # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    
    # use normalization
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    return img


def read_image(uploaded_file):
    # Read image for display (OpenCV, grayscale, normalized)
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return img

def preprocess_monai_for_model(uploaded_file):
    # Save uploaded file to a temporary location for MONAI loading
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    trans = Compose([
        LoadImaged(keys=["image"], reader="PILReader"),
        EnsureChannelFirstd(keys=["image"]),
        Lambdad(keys=["image"], func=lambda x: x.mean(axis=0, keepdims=True)),
        Resized(keys=["image"], spatial_size=MODEL_IMAGE_SIZE, mode="bicubic"),
        NormalizeIntensityd(keys=["image"], channel_wise=True),
        ToTensord(keys=["image"]),
    ])
    data = {"image": tmp_path}
    data = trans(data)
    img = data["image"]
    os.remove(tmp_path)
    img = img.squeeze().cpu().numpy().astype(np.float32)
    return img


def preprocess_opencv(img01, clahe_clip_limit=0, denoise_h=0):
    x = (img01 * 255).astype(np.uint8)

    if denoise_h > 0:
        x = cv2.fastNlMeansDenoising(x, h=denoise_h)

    if clahe_clip_limit > 0:
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        x = clahe_obj.apply(x)
    
    return x.astype(np.float32) / 255.0


def binarize_mask(mask01, thr=0.5):
    return (mask01 >= thr).astype(np.uint8)


def accuracy_score(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    return float((pred == gt).mean())


def dice_score(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = np.sum((pred == 1) & (gt == 1))
    denom = np.sum(pred == 1) + np.sum(gt == 1)
    return float((2 * inter + 1e-8) / (denom + 1e-8))


def iou_score(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    inter = np.sum((pred == 1) & (gt == 1))
    union = np.sum((pred == 1) | (gt == 1))
    return float((inter + 1e-8) / (union + 1e-8))


def hausdorff_distance(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        h, w = pred.shape
        return float(np.sqrt(h * h + w * w))

    kernel = np.ones((3, 3), np.uint8)
    pred_er = cv2.erode(pred, kernel, iterations=1)
    gt_er = cv2.erode(gt, kernel, iterations=1)
    pred_b = (pred - pred_er).astype(np.uint8)
    gt_b = (gt - gt_er).astype(np.uint8)

    dt_gt = cv2.distanceTransform((1 - gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((1 - pred_b).astype(np.uint8), cv2.DIST_L2, 3)

    d_pred_to_gt = float(dt_gt[pred_b == 1].max()) if pred_b.sum() > 0 else 0.0
    d_gt_to_pred = float(dt_pred[gt_b == 1].max()) if gt_b.sum() > 0 else 0.0
    return max(d_pred_to_gt, d_gt_to_pred)


def overlay_mask_on_image(img01, mask01, alpha=0.45, color_bgr=(0, 0, 255)):
    img_u8 = (img01 * 255).clip(0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    color_layer = np.zeros_like(img_rgb)
    color_layer[:, :, 0] = color_rgb[0]
    color_layer[:, :, 1] = color_rgb[1]
    color_layer[:, :, 2] = color_rgb[2]

    m = (mask01.astype(np.uint8) * 255)
    out = img_rgb.copy()
    out[m == 255] = cv2.addWeighted(img_rgb[m == 255], 1 - alpha, color_layer[m == 255], alpha, 0)
    
    return out


def metrics_df(pred, gt):
    if gt is None:
        return pd.DataFrame({"Metric": ["Accuracy", "Dice", "Jaccard", "Hausdorff (px)"], "Value": ["—", "—", "—", "—"]})

    acc = accuracy_score(pred, gt)
    dice = dice_score(pred, gt)
    iou = iou_score(pred, gt)
    hd = hausdorff_distance(pred, gt)

    return pd.DataFrame(
        {
            "Metric": ["Accuracy", "Dice", "Jaccard", "Hausdorff (px)"],
            "Value": [f"{acc:.2f}", f"{dice:.2f}", f"{iou:.2f}", f"{hd:.2f}"],
        }
    )


def title_row(text, dot_color=None):
    if dot_color:
        st.markdown(
            f"<div class='sec-title'><span class='dot' style='background:{dot_color};'></span>{text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div class='sec-title'>{text}</div>", unsafe_allow_html=True)


@st.cache_resource
def load_unet_model(weights_path: Path):
    # if not weights_path.exists():
    #     raise FileNotFoundError(f"U-Net weights not found: {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(n_classes=1).to(device)

    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state, strict=False)

    model.eval()
    return model, device


def predict_unet_real(img01, model, device):
    x = cv2.resize(img01, MODEL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        p = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    mask = (p > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (img01.shape[1], img01.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


@st.cache_resource
def load_text_guided_cached():
    ckpt_path = Path(LANGUIDE_CKPT)
    # if not ckpt_path.exists():
    #     return None, None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg = load_text_guided_model(LANGUIDE_CFG, str(ckpt_path), device=device)
    return model, tokenizer, cfg, device


st.set_page_config(page_title="Medical Image Segmentation – Text Guided", layout="wide")

st.markdown(
    f"""
    <style>
      img {{ border: none !important; box-shadow: none !important; }}
      .stImage {{ border: none !important; box-shadow: none !important; }}
      .stImage > img {{ border: none !important; box-shadow: none !important; }}

      div[data-testid="column"] {{ padding-left: 6px !important; padding-right: 6px !important; }}

      .stProgress > div > div > div > div {{
        background-color: {NAVY} !important;
        border-radius: 999px !important;
      }}
      .st-emotion-cache-zy6yx3 {{
          padding-top: 1rem !important;
      }}

      div.stButton > button {{
        background-color: {NAVY} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.45em 1.25em !important;
        font-weight: 700 !important;
      }}

      div.stButton > button:hover {{
        background-color: {NAVY_HOVER} !important;
        color: white !important;
      }}

      .sec-title {{
        font-size: 22px;
        font-weight: 800;
        color: #111827;
        margin: 10px 0 10px 0;
      }}

      .dot {{
        display:inline-block;
        width: 12px;
        height: 12px;
        border-radius: 999px;
        margin-right: 8px;
        transform: translateY(1px);
      }}

      .metrics-title {{
        font-size: 18px;
        font-weight: 800;
        color: #111827;
        margin: 10px 0 10px 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Medical Image Segmentation – Text Guided")

st.session_state.setdefault("img_file_saved", None)
st.session_state.setdefault("gt_file_saved", None)
st.session_state.setdefault("prompt", "")
st.session_state.setdefault("last_pred_unet", None)
st.session_state.setdefault("last_pred_text", None)

# Load models on startup
unet_model, unet_device = load_unet_model(UNET_WEIGHTS)
tg_model, tg_tokenizer, tg_cfg, tg_device = load_text_guided_cached()

st.sidebar.header("Preprocessing (OpenCV)")
denoise_h = st.sidebar.slider(
    "Denoising", min_value=0, max_value=19, value=0, step=1,
    help="Higher values remove more noise but may remove details."
)
clahe_clip_limit = st.sidebar.slider(
    "Contrast Enhancement", min_value=0.0, max_value=5.0, value=0.0, step=0.1,
    help="Higher values give more contrast enhancement."
)

img_file = None
gt_file = None

left, right = st.columns([1.0, 1.25], gap="large")

with left:
    if img_file is None:
        img_file = st.session_state["img_file_saved"]
    if "gt_file" not in locals() or gt_file is None:
        gt_file = st.session_state["gt_file_saved"]

    tab_img, tab_gt = st.tabs(["Input image", "Ground Truth"])

    with tab_img:
        img_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key="img_up_tab",
        )
        # Reset all values if a new image is uploaded
        if img_file is not None:
            if (
                "last_uploaded_img_file" not in st.session_state
                or st.session_state["last_uploaded_img_file"] != img_file
            ):
                st.session_state["img_file_saved"] = img_file
                st.session_state["gt_file_saved"] = None
                st.session_state["prompt"] = ""
                st.session_state["last_pred_unet"] = None
                st.session_state["last_pred_text"] = None
                st.session_state["text_guided_ran_once"] = False
                st.session_state["last_uploaded_img_file"] = img_file
            else:
                st.session_state["img_file_saved"] = img_file
        # Display input image in tab
        if "img_file_saved" in st.session_state and st.session_state["img_file_saved"] is not None:
            img01_tab = read_image(st.session_state["img_file_saved"])
            if img01_tab is not None:
                title_row("Input Image")
                img01_tab_proc = preprocess_opencv(img01_tab, clahe_clip_limit=clahe_clip_limit, denoise_h=denoise_h)
                st.image((img01_tab_proc * 255).astype(np.uint8), caption="Input Image (preprocessed)", clamp=True, width=IMG_WIDTH)

    with tab_gt:
        gt_file = st.file_uploader(
            "Upload Ground Truth Mask (optional)",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            key="gt_up_tab",
        )
        if gt_file is not None:
            st.session_state["gt_file_saved"] = gt_file
        # Display ground truth image in tab
        if "gt_file_saved" in st.session_state and st.session_state["gt_file_saved"] is not None:
            gt01_tab = read_image(st.session_state["gt_file_saved"])
            if gt01_tab is not None:
                # write title
                title_row("Ground Truth")
                img01_tab_proc = preprocess_opencv(img01_tab, clahe_clip_limit=clahe_clip_limit, denoise_h=denoise_h)
                gt_overlay = overlay_mask_on_image(img01_tab_proc, gt01_tab, color_bgr=(0, 255, 0))
                st.image(gt_overlay, caption=None, clamp=True, width=IMG_WIDTH, output_format="PNG")
        
    if img_file is None:
        st.stop()

    img01_raw = read_image(img_file)
    if img01_raw is None:
        st.error("Could not read the image. Please try a different format.")
        st.stop()

    # For display, use OpenCV preprocessing
    img01 = preprocess_opencv(img01_raw, clahe_clip_limit=clahe_clip_limit, denoise_h=denoise_h)

    # For both UNet and text-guided model, use OpenCV preprocessing before MONAI
    # Read and preprocess with OpenCV
    img01_opencv = preprocess_opencv(img01_raw, clahe_clip_limit=clahe_clip_limit, denoise_h=denoise_h)

    # Save OpenCV-preprocessed image to temp file for MONAI

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        imageio.imwrite(tmp.name, (img01_opencv * 255).astype(np.uint8))
        opencv_temp_path = tmp.name

    def monai_from_path(image_path):
        trans = Compose([
            LoadImaged(keys=["image"], reader="PILReader"),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys=["image"], func=lambda x: x.mean(axis=0, keepdims=True)),
            Resized(keys=["image"], spatial_size=MODEL_IMAGE_SIZE, mode="bicubic"),
            NormalizeIntensityd(keys=["image"], channel_wise=True),
            ToTensord(keys=["image"]),
        ])
        data = {"image": image_path}
        data = trans(data)
        img = data["image"]
        img = img.squeeze().cpu().numpy().astype(np.float32)
        return img

    img01_monai = monai_from_path(opencv_temp_path)

    gt01 = None
    if gt_file is not None:
        gt_img = read_image(gt_file)
        if gt_img is not None:
            if gt_img.shape != img01.shape:
                gt_img = cv2.resize(gt_img, (img01.shape[1], img01.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt01 = binarize_mask(gt_img, 0.5)

    # Run UNet segmentation with MONAI-preprocessed image
    pred_unet = predict_unet_real(img01_monai, unet_model, unet_device)
    # Remove temp file after MONAI preprocessing
    os.remove(opencv_temp_path)
    # Fix orientation if mask is mirrored or rotated
    mask_unet_fixed = pred_unet
    if mask_unet_fixed.shape != img01.shape:
        mask_unet_fixed = cv2.resize(mask_unet_fixed, (img01.shape[1], img01.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Try flipping horizontally if mirrored
    if not np.allclose(mask_unet_fixed, img01):
        mask_unet_fixed = np.fliplr(mask_unet_fixed)
        # turn it 90 degrees to left if still not matching
        if not np.allclose(mask_unet_fixed, img01):
            mask_unet_fixed = np.rot90(mask_unet_fixed, k=1)
    st.session_state["last_pred_unet"] = mask_unet_fixed
    st.session_state["last_pred_unet_metrics"] = mask_unet_fixed

    title_row("Text Prompt")
    prompt = st.text_area(
        "Enter medical text prompt",
        key="prompt",
        height=160,
        placeholder="e.g., COVID-19 lung infection",
    )
    run = st.button("Run", type="primary")

    # Track if text-guided segmentation has ever been run
    if "text_guided_ran_once" not in st.session_state:
        st.session_state["text_guided_ran_once"] = False

    # If button is clicked, set the flag and run text-guided segmentation
    if run:
        st.session_state["text_guided_ran_once"] = True
        try:
            if tg_model is None:
                st.warning(
                    "Text-guided checkpoint not found.\n\n"
                    f"Expected path:\n{LANGUIDE_CKPT}\n\n"
                    "Showing UNet result instead."
                )
                st.session_state["last_pred_text"] = st.session_state["last_pred_unet"]
            else:
                pred_text = predict_mask_one(
                    model=tg_model,
                    tokenizer=tg_tokenizer,
                    cfg=tg_cfg,
                    img01=img01_monai,
                    prompt=prompt,
                    device=tg_device,
                )
                st.session_state["last_pred_text"] = pred_text
        except Exception as e:
            st.error(f"Text-guided inference error: {e}")

    # If the button has ever been clicked, update text-guided segmentation on every change
    elif st.session_state["text_guided_ran_once"]:
        try:
            if tg_model is None:
                st.session_state["last_pred_text"] = st.session_state["last_pred_unet"]
            else:
                pred_text = predict_mask_one(
                    model=tg_model,
                    tokenizer=tg_tokenizer,
                    cfg=tg_cfg,
                    img01=img01_monai,
                    prompt=prompt,
                    device=tg_device,
                )
                st.session_state["last_pred_text"] = pred_text
        except Exception as e:
            st.error(f"Text-guided inference error: {e}")

if run:
    # progress = st.progress(0, text="Running inference...")
    try:
        # progress.progress(55, text="Text-Guided model...")
        if tg_model is None:
            st.warning(
                "Text-guided checkpoint not found.\n\n"
                f"Expected path:\n{LANGUIDE_CKPT}\n\n"
                "Showing UNet result instead."
            )
            st.session_state["last_pred_text"] = st.session_state["last_pred_unet"]
        else:
            # progress.progress(80, text="Text-guided inference...")
            pred_text = predict_mask_one(
                model=tg_model,
                tokenizer=tg_tokenizer,
                cfg=tg_cfg,
                img01=img01_monai,
                prompt=prompt,
                device=tg_device,
            )
            st.session_state["last_pred_text"] = pred_text

        # progress.progress(100, text="Done!")
    except Exception as e:
        st.error(f"Inference error: {e}")

pred_unet = st.session_state["last_pred_unet"]
pred_text = st.session_state["last_pred_text"]

with right:
    title_row("UNet Segmentation")
    unet_overlay = overlay_mask_on_image(img01, pred_unet, color_bgr=(0, 0, 255))
    img_col, met_col = st.columns([1.15, 0.85], gap="small")
    with img_col:
        st.image(unet_overlay, width=IMG_WIDTH, output_format="PNG")
    with met_col:
        st.markdown("<div class='metrics-title'>Metrics</div>", unsafe_allow_html=True)
        st.dataframe(metrics_df(st.session_state.get("last_pred_unet_metrics", pred_unet), gt01), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    title_row("Text-Guided Segmentation")
    if pred_text is None:
        st.info("Provide a text prompt to see text-guided segmentation results.")
    else:
        # Fix orientation if mask is mirrored or rotated
        mask_fixed = pred_text
        if mask_fixed.shape != img01.shape:
            mask_fixed = cv2.resize(mask_fixed, (img01.shape[1], img01.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Try flipping horizontally if mirrored
        if not np.allclose(mask_fixed, pred_unet):
            mask_fixed = np.fliplr(mask_fixed)
            # turn it 90 degrees to left
            if not np.allclose(mask_fixed, pred_unet):
                mask_fixed = np.rot90(mask_fixed, k=1)
        st.session_state["last_pred_text_metrics"] = mask_fixed
        tg_overlay = overlay_mask_on_image(img01, mask_fixed, color_bgr=(255, 0, 0))
        img_col2, met_col2 = st.columns([1.15, 0.85], gap="small")
        with img_col2:
            st.image(tg_overlay, width=IMG_WIDTH, output_format="PNG")
        with met_col2:
            st.markdown("<div class='metrics-title'>Metrics</div>", unsafe_allow_html=True)
            st.dataframe(metrics_df(st.session_state.get("last_pred_text_metrics", pred_text), gt01), use_container_width=True, hide_index=True)

# if gt01 is not None:
#     st.divider()
#     title_row("Ground Truth Preview")
#     gt_overlay = overlay_mask_on_image(img01, gt01, color_bgr=(0, 255, 0))
#     st.image(gt_overlay, caption=None, clamp=True, width=IMG_WIDTH, output_format="PNG")
