import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torchmetrics import JaccardIndex, Accuracy
from torch.utils.data import DataLoader
from argparse import Namespace

from utils.dataset import MoNuSeg, QaTa
from engine.wrapper import LanGuideMedSegWrapper

dice_metric = DiceMetric(include_background=False, reduction="mean")
iou_metric = JaccardIndex(num_classes=2)
hd95_metric = HausdorffDistanceMetric(percentile=95)
accuracy_metric = Accuracy(task='binary')

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

args = Namespace(
    bert_type="bert-base-uncased",
    vision_type="facebook/convnext-tiny-224",
    project_dim=768,
    lr=0.0003
)

# load model
model = LanGuideMedSegWrapper(args).to(device)
checkpoint = torch.load('./monuseg_models/text_guided.ckpt',map_location='cpu', weights_only=False)["state_dict"]
model.load_state_dict(checkpoint,strict=False)

model.eval()

# dl_test = DataLoader(
#     QaTa(
#         csv_path="./QaTa-COV19-v2/test_annotations.csv",
#         root_path="./QaTa-COV19-v2/test",
#         mode="test",
#         tokenizer="microsoft/BiomedVLP-CXR-BERT-specialized",
#         image_size=[224, 224]
#     ),
#     batch_size=8,
#     shuffle=False,
#     num_workers=8
# )


dl_test = DataLoader(
    MoNuSeg(
        csv_path="./MoNuSeg/test_annotations.csv",
        root_path="./MoNuSeg/test",
        mode="test",
        tokenizer="bert-base-uncased",
        image_size=[224, 224]
    ),
    batch_size=8,
    shuffle=False,
    num_workers=8
)


all_preds = []
all_labels = []


with torch.no_grad():
    for batch in dl_test:
        (images, texts), labels = batch

        images = images.to(device)
        labels = labels.to(device)

        # Overwrite text prompt with empty string (simulate no text prompt)
        # if 'input_ids' in texts and 'attention_mask' in texts:
        #     # Create empty prompt tokens with same shape as original
        #     empty_prompt = ""
        #     tokenizer = texts['input_ids'].new_zeros(texts['input_ids'].shape)
        #     attention_mask = texts['attention_mask'].new_zeros(texts['attention_mask'].shape)
        #     texts['input_ids'] = tokenizer
        #     texts['attention_mask'] = attention_mask
            
        texts = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in texts.items()}

        x = [images, texts]
        preds = model(x)
        preds_bin = (preds > 0.5).float()  # binarize
        all_preds.append(preds_bin.cpu())
        all_labels.append(labels.cpu())

# ---- stack all predictions ----
pred = torch.cat(all_preds, dim=0)
gt   = torch.cat(all_labels, dim=0)

# ===== Compute metrics =====
dice_score = dice_metric(pred, gt).mean()
hd95_score = hd95_metric(pred, gt).view(-1)
hd95_score = torch.nanmean(hd95_score).item()
iou_score  = iou_metric(pred, gt)
accuracy_score = accuracy_metric(pred.int(), gt.int()).item()


print("\n===== Evaluation Results =====")
print(f"Dice Score:    {dice_score:.4f}")
print(f"IoU Score:     {iou_score:.4f}")
print(f"Hausdorff95:   {hd95_score:.4f}")
print(f"Accuracy:      {accuracy_score:.4f}")

# ===== Evaluation Results =====
# QaTa-LanGuideMedSeg Model
# Dice Score:    0.8385
# IoU Score:     0.9042
# Hausdorff95:   16.6786
# Accuracy:      0.9783

# ===== Evaluation Results =====
# QaTa-LanGuideMedSeg Model without text prompt
# Dice Score:    0.6993
# IoU Score:     0.7659
# Hausdorff95:   46.6465
# Accuracy:      0.9482

# ===== Evaluation Results =====
# MoNuSeg-LanGuideMedSeg Model
# Dice Score:    0.6865
# IoU Score:     0.6841
# Hausdorff95:   3.9053
# Accuracy:      0.8666

# ===== Evaluation Results =====
# MoNuSeg-LanGuideMedSeg Model without text prompt
# Dice Score:    0.6878
# IoU Score:     0.6843
# Hausdorff95:   4.0054
# Accuracy:      0.8657