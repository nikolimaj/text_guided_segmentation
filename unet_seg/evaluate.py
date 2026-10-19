import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torchmetrics import JaccardIndex, Accuracy
from torch.utils.data import DataLoader

from dataset import MoNuSegUNet, QaTaUNet
from unet import UNet

dice_metric = DiceMetric(include_background=False, reduction="mean")
iou_metric = JaccardIndex(num_classes=2)
hd95_metric = HausdorffDistanceMetric(percentile=95)
accuracy_metric = Accuracy(task='binary')


all_preds = []
all_labels = []

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# load model
model = UNet(n_classes=1).to(device)
checkpoint = torch.load('/app/qata_models/unet.pth',map_location='cpu')
model.load_state_dict(checkpoint,strict=True)

model.eval()

dl_test = DataLoader(
    QaTaUNet(
        csv_path="./QaTa-COV19-v2/test_annotations.csv",
        root_path="./QaTa-COV19-v2/test",
        mode="test",
        image_size=[224, 224]
    ),
    batch_size=8,
    shuffle=False,
    num_workers=8
)

# dl_test = DataLoader(
#     MoNuSegUNet(
#         csv_path="./MoNuSeg/test_annotations.csv",
#         root_path="./MoNuSeg/test",
#         mode="test",
#         image_size=[224, 224]
#     ),
#     batch_size=8,
#     shuffle=False,
#     num_workers=8
# )

with torch.no_grad():
    for batch in dl_test:
        images, labels = batch 

        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        preds_bin = (preds > 0.5).float() # binarize
        
        # cast to int for metric calculation
        preds_bin = preds_bin.int()
        labels = labels.int()

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
# QaTa-UNet Model
# Dice Score:    0.7559
# IoU Score:     0.8525
# Hausdorff95:   33.3305
# Accuracy:      0.9655

# ===== Evaluation Results =====
# MoNuSeg-UNet Model
# Dice Score:    0.7476
# IoU Score:     0.7351
# Hausdorff95:   5.3089
# Accuracy:      0.8895