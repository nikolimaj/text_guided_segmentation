import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from dataset import QaTaUNet, MoNuSegUNet, MosMedUNet
import numpy as np


def dice_coef(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()


train_set = QaTaUNet(
    csv_path="./QaTa-COV19-v2/train_annotations.csv",
    root_path="./QaTa-COV19-v2/train",
    mode="train",
    image_size=[224, 224]
)
valid_set = QaTaUNet(
    csv_path="./QaTa-COV19-v2/train_annotations.csv",
    root_path="./QaTa-COV19-v2/train",
    mode="valid",
    image_size=[224, 224]
)

# train_set = MoNuSegUNet(
#     csv_path="./MoNuSeg/train_annotations.csv",
#     root_path="./MoNuSeg/train",
#     mode="train",
#     image_size=[224, 224]
# )
# valid_set = MoNuSegUNet(
#     csv_path="./MoNuSeg/val_annotations.csv",
#     root_path="./MoNuSeg/val",
#     mode="valid",
#     image_size=[224, 224]
# )

# train_set = MosMedUNet(csv_path="./MosMed/train_annotations.csv",
#                 root_path="./MosMed/train",
#                 mode='train',
#                 image_size=[224, 224])

# valid_set = MosMedUNet(csv_path="./MosMed/train_annotations.csv",
#                 root_path="./MosMed/train",
#                 mode='valid',
#                 image_size=[224, 224])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=4)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = UNet(n_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val = -np.inf

min_epochs = 20
max_epochs = 100
patience = 20

best_val = -np.inf
epochs_no_improve = 0
best_epoch = 0

for epoch in range(max_epochs):
    model.train()
    train_loss = 0
    train_dice = 0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
        images = images.to(device)
        masks = masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_dice += dice_coef(preds, masks).item()

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)

    # Validation
    model.eval()
    valid_loss = 0
    valid_dice = 0
    with torch.no_grad():
        for images, masks in valid_loader:
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            valid_loss += loss.item()
            valid_dice += dice_coef(preds, masks).item()
    valid_loss /= len(valid_loader)
    valid_dice /= len(valid_loader)

    print(f"Epoch {epoch+1}/{max_epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
    print(f"  Valid Loss: {valid_loss:.4f} | Valid Dice: {valid_dice:.4f}")

    # Save best model
    if valid_dice > best_val:
        torch.save(model.state_dict(), f"qata_models/unet.pth")
        best_val = valid_dice
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Early stopping (only after min_epochs)
    if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
        break

print("Training completed. Best model saved.")