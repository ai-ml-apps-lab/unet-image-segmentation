import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import SegmentationDataset
from unet import UNet


if __name__ == "__main__":

    # mode = "binary"
    mode = "multiclass"

    if mode == 'binary':
        out_channels = 1
        class_dict_path = None
        IMAGE_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\binary_dataset\images"
        MASK_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\binary_dataset\masks"
        criterion = nn.BCEWithLogitsLoss()

    elif mode == 'multiclass':
        out_channels = 32  # Adjust based on the number of classes
        class_dict_path = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\multiclass_dataset\class_dict.csv"
        IMAGE_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\multiclass_dataset\images"
        MASK_DIR = r"E:\AB\ai_ml_apps_lab_github_2026\5U-Net\dataset\multiclass_dataset\masks"
        criterion = nn.CrossEntropyLoss()

    MODEL_PATH = "./unet_model.pth"

    BATCH_SIZE = 2
    EPOCHS = 20
    LEARNING_RATE = 1e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



    # ------------------------
    # Dataset
    # ------------------------

    dataset = SegmentationDataset(
        IMAGE_DIR,
        MASK_DIR,
        class_dict_path = class_dict_path if mode == "multiclass" else None,
        image_size=256
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    # ------------------------
    # Model
    # ------------------------

    model = UNet(out_channels=out_channels).to(DEVICE)


    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ------------------------
    # Training
    # ------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            loss = criterion(predictions, masks)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")


    # ------------------------
    # Save Model
    # ------------------------

    torch.save(model.state_dict(), MODEL_PATH)

    print("Model saved successfully.")

