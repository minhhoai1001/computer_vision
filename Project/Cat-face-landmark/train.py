import torch
from dataset import FaceCatLandmark
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_rmse,
    get_submission
)


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    num_examples = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        scores[targets == -1] = -1
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores[targets != -1])
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss average over epoch: {(sum(losses)/num_examples)**0.5}")

def main():
    train_dataset = FaceCatLandmark(txt_path='data/train.txt', transform=config.img_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                    shuffle=False, num_workers=config.NUM_WORKERS)

    val_dataset = FaceCatLandmark(txt_path='data/val.txt', transform=config.img_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                                    shuffle=False, num_workers=config.NUM_WORKERS)

    loss_fn = nn.MSELoss(reduction="sum")
    model = EfficientNet.from_pretrained("efficientnet-b0")
    feature = model._fc.in_features
    model._fc = nn.Linear(feature, 18)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    # get_submission(test_loader, test_ds, model_15, model_4)

    for epoch in range(config.NUM_EPOCHS):
        print(f"==== Epoch: {epoch} ====")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
        get_rmse(val_loader, model, loss_fn, config.DEVICE)

        # get on validation
        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

if __name__ == "__main__":
    main()