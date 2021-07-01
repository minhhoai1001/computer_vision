import argparse
import time
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import efficientNet, save_checkpoint
from dataset import DodAndCatDataset
import config
from utils import EarlyStopping, LRScheduler, VisualResult

def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy

# validation function
def validate(model, test_dataloader, val_dataset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(len(val_dataset)/test_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(config.DEVICE), data[1].to(config.DEVICE)
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss / counter
        val_accuracy = 100. * val_running_correct / total
        return val_loss, val_accuracy

def main(agrs):
    train_dataset = DodAndCatDataset(root=agrs.train_path, transform=config.train_transform)
    val_dataset = DodAndCatDataset(root=agrs.val_path, transform=config.val_transform)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = efficientNet(name='efficientnet-b0', num_output=2, pretrained=True)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # either initialize early stopping or learning rate scheduler
    if args.lr_scheduler:
        print('INFO: Initializing learning rate scheduler')
        lr_scheduler = LRScheduler(optimizer)
    if args.early_stopping:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()

    # lists to store per-epoch loss and accuracy values
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    start = time.time()

    for epoch in range(agrs.epochs):
        print(f"Epoch {epoch+1} of {agrs.epochs}")
        train_epoch_loss, train_epoch_accuracy = fit(model, train_loader, train_dataset, optimizer, criterion)
        val_epoch_loss, val_epoch_accuracy = validate(model, val_loader, val_dataset, criterion)

        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        if args.lr_scheduler:
            lr_scheduler(val_epoch_loss)
        if args.early_stopping:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break
        print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
    end = time.time()
    print(f"Training time: {(end-start)/60:.3f} minutes")
    print('Saving loss and accuracy plots...')
    VisualResult(train_accuracy, val_accuracy, train_loss, val_loss)
    # serialize the model to disk
    print('Saving model...')
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)
    
    print('TRAINING COMPLETE')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_scheduler', dest='lr_scheduler', action='store_true')
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true')
    parser.add_argument('--train_path', type=str, default='./data/train')
    parser.add_argument('--val_path', type=str, default='./data/val')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)