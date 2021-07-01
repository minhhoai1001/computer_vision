import argparse
import torch
import config
import models
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated

def predict(model, loader):
    model.eval()

    with torch.no_grad():
        for img, y in loader:
            x = img.to(config.DEVICE)
            output = model(x)
            _, preds = torch.max(output.data, 1)
            pred = preds.cpu().numpy()[0]
            if pred == 0: 
                lable = 'cat'
            else:
                lable = 'dog'
            # Get a batch of training data
            out = utils.make_grid(x)
            imshow(out, lable)

def check_accuracy(model, loader, loss):
    pass

def main(args):
    model = models.efficientNet('efficientnet-b0', num_output=2, pretrained=False)
    model = model.to(config.DEVICE)
    model =  models.load_checkpoint(torch.load(args.weight), model)
    test_dataset = datasets.ImageFolder(args.img_path, transform=config.val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    predict(model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='b0.pth')
    parser.add_argument('--img_path', type=str, default='./data/test')
    args = parser.parse_args()
    main(args)