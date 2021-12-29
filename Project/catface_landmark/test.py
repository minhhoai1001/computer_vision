import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import config
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from dataset import FaceCatLandmark
from utils import load_checkpoint

def imshow(inp, landmarks, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    # x = [classes.numpy()[0][i] for i in range(len(classes.numpy()[0])) if i%2==0]
    # y = [classes.numpy()[0][i] for i in range(len(classes.numpy()[0])) if i%2==1]
    # plt.plot(x, y, 'bo')
    landmarks = landmarks.numpy().reshape(-1, 2) # Get data from batch 0 and convert tensor to numpy
    # print(landmarks)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=20, marker='.', c='r')
    nose_line = np.concatenate((landmarks[:3,:],landmarks[0:1,:]))
    ear_line = np.concatenate((landmarks[3:,:],landmarks[3:4,:]))
    # print(f"nose_line: {landmarks[0:1,:]}")
    # plt.plot(landmarks[3:,0], landmarks[3:,1], c='b')
    plt.plot(nose_line[:,0], nose_line[:,1], c='b')
    plt.plot(ear_line[:,0], ear_line[:,1], c='b')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated
    plt.close()

def predict(loader, model,  device):
    model.eval()
    num_examples = 0
    losses = []
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            # targets = targets.to(device=device)

            # forward
            scores = model(data)
            # loss = loss_fn(scores[targets != -1], targets[targets != -1])
            num_examples += scores[targets != -1].shape[0]
            # losses.append(loss.item())
            # print(f"num example: {num_examples}")
            # print(f"score: {scores.cpu().numpy()}")
            out = utils.make_grid(data)
            imshow(out.cpu(), scores.cpu())
            

def main():
    img_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    catdataset = FaceCatLandmark(txt_path='data/test.txt', transform=img_transform)
    loader = DataLoader(catdataset, batch_size=1, shuffle=False, num_workers=4)

    # inputs, landmarks = next(iter(loader))
    # out = utils.make_grid(inputs)
    # imshow(out, landmarks)
    # print(landmarks)

    model = EfficientNet.from_pretrained("efficientnet-b0")
    feature = model._fc.in_features
    model._fc = torch.nn.Linear(feature, 18)
    model = model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    predict(loader, model, config.DEVICE)

if __name__ == '__main__':
    main()