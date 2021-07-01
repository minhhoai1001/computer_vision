import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DodAndCatDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img = Image.open(os.path.join(self.root, img_name))

        if self.transform is not None:
            img = self.transform(img)

        # label_name = [x.split(".")[0] for x in dir][index]
        if 'cat' in img_name : label = 0
        elif 'dog' in img_name: label = 1
        else : label = -1

        return img, label

        