import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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

if __name__ == '__main__':
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    catdataset = DodAndCatDataset(root='./data/val', transform=val_transform)
    loader = DataLoader(catdataset, batch_size=1, shuffle=True, num_workers=4)

    print(len(loader))
    # for x, y in loader:
    #     print(x[0])
    #     print(y[0])
    inputs, classes = next(iter(loader))
    out = utils.make_grid(inputs)
    print(classes)
        