import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FaceCatLandmark(Dataset):
    def __init__(self, txt_path, transform=None):
        self.data_path = open(txt_path, 'r')
        self.data = self.data_path.readlines()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def read_label(self, txt_path, img_h, img_w):
        f = open(txt_path, 'r')
        lines = f.readline().strip(" ")  # remove " " and begin and end of lines
        points = lines.split(" ")
        num_point = int(points.pop(0)) # remove element 0
        label = []
        for i in range(len(points)):
            data = 0
            if i%2 == 0:
                data = int(int(points[i])*224.0/img_w)
            else:
                data = int(int(points[i])*224.0/img_h)
            label.append(data)
        
        return np.array(label)

    def __getitem__(self, idx):
        img_path = self.data[idx].strip()
        image = Image.open(img_path)
        img_h = image.height
        img_w = image.width
        label_path = self.data[idx].strip() + '.cat'
        label = self.read_label(label_path, img_h, img_w)

        if self.transform is not None:
            img = self.transform(image)

        return img, torch.Tensor(label)

# if __name__ == '__main__':
#     img_transform = transforms.ToTensor()
#     catdataset = FaceCatLandmark(txt_path='data/val.txt', transform=img_transform, label_transform=None)
#     loader = DataLoader(catdataset, batch_size=1, shuffle=True, num_workers=4)

#     # # print(len(loader))
#     # for x, y in loader:
#     #     print(x[0])
#     #     print(y[0])
#     inputs, classes = next(iter(loader))
#     out = utils.make_grid(inputs)
#     print(classes)