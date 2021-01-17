## List of data for classification
1. [cassava-lead-disease](https://www.kaggle.com/c/cassava-leaf-disease-classification/data)
2. [hymenoptera_data](https://download.pytorch.org/tutorial/hymenoptera_data.zip)

## Download from [TORCHVISION.DATASETS](https://pytorch.org/docs/stable/torchvision/datasets.html)

### ImageNet
```
imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
```

```
torchvision.datasets.'data'(root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
```

data = 
* CelebA
* CIFAR
* Cityscapes
* COCO
* Captions
* Detection
* DatasetFolder
* EMNIST
* FakeData
* Fashion-MNIST
* Flickr
* HMDB51
* ImageFolder
* ImageNet
* Kinetics-400
* KMNIST
* LSUN
* MNIST
* Omniglot
* PhotoTour
* Places365
* QMNIST
* SBD
* SBU
* STL10
* SVHN
* UCF101
* USPS
* VOC