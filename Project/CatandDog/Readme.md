# The simple project to classification dog and cat with pytorch

## Directory Structure 
```
├─── data
│    ├─── train
│    ├─── val    
│    ├─── test    
│         ├─── cat
│         └─── dog
│           
├───outputs
│
├─── dataset.py
├─── models.py
├─── train.py
├─── utils.py
├─── config.py
├─── test.py

```
## Input Data
The data from [Cats and Dogs dataset to train a DL model](https://www.kaggle.com/tongpython/cat-and-dog) of Kaggle.

## Training

```
python3 train.py --early_stopping --train_path=./data/train --val_path=./data/val --epochs=20
```

## Testing
```
python3 test.py --weight=b4.pth --img_path=./data/test```
```