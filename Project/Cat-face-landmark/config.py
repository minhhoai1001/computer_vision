import torch
from torchvision import transforms
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 10
NUM_EPOCHS = 200
NUM_WORKERS = 4
CHECKPOINT_FILE = "b0.pth"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

img_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.485, 0.456, 0.406), 
                                                        (0.229, 0.224, 0.225))])
