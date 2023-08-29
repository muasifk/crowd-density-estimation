


import os
import numpy as np
import torch
import torchvision
from torch import nn
from torchinfo import summary
import matplotlib.pyplot as plt
from glob import glob

''' SEED Everything '''
from src.utils import seed_everything
seed_everything(seed=42)
print("Using Torch", torch.__version__, 'and Torchvision:', torchvision.__version__) 


''' Load training data '''
root_dir        = 'path/to/dataset/dir/' 
train_img_dir   = root_dir + '/train_data/images'       # train images inside this dir
train_gt_dir    = root_dir + '/train_data/ground-truth' # train GTs inside this dir
test_img_dir    = root_dir + '/test_data/images'        # test images inside this dir
test_gt_dir     = root_dir + '/test_data/ground-truth'  # test GTs inside this dir
train_img_paths = glob(train_img_dir + '/*.jpg')
test_img_paths  = glob(test_img_dir + '/*.jpg')
train_gt_paths  = glob(train_gt_dir + '/*.mat')
test_gt_paths   = glob(test_gt_dir + '/*.mat')
train_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
test_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
train_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
test_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        


''' Create dataset '''
from src.Dataset import CrowdDataset2 as CrowdDataset
img_size   = (800, 800, 3) # size of the input image
gt_size    = (200, 200)  # size of the density map
sigma      = 10 # Gaussian kernal
batch_size = 2
num_workers= 1
train_ds   = CrowdDataset(train_img_paths, train_gt_paths, img_size, gt_size, sigma=sigma, augmentation=True) 
test_ds    = CrowdDataset(test_img_paths, test_gt_paths, img_size, gt_size, sigma=sigma, augmentation=False) 
train_dl   = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
test_dl    = torch.utils.data.DataLoader(test_ds,  batch_size=1, num_workers=num_workers, pin_memory=True, shuffle=False)

###############  Taking a subset for test (Comment the below three lines to use full dataset)
from torch.utils.data import Subset
indices    = torch.randint(low=0, high=20, size=(32,))
train_ds   = Subset(train_ds, indices)
test_ds    = Subset(test_ds, indices)



from src.models.CSRNet import CSRNet
# torch.backends.cudnn.enabled   = True
# torch.backends.cudnn.benchmark = True


''' Define crowd counting model '''
device          = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
model           = CSRNet().to(device)
model_name      = model.__class__.__name__
optimizer       = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
n_epochs        = 100
weights         = None # Define path to model weights if want to load for finetuning
# weights         = f'{os.getcwd()}/checkpoints/{model_name}_{ds_name}.pth' 
checkpoint      = f'{os.getcwd()}/checkpoints/{model_name}.pth' 
val_metric_best = np.Inf
params          = {'model': model,
                   'train_dl':train_dl,
                   'val_dl': test_dl,
                   'optimizer': optimizer,
                   'n_epochs': n_epochs,
                   'weights': weights,
                   'checkpoint': checkpoint,
                   'val_metric_best': val_metric_best,
                   'device': device
                   }
print(f'Models will be saved to >>>> {checkpoint}')

''' Start training '''
from src.trainer import Trainer
trainer = Trainer(params)
history = trainer.fit()
