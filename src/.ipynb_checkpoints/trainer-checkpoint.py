




import numpy as np
import os, glob, re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tqdm.notebook import trange, tqdm

class Trainer:
    def __init__(self, params): # model, train_data, val_data, loss_fn, n_epochs, optimizer, lr_scheduler, checkpoint, best_val_metric=np.inf, device='cpu'
        self.model           = params['model']
        self.train_dl        = params['train_dl']
        self.val_dl          = params['val_dl']
        self.optimizer       = params['optimizer']
        self.n_epochs        = params['n_epochs']
        self.weights         = params['weights']
        self.checkpoint      = params['checkpoint']
        self.val_metric_best = params['val_metric_best']
        self.device          = params['device']
        

        ##  Placeholders
        self.train_losses   = []
        self.val_losses     = []
        self.train_metrics  = []
        self.val_metrics    = []
        
        # Define the loss functions
        self.loss_mse  = nn.MSELoss(reduction='mean').to(self.device)

        
    def train_epoch(self):
        '''  Train a single epoch '''
        self.model.train()
        train_loss_epoch     = 0.0
        train_metric_epoch   = 0.0
        for data in self.train_dl:
            img, gt  = data[0].to(self.device), data[1].to(self.device) # Read a single batch
            self.optimizer.zero_grad()  # sets gradients to zeros
            et   = self.model(img) # predict the outputs (inputs is batch of images)
            batch_loss  = self.loss_mse(gt, et) # calculate loss (scalar value: mean or sum of losses for all images in the batch)
            train_loss_epoch += batch_loss.item() # add batch_loss to find cumulative epoch_loss which will be averaged later
            train_metric_epoch  += abs(gt.sum() - et.sum())         
            batch_loss.backward()  # Backpropagation
            self.optimizer.step()
        train_loss_epoch   = train_loss_epoch/len(self.train_dl.dataset) # average over the number of images to get mean error for thw whole epoch
        train_metric_epoch = train_metric_epoch/len(self.train_dl.dataset) # average over the number of images to get mean error for thw whole epoch
        self.train_losses.append(train_loss_epoch)
        self.train_metrics.append(train_metric_epoch)
        return train_loss_epoch, train_metric_epoch

        
        
    def evaluate(self):
        '''  Validate a single epoch '''
        self.model.eval()
        val_loss_epoch   = 0.0
        val_metric_epoch = 0.0
        val_mse_epoch    = 0.0
        with torch.no_grad():
            for data in self.val_dl:
                img, gt = data[0].to(self.device), data[1].to(self.device) # Read a single batch
                et      = self.model(img) # predict the output
                batch_loss  = self.loss_mse(gt, et) # calculate loss (scalar value: mean or sum of losses for all images in the batch)
                val_loss_epoch += batch_loss.item() # add batch_loss to find cumulative epoch_loss which will be averaged later
                val_metric_epoch  += abs(gt.sum() - et.sum())         
                # val_mse_epoch  += (gt.sum() - et.sum())**2
        val_loss_epoch     = val_loss_epoch/len(self.val_dl.dataset) # find average over all batches
        val_metric_epoch   = val_metric_epoch/len(self.val_dl.dataset) # find average over all batches
        # Divide epoch loss by size of dataset
        # epoch_loss /= len(self.val_data)
        # epoch_metric /= len(self.val_data)
        self.val_losses.append(val_loss_epoch)
        self.val_metrics.append(val_metric_epoch)
        return val_loss_epoch, val_metric_epoch
    
    def save_checkpoint(self):
        if self.checkpoint is not None:
        torch.save(self.model.state_dict(), self.checkpoint)
        print(f"Checkpoint saved at: {self.checkpoint}")
        
    def load_checkpoint(self):
        if os.path.isfile(self.weights):
            self.model.load_state_dict(torch.load(self.weights))
            val_loss_epoch, val_metric_epoch = self.evaluate()
            self.val_metric_best = val_metric_epoch
            print(f"Checkpoint loaded successfully: MAE {self.val_metric_best}")
        else:
            self.val_metric_best = np.Inf
            print(f"No checkpoint found. Starting training from scratch.")
        return self.val_metric_best
            

    def fit(self):
        ''' Train the model'''
        # Load checkpoint if exists
        if self.weights is not None:
            print(f"Looking for existing checkpoints in: {self.weights}")
            self.val_metric_best = self.load_checkpoint()
        
        print('\n')    
        progress_bar = tqdm(total=self.n_epochs, position=0, leave=False)
        for epoch in range(self.n_epochs):
            train_loss_epoch, train_metric_epoch = self.train_epoch()
            val_loss_epoch, val_metric_epoch     = self.evaluate()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch: [{epoch}/{self.n_epochs}]  "
                                         f"Train Loss: [{self.train_losses[-1]:.4f}]  "
                                         f"Val Metric: [{self.val_metrics[-1]:.4f}]  ")
            # Saving checkpoints
            if self.checkpoint is not None:
                if val_metric_epoch <= self.val_metric_best:
                    # Save checkpoint
                    self.save_checkpoint()
                    # Update the best_val_metric
                    self.val_metric_best = val_metric_epoch
        print("Training completed.")
        history = {'train_losses': self.train_losses, 'val_losses': self.val_losses, 'train_metrics': self.train_metrics, 'val_metrics': self.val_metrics}
        return history
        



