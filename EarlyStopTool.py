#Early Stopping for CNN
import numpy as np
import torch

class early_stopping:
    def __init__(self, patience=7, verbose=False, path='checkpoint.pth', trace_func = print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'        
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.path = path
        self.vloss_mem = []
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        if val_loss.item() <= self.best_score:
            self.best_score = val_loss
            self.counter = 0 
            self.vloss_mem.append(val_loss.item())
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose: 
                self.trace_func(f'Early stopping ---> {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
    
    def best_validation_score(self):
        return min(self.vloss_mem)
