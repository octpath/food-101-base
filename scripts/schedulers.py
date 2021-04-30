# coding: utf-8
'''
Scheduler
* num_epochs
* get_next_lr(self, next_epoch, curr_loss, curr_score)
'''

import numpy as np
import math


class CosineScheduler:
    def __init__(self, num_epochs, init_lr, min_lr):
        self.num_epochs = num_epochs
        self.curr_lr = init_lr
        self.init_lr = init_lr
        self.min_lr = min_lr
    
    def get_next_lr(self, next_epoch, curr_loss, curr_score):
        self.curr_lr = (self.init_lr - self.min_lr) * 0.5 * (np.cos(np.pi * next_epoch / self.num_epochs) + 1.) + self.min_lr
        return self.curr_lr

    
class ReduceOnPlateauScheduler:
    def __init__(self, num_epochs, init_lr, min_lr, decay, max_patience):
        self.num_epochs = num_epochs
        self.curr_lr = init_lr
        self.min_lr = min_lr
        self.decay = decay
        self.max_patience = max_patience
        self.patience = 0
        self.best_loss = 10000.
        self.best_score = 0.
        
    def get_next_lr(self, next_epoch, curr_loss, curr_score):
        self.patience += 1
        
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss
            self.patience = 0
            
        if curr_score > self.best_score:
            self.best_score = curr_score
            self.patience = 0
        
        if self.patience > self.max_patience:
            self.patience = 0
            self.curr_lr = max(self.curr_lr * self.decay, self.min_lr)
        
        return self.curr_lr