from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np


def masked_weighted_cross_entropy_loss(weight, logits, target, mask):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        
    Returns:
        loss: An average loss value.
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss_fn = nn.NLLLoss(weight = weight, reduction = 'none')
       
    losses = loss_fn(log_probs, target)
    
    mask = mask.reshape(-1,1).squeeze(1)
    losses = losses * mask.float()
    
    if mask.sum() == 0:
        loss = mask.sum().float().clone().detach().requires_grad_(True)
        return loss
    if weight is not None:
        loss = losses.sum() / ((weight[target] * mask).sum())
    else:
        loss = losses.sum()/mask.sum()
    
    return loss



def masked_mse_loss(input, target,mask):
    #print(target)
    loss_fn = nn.MSELoss(reduction = 'none')
    losses = loss_fn(input, target)
    
    #print(input.shape, target.shape, losses.shape)
    mask = mask.reshape(-1,1)
    #print('losses',losses.sum().item())
    #print(mask.shape)
    #print(losses.shape)
    losses = losses*mask
    #print('after',losses.sum().item())
    
    if mask.sum() == 0:
        loss = mask.sum().float().clone().detach().requires_grad_(True)
        return loss
    #print(losses.sum())
    loss = losses.sum()/mask.sum()
   # print(loss.item())
    return loss





def cross_entropy_one_hot(input, target, weight):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss(weight = weight)(input, labels)


