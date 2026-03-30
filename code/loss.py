import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

def caption_loss_fn(output, target, ignore_index):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)(output, target)

def app_loss_fn(pred, target):
    return nn.CrossEntropyLoss()(pred, target)

def distance_loss_fn(generated_embs, caption_embs):
    if generated_embs.shape[0] == 0:
        return torch.tensor(0.0, device=generated_embs.device)
    generated_norm = F.normalize(generated_embs, dim=-1)
    caption_norm = F.normalize(caption_embs, dim=-1)
    return 1 - torch.mean(torch.sum(generated_norm * caption_norm, dim=-1))

def contrastive_loss_fn(prototype_layer, pooled_output, app_type_ids):
    return prototype_layer.compute_contrastive_loss(pooled_output, app_type_ids)

def total_loss_fn(caption_loss, app_loss, distance_loss, contrastive_loss):
    return caption_loss + APP_LOSS_WEIGHT * app_loss + DISTANCE_WEIGHT * distance_loss + CONTRASTIVE_WEIGHT * contrastive_loss