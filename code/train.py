import torch
from tqdm import tqdm
from config import *

def train_epoch(model, dataloader, optimizer, device, teacher_forcing_ratio):
    model.train()
    epoch_caption_loss = epoch_app_loss = epoch_distance_loss = epoch_contrastive_loss = epoch_total_loss = 0
    batch_count = 0
    for src_features, trg_tokens, _, trg_app_types, caption_embs in tqdm(dataloader, desc="Training", leave=False):
        if src_features is None or trg_tokens is None or caption_embs is None:
            print("Warning: Invalid batch (None values), skipping.")
            continue
        src_features = src_features.to(device)
        trg_tokens = trg_tokens.to(device)
        trg_app_types = trg_app_types.to(device)
        caption_embs = caption_embs.to(device)
        optimizer.zero_grad()
        try:
            outputs, app_type_pred, caption_loss, app_loss, distance_loss, contrastive_loss, total_loss = model(
                src_features, trg_tokens, trg_app_types, caption_embs, teacher_forcing_ratio)
            if outputs is None or outputs.shape[0] == 0:
                print("Warning: Model outputs are invalid or empty, skipping batch.")
                continue
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Warning: Loss is NaN or Inf, skipping batch")
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_caption_loss += caption_loss.item()
            epoch_app_loss += app_loss.item()
            epoch_distance_loss += distance_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            epoch_total_loss += total_loss.item()
            batch_count += 1
        except Exception as e:
            print(f"Error during batch processing: {e}")
            continue
    if batch_count == 0:
        print("Warning: No valid batches processed in epoch")
        return 0, 0, 0, 0, 0
    return (epoch_caption_loss / batch_count, epoch_app_loss / batch_count,
            epoch_distance_loss / batch_count, epoch_contrastive_loss / batch_count,
            epoch_total_loss / batch_count)