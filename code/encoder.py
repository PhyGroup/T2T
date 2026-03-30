import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from sentence_transformers import SentenceTransformer
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, num_heads, dropout, num_app_types, app_embedding_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, emb_dim) if input_dim != emb_dim else None
        self.embedding = nn.Linear(emb_dim, hidden_dim)
        sbert_model = SentenceTransformer('D:/PythonTest/TrafficCaptioning/all-MiniLM-L6-v2').to(DEVICE)
        app_type_texts = APP_TYPES
        sbert_app_embeddings = sbert_model.encode(app_type_texts, convert_to_tensor=True).to(DEVICE)
        app_embedding_proj = nn.Linear(384, app_embedding_dim).to(DEVICE)
        sbert_app_embeddings = app_embedding_proj(sbert_app_embeddings)
        self.app_embedding_layer = nn.Embedding(num_app_types, app_embedding_dim)
        self.app_embedding_layer.weight.data = sbert_app_embeddings
        self.app_embedding_layer.weight.requires_grad = False
        self.film_scale_fc = nn.Linear(app_embedding_dim, hidden_dim)
        self.film_bias_fc = nn.Linear(app_embedding_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src, app_type_ids=None):
        embedded = torch.relu(self.input_fc(src)) if self.input_fc else src
        embedded = self.embedding(embedded)
        if app_type_ids is not None:
            app_emb = self.app_embedding_layer(app_type_ids)
            scale = self.film_scale_fc(app_emb).unsqueeze(1)
            bias = self.film_bias_fc(app_emb).unsqueeze(1)
            embedded = embedded * scale + bias
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded)
        pooled_output = output.mean(dim=1)
        hidden = self.fc_hidden(pooled_output).unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = self.fc_cell(pooled_output).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return output, hidden, cell, pooled_output

class ActionPrototypeLayer(nn.Module):
    def __init__(self, num_app_types, num_prototypes, hidden_dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.zeros(num_app_types, num_prototypes, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.temperature = CONTRASTIVE_TEMPERATURE

    def forward(self, pooled_output, app_type_ids):
        batch_size = pooled_output.size(0)
        if app_type_ids.max() >= self.prototypes.size(0) or app_type_ids.min() < 0:
            raise ValueError(f"Invalid app_type_ids: {app_type_ids.tolist()}, expected range [0, {self.prototypes.size(0) - 1}]")
        selected_prototypes = self.prototypes[app_type_ids]
        pooled_output_expanded = pooled_output.unsqueeze(1)
        attn_scores = torch.bmm(pooled_output_expanded, selected_prototypes.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores.squeeze(1) / (HIDDEN_DIM ** 0.5), dim=-1)
        action_embedding = torch.bmm(attn_weights.unsqueeze(1), selected_prototypes).squeeze(1)
        action_embedding = self.layer_norm(action_embedding + pooled_output)
        contrastive_loss = self.compute_contrastive_loss(pooled_output, app_type_ids)
        return self.dropout(action_embedding), contrastive_loss

    def compute_contrastive_loss(self, pooled_output, app_type_ids):
        batch_size = pooled_output.size(0)
        contrastive_loss = torch.tensor(0.0, device=pooled_output.device)
        norm_pooled = F.normalize(pooled_output, dim=-1)
        norm_prototypes = F.normalize(self.prototypes, dim=-1)

        for i in range(batch_size):
            app_id = app_type_ids[i]
            pos_prototypes = norm_prototypes[app_id]
            pos_sim = torch.matmul(norm_pooled[i:i + 1], pos_prototypes.transpose(0, 1))
            pos_sim = pos_sim / self.temperature
            neg_indices = [j for j in range(self.prototypes.size(0)) if j != app_id]
            neg_prototypes = norm_prototypes[neg_indices]
            neg_prototypes = neg_prototypes.view(-1, self.prototypes.size(-1))
            neg_sim = torch.matmul(norm_pooled[i:i + 1], neg_prototypes.transpose(0, 1))
            neg_sim = neg_sim / self.temperature
            all_sim = torch.cat([pos_sim, neg_sim], dim=1)
            exp_sim = torch.exp(all_sim)
            pos_sum = exp_sim[:, :pos_sim.size(1)].sum(dim=1)
            total_sum = exp_sim.sum(dim=1)
            loss = -torch.log(pos_sum / (total_sum + 1e-10))
            contrastive_loss += loss.mean()

        contrastive_loss = contrastive_loss / batch_size
        contrastive_loss = torch.clamp(contrastive_loss, min=0.0)
        return contrastive_loss

class AppContextualizedActionInterpreter(nn.Module):
    def __init__(self, hidden_dim, action_embedding_dim, num_app_types, num_prototypes, dropout_rate=0.5):
        super().__init__()
        self.prototype_layer = ActionPrototypeLayer(num_app_types, num_prototypes, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_embedding_dim)
        self.layer_norm = nn.LayerNorm(action_embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, pooled_output, app_type_ids):
        action_emb_proto, contrastive_loss = self.prototype_layer(pooled_output, app_type_ids)
        action_emb = self.alpha * action_emb_proto + (1 - self.alpha) * pooled_output
        action_emb = self.fc(action_emb)
        action_emb = self.layer_norm(action_emb)
        return self.dropout(action_emb), contrastive_loss

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, num_app_types, action_embedding_dim, num_prototypes, text_embedding_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.num_app_types = num_app_types
        self.action_interpreter = AppContextualizedActionInterpreter(HIDDEN_DIM, action_embedding_dim, num_app_types, num_prototypes, DROPOUT_RATE)
        self.app_classifier = nn.Linear(HIDDEN_DIM, num_app_types)
        self.sbert_caption_projector = nn.Linear(384, action_embedding_dim)
        self.generated_caption_projector = nn.Linear(text_embedding_dim, action_embedding_dim)

    def forward(self, src_features, trg_tokens, trg_app_types, caption_embs, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg_tokens.shape
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(self.device)
        caption_loss = torch.tensor(0.0, device=self.device)
        app_loss = torch.tensor(0.0, device=self.device)
        distance_loss = torch.tensor(0.0, device=self.device)
        contrastive_loss = torch.tensor(0.0, device=self.device)

        try:
            encoder_outputs_unconditioned, hidden_unconditioned, cell_unconditioned, pooled_output_unconditioned = self.encoder(src_features, None)
            app_type_prediction = self.app_classifier(pooled_output_unconditioned)
            app_loss = nn.CrossEntropyLoss()(app_type_prediction, trg_app_types)
            predicted_app_ids_for_conditioning = app_type_prediction.argmax(dim=1)
            encoder_outputs, hidden, cell, pooled_output = self.encoder(src_features, predicted_app_ids_for_conditioning)
            action_embedding, cont_loss_cond = self.action_interpreter(pooled_output, predicted_app_ids_for_conditioning)
            contrastive_loss = cont_loss_cond

            input_token = trg_tokens[:, 0]
            generated_embeddings_list = []
            for t in range(1, trg_len):
                output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, action_embedding)
                outputs[:, t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                input_token = trg_tokens[:, t] if teacher_force else output.argmax(1)
                current_generated_emb = self.decoder.embedding(input_token)
                generated_embeddings_list.append(current_generated_emb)

            output = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            trg = trg_tokens[:, 1:].reshape(-1)
            caption_loss = nn.CrossEntropyLoss(ignore_index=self.decoder.embedding.weight.shape[0] - 1)(output, trg)

            if generated_embeddings_list:
                generated_embeddings_stacked = torch.stack(generated_embeddings_list, dim=1)
                mask_tokens = trg_tokens[:, 1:]
                mask = (mask_tokens != self.decoder.embedding.weight.shape[0] - 1).float().unsqueeze(-1)
                masked_generated_embeddings = generated_embeddings_stacked * mask
                sum_of_mask = mask.sum(dim=1)
                generated_sentence_embs = masked_generated_embeddings.sum(dim=1) / (sum_of_mask + 1e-10)
                generated_emb_normalized = F.normalize(self.generated_caption_projector(generated_sentence_embs), dim=-1)
                projected_caption_embs = self.sbert_caption_projector(caption_embs.float())
                caption_embs_normalized = F.normalize(projected_caption_embs, dim=-1)
                distance_loss = 1 - torch.mean(torch.sum(generated_emb_normalized * caption_embs_normalized, dim=-1))

            total_loss = caption_loss + APP_LOSS_WEIGHT * app_loss + DISTANCE_WEIGHT * distance_loss + CONTRASTIVE_WEIGHT * contrastive_loss

        except Exception as e:
            print(f"Error in Seq2Seq forward pass: {e}")
            outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim, device=self.device)
            app_type_prediction = torch.zeros(batch_size, self.num_app_types, device=self.device)
            caption_loss = torch.tensor(0.0, device=self.device)
            app_loss = torch.tensor(0.0, device=self.device)
            distance_loss = torch.tensor(0.0, device=self.device)
            contrastive_loss = torch.tensor(0.0, device=self.device)
            total_loss = torch.tensor(0.0, device=self.device)

        return outputs, app_type_prediction, caption_loss, app_loss, distance_loss, contrastive_loss, total_loss