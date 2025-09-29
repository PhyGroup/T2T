import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from settings import (DEVICE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT_RATE, NUM_APP_TYPES,
                     APP_EMBEDDING_DIM, APP_TYPES, CONTRASTIVE_TEMPERATURE)

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
            raise ValueError(
                f"Invalid app_type_ids: {app_type_ids.tolist()}, expected range [0, {self.prototypes.size(0) - 1}]")
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
            pos_sim = torch.matmul(norm_pooled[i:i+1], pos_prototypes.transpose(0, 1)) / self.temperature
            neg_indices = [j for j in range(self.prototypes.size(0)) if j != app_id]
            neg_prototypes = norm_prototypes[neg_indices].view(-1, self.prototypes.size(-1))
            neg_sim = torch.matmul(norm_pooled[i:i+1], neg_prototypes.transpose(0, 1)) / self.temperature
            all_sim = torch.cat([pos_sim, neg_sim], dim=1)
            exp_sim = torch.exp(all_sim)
            pos_sum = exp_sim[:, :pos_sim.size(1)].sum(dim=1)
            total_sum = exp_sim.sum(dim=1)
            loss = -torch.log(pos_sum / (total_sum + 1e-10))
            contrastive_loss += loss.mean()
        contrastive_loss = contrastive_loss / batch_size
        contrastive_loss = torch.clamp(contrastive_loss, min=0.0)
        return contrastive_loss

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, num_heads, dropout, num_app_types,
                 app_embedding_dim):
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                                                  dropout=dropout, batch_first=True)
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