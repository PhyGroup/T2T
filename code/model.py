import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import traceback
from settings import (DEVICE, NUM_APP_TYPES, ACTION_EMBEDDING_DIM, NUM_ACTION_PROTOTYPES, MAX_CAPTION_LEN,
                     BEAM_WIDTH, APP_LOSS_WEIGHT, DISTANCE_WEIGHT, CONTRASTIVE_WEIGHT, TEXT_EMBEDDING_DIM)
from encoder import AppContextualizedActionInterpreter

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, num_app_types, action_embedding_dim, num_prototypes):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.num_app_types = num_app_types
        self.action_interpreter = AppContextualizedActionInterpreter(
            hidden_dim=512, action_embedding_dim=action_embedding_dim, num_app_types=num_app_types,
            num_prototypes=num_prototypes)
        self.app_classifier = nn.Linear(512, num_app_types)
        self.sbert_caption_projector = nn.Linear(384, action_embedding_dim)
        self.generated_caption_projector = nn.Linear(TEXT_EMBEDDING_DIM, action_embedding_dim)

    def forward(self, src_features, trg_tokens, trg_app_types, caption_embs, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg_tokens.shape
        outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(self.device)
        caption_loss = torch.tensor(0.0, device=self.device)
        app_loss = torch.tensor(0.0, device=self.device)
        distance_loss = torch.tensor(0.0, device=self.device)
        contrastive_loss = torch.tensor(0.0, device=self.device)
        try:
            encoder_outputs_unconditioned, hidden_unconditioned, cell_unconditioned, pooled_output_unconditioned = self.encoder(
                src_features, None)
            app_type_prediction = self.app_classifier(pooled_output_unconditioned)
            app_loss = nn.CrossEntropyLoss()(app_type_prediction, trg_app_types)
            predicted_app_ids_for_conditioning = app_type_prediction.argmax(dim=1)
            encoder_outputs, hidden, cell, pooled_output = self.encoder(src_features,
                                                                       predicted_app_ids_for_conditioning)
            action_embedding, cont_loss_cond = self.action_interpreter(pooled_output,
                                                                      predicted_app_ids_for_conditioning)
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
                generated_emb_normalized = F.normalize(self.generated_caption_projector(generated_sentence_embs),
                                                      dim=-1)
                projected_caption_embs = self.sbert_caption_projector(caption_embs.float())
                caption_embs_normalized = F.normalize(projected_caption_embs, dim=-1)
                distance_loss = 1 - torch.mean(torch.sum(generated_emb_normalized * caption_embs_normalized, dim=-1))
            total_loss = caption_loss + APP_LOSS_WEIGHT * app_loss + DISTANCE_WEIGHT * distance_loss + CONTRASTIVE_WEIGHT * contrastive_loss
        except Exception as e:
            print(f"Error in Seq2Seq forward pass: {e}, traceback: {traceback.format_exc()}")
            outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim, device=self.device)
            app_type_prediction = torch.zeros(batch_size, self.num_app_types, device=self.device)
            caption_loss = torch.tensor(0.0, device=self.device)
            app_loss = torch.tensor(0.0, device=self.device)
            distance_loss = torch.tensor(0.0, device=self.device)
            contrastive_loss = torch.tensor(0.0, device=self.device)
            total_loss = torch.tensor(0.0, device=self.device)
        return outputs, app_type_prediction, caption_loss, app_loss, distance_loss, contrastive_loss, total_loss

    def predict_app_type(self, src_features):
        self.eval()
        with torch.no_grad():
            _, _, _, pooled_output = self.encoder(src_features, None)
            app_type_prediction = self.app_classifier(pooled_output)
            return app_type_prediction

    def generate_caption(self, src_features, vocab, app_type_id, max_len=MAX_CAPTION_LEN, beam_width=BEAM_WIDTH):
        self.eval()
        with torch.no_grad():
            app_type_tensor = torch.tensor([app_type_id]).long().to(self.device)
            encoder_outputs, hidden, cell, pooled_output = self.encoder(src_features, app_type_tensor)
            action_embedding, _ = self.action_interpreter(pooled_output, app_type_tensor)
            beams = [([vocab.stoi['<sos>']], 0.0, hidden, cell)]
            completed_beams = []
            for _ in range(max_len):
                new_beams = []
                for tokens, score, h, c in beams:
                    input_token = torch.tensor([tokens[-1]]).to(self.device)
                    output, h, c, _ = self.decoder(input_token, h, c, encoder_outputs, action_embedding)
                    probs = torch.softmax(output, dim=-1)
                    top_probs, top_indices = probs.topk(beam_width, dim=-1)
                    top_probs = top_probs.squeeze(0)
                    top_indices = top_indices.squeeze(0)
                    for prob, idx in zip(top_probs, top_indices):
                        new_score = score + math.log(prob.item() + 1e-10)
                        new_tokens = tokens + [idx.item()]
                        new_h, new_c = h.clone(), c.clone()
                        if idx.item() == vocab.stoi['<eos>'] and len(new_tokens) >= 5:
                            completed_beams.append((new_tokens, new_score / len(new_tokens), new_h, new_c))
                        else:
                            new_beams.append((new_tokens, new_score, new_h, new_c))
                beams = sorted(new_beams, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
                if not beams and completed_beams:
                    break
            if completed_beams:
                best_tokens, best_score, _, _ = max(completed_beams, key=lambda x: x[1])
            else:
                best_tokens, best_score, _, _ = max(beams, key=lambda x: x[1] / len(x[0]))
            caption_words = [vocab.itos[t] for t in best_tokens if t != vocab.stoi['<pad>']]
            return " ".join(caption_words), []