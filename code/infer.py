import torch
import math
import random
from config import *

def predict_app_type(model, src_features):
    model.eval()
    with torch.no_grad():
        _, _, _, pooled_output = model.encoder(src_features, None)
        app_type_prediction = model.app_classifier(pooled_output)
        return app_type_prediction

def generate_caption(model, src_features, vocab, app_type_id, max_len=MAX_CAPTION_LEN, beam_width=BEAM_WIDTH):
    model.eval()
    with torch.no_grad():
        app_type_tensor = torch.tensor([app_type_id]).long().to(DEVICE)
        encoder_outputs, hidden, cell, pooled_output = model.encoder(src_features, app_type_tensor)
        action_embedding, _ = model.action_interpreter(pooled_output, app_type_tensor)
        beams = [([vocab.stoi['<sos>']], 0.0, hidden, cell)]
        completed_beams = []
        for _ in range(max_len):
            new_beams = []
            for tokens, score, h, c in beams:
                input_token = torch.tensor([tokens[-1]]).to(DEVICE)
                output, h, c, _ = model.decoder(input_token, h, c, encoder_outputs, action_embedding)
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