import random
import re
from collections import Counter
import numpy as np
import torch
import pickle
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from settings import MAX_CAPTION_LEN, VOCAB_MIN_FREQ, SBERT_EMBEDDINGS_PATH, BERT_MODEL_PATH, DEVICE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def precompute_sbert_embeddings(descriptions_dict, sbert_model, output_path):
    embeddings = {}
    for s_id, descs in tqdm(descriptions_dict.items(), desc="Precomputing SBERT embeddings"):
        valid_descs = [d for d in descs if isinstance(d, str) and d.strip()]
        if valid_descs:
            embeddings[s_id] = sbert_model.encode(valid_descs, convert_to_tensor=True).cpu()
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"SBERT embeddings precomputed and saved to {output_path}")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def preprocess_sentence(sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"[^\w\s]", "", sentence)
        return sentence.split()[:MAX_CAPTION_LEN]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)
        for sentence in sentence_list:
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            for word in self.preprocess_sentence(sentence):
                frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.preprocess_sentence(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]

def load_bert_embeddings_for_vocab(vocab, bert_model_path, device):
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    bert_model = BertModel.from_pretrained(bert_model_path).to(device)
    bert_model.eval()
    embedding_dim = bert_model.config.hidden_size
    embedding_matrix = torch.zeros(len(vocab), embedding_dim)
    with torch.no_grad():
        for idx, word in vocab.itos.items():
            if word == "<pad>":
                embedding_matrix[idx] = torch.zeros(embedding_dim)
            elif word in ["<sos>", "<eos>", "<unk>"]:
                token = "<unk>" if word == "<unk>" else word.strip('<>')
                inputs = tokenizer(token, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = bert_model(**inputs)
                embedding_matrix[idx] = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
            else:
                inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = bert_model(**inputs)
                embedding_matrix[idx] = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
    embedding_matrix = (embedding_matrix - embedding_matrix.mean()) / (embedding_matrix.std() + 1e-8)
    return embedding_matrix