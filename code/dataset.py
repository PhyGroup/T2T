import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from settings import CSV_FEATURE_PATH, EMBEDDING_DIM_TRAFFIC, MAX_TRAFFIC_SEQ_LEN, SBERT_EMBEDDINGS_PATH, DEVICE


class TrafficCaptionDataset(Dataset):
    def __init__(self, csv_path, descriptions_dict, sample_ids, vocab, app_type_map, sbert_embeddings_path,
                 max_traffic_seq_len=50):
        self.csv_path = csv_path
        self.descriptions_dict = descriptions_dict
        self.sample_ids = sample_ids
        self.vocab = vocab
        self.app_type_map = app_type_map
        self.max_traffic_seq_len = max_traffic_seq_len
        try:
            with open(sbert_embeddings_path, 'rb') as f:
                self.sbert_embeddings = pickle.load(f)
            print(f"Successfully loaded SBERT embeddings from {sbert_embeddings_path}")
        except FileNotFoundError:
            print(f"Error: SBERT embeddings file not found at {sbert_embeddings_path}. Please precompute them.")
            self.sbert_embeddings = {}
        except Exception as e:
            print(f"Error loading SBERT embeddings: {e}")
            self.sbert_embeddings = {}
        self.data_pairs = self._load_data_pairs()

    def _load_data_pairs(self):
        data_pairs = []
        for sample_id in tqdm(self.sample_ids, desc="Loading dataset samples"):
            if sample_id not in self.sbert_embeddings or sample_id not in self.app_type_map:
                continue
            csv_file = self.csv_path / f"{sample_id}.csv"
            if not csv_file.exists():
                continue
            try:
                df = pd.read_csv(csv_file).fillna(0)
                numeric_df = df.select_dtypes(include=np.number)
                if numeric_df.empty:
                    continue
                current_dim = numeric_df.shape[1]
                if current_dim != EMBEDDING_DIM_TRAFFIC:
                    if current_dim > EMBEDDING_DIM_TRAFFIC:
                        numeric_df = numeric_df.iloc[:, :EMBEDDING_DIM_TRAFFIC]
                    else:
                        padding = pd.DataFrame(0, index=numeric_df.index,
                                              columns=[f'pad_{i}' for i in range(EMBEDDING_DIM_TRAFFIC - current_dim)])
                        numeric_df = pd.concat([numeric_df, padding], axis=1)
                traffic_features = torch.FloatTensor(numeric_df.values)
                if traffic_features.shape[0] > self.max_traffic_seq_len:
                    traffic_features = traffic_features[:self.max_traffic_seq_len, :]
                descs = self.descriptions_dict.get(sample_id, [])
                valid_descs = [d for d in descs if
                               isinstance(d, str) and d.strip() and len(self.vocab.preprocess_sentence(d)) >= 2]
                if not valid_descs:
                    continue
                sample_caption_embeddings = self.sbert_embeddings.get(sample_id)
                if sample_caption_embeddings is None or sample_caption_embeddings.shape[0] != len(valid_descs):
                    print(
                        f"Warning: Mismatch in SBERT embeddings for {sample_id}. Expected {len(valid_descs)} but got {sample_caption_embeddings.shape[0] if sample_caption_embeddings is not None else 'None'}. Skipping.")
                    continue
                for desc, caption_emb in zip(valid_descs, sample_caption_embeddings):
                    data_pairs.append(
                        (traffic_features, desc, sample_id, self.app_type_map[sample_id], caption_emb.to(DEVICE)))
            except Exception as e:
                print(f"Warning: Failed to process {csv_file} ({e}), skipping sample {sample_id}")
                continue
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        traffic_features, description, sample_id, app_type_id, caption_emb = self.data_pairs[index]
        numericalized_caption = [self.vocab.stoi["<sos>"]] + self.vocab.numericalize(description) + [
            self.vocab.stoi["<eos>"]]
        caption_tensor = torch.LongTensor(numericalized_caption)
        return traffic_features, caption_tensor, sample_id, app_type_id, caption_emb

def collate_fn(batch):
    batch = [(f, c, s, a, ce) for f, c, s, a, ce in batch if f is not None and c is not None and ce is not None]
    if not batch:
        print("Warning: Empty batch after filtering, returning None")
        return None, None, [], None, None
    traffic_features_list, captions_list, sample_ids_list, app_type_ids_list, caption_embs = [], [], [], [], []
    for features, caption, s_id, app_id, ce in batch:
        if features.numel() == 0 or caption.numel() == 0 or ce.numel() == 0:
            print(f"Warning: Invalid sample {s_id} with empty tensor, skipping.")
            continue
        traffic_features_list.append(features)
        captions_list.append(caption)
        sample_ids_list.append(s_id)
        app_type_ids_list.append(app_id)
        caption_embs.append(ce)
    if not traffic_features_list:
        print("Warning: No valid samples in batch after detailed checks, returning None")
        return None, None, [], None, None
    padded_traffic = pad_sequence(traffic_features_list, batch_first=True, padding_value=0.0)
    padded_captions = pad_sequence(captions_list, batch_first=True, padding_value=0)
    app_type_tensors = torch.LongTensor(app_type_ids_list)
    caption_embs_tensor = torch.stack(caption_embs)

    return padded_traffic, padded_captions, sample_ids_list, app_type_tensors, caption_embs_tensor
