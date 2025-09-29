import pickle
import random
import re
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import sys
from settings import (BASE_DATA_PATH, VIDEO_DESCRIPTIONS_PATH, SBERT_EMBEDDINGS_PATH, CSV_FEATURE_PATH,
                     NUM_EPOCHS, DEVICE, RANDOM_SEED, APP_TYPES, NUM_APP_TYPES, EMBEDDING_DIM_TRAFFIC,
                     HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT_RATE, APP_EMBEDDING_DIM, ACTION_EMBEDDING_DIM,
                     NUM_ACTION_PROTOTYPES, BATCH_SIZE, LEARNING_RATE, COCO_CAPTION_PATH_BASE,VOCAB_MIN_FREQ,
                     BERT_MODEL_PATH,MAX_TRAFFIC_SEQ_LEN,TEXT_EMBEDDING_DIM,TEACHER_FORCING_RATIO)
from preprocess import set_seed, precompute_sbert_embeddings, Vocabulary, load_bert_embeddings_for_vocab
from dataset import TrafficCaptionDataset, collate_fn
from encoder import TransformerEncoder
from decoder import Attention, Decoder
from model import Seq2Seq
from train import train_epoch
from eval import evaluate_model

sys.path.append(str(COCO_CAPTION_PATH_BASE))
from utils.coco_caption.pycocoevalcap.bleu.bleu import Bleu
from utils.coco_caption.pycocoevalcap.cider.cider import Cider
from utils.coco_caption.pycocoevalcap.meteor.meteor import Meteor
from utils.coco_caption.pycocoevalcap.rouge.rouge import Rouge

if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    print(f"Device: {DEVICE}")
    concrete_app_to_general_type = {
        "qq": "messaging app", "wechat": "messaging app", "messenger": "messaging app", "whatsapp": "messaging app",
        "iqiyi": "video streaming app", "youku": "video streaming app", "tencentvideo": "video streaming app", "youtube": "video streaming app",
        "weibo": "social media app", "redbook": "social media app", "baidutieba": "social media app", "instagram": "social media app",
        "wangyiyun": "music app", "qqmusic": "music app", "kugou": "music app", "spotify": "music app",
        "jd": "shopping app", "taobao": "shopping app", "pdd": "shopping app", "amazon": "shopping app",
    }
    app_type_to_id = {app_name: i for i, app_name in enumerate(APP_TYPES)}
    with open(VIDEO_DESCRIPTIONS_PATH, "rb") as f:
        video_descriptions = pickle.load(f)
    filtered_video_descriptions = {}
    sample_id_to_app_type = {}
    for s_id in video_descriptions:
        match = re.match(r"([a-zA-Z]+)\d+", s_id)
        concrete_app_name = match.group(1).lower() if match else None
        general_app_type_str = concrete_app_to_general_type.get(concrete_app_name)
        descs = video_descriptions.get(s_id, [])
        has_valid_descriptions = any(
            isinstance(d, str) and d.strip() and len(Vocabulary.preprocess_sentence(d)) >= 2 for d in descs)
        if match and general_app_type_str in app_type_to_id and has_valid_descriptions:
            filtered_video_descriptions[s_id] = [d for d in descs if isinstance(d, str) and d.strip() and len(
                Vocabulary.preprocess_sentence(d)) >= 2]
            sample_id_to_app_type[s_id] = app_type_to_id[general_app_type_str]
    print(f"Loaded {len(video_descriptions)} samples, filtered to {len(filtered_video_descriptions)} valid samples")
    print(f"App type distribution: {Counter(sample_id_to_app_type.values())}")
    sbert_model = SentenceTransformer('D:/PythonTest/TrafficCaptioning/all-MiniLM-L6-v2').to(DEVICE)
    if not SBERT_EMBEDDINGS_PATH.exists():
        print("SBERT embeddings not found, precomputing...")
        precompute_sbert_embeddings(filtered_video_descriptions, sbert_model, SBERT_EMBEDDINGS_PATH)
    else:
        print("SBERT embeddings found, skipping precomputation.")
    valid_sample_ids = [s_id for s_id in filtered_video_descriptions if
                        (CSV_FEATURE_PATH / f"{s_id}.csv").exists() and s_id in sample_id_to_app_type]
    print(f"Found {len(valid_sample_ids)} samples with CSV, descriptions, and app types")
    random.shuffle(valid_sample_ids)
    train_split = int(0.8 * len(valid_sample_ids))
    val_split = int(0.1 * len(valid_sample_ids))
    train_ids = valid_sample_ids[:train_split]
    val_ids = valid_sample_ids[train_split:train_split + val_split]
    test_ids = valid_sample_ids[train_split + val_split:]
    if len(valid_sample_ids) > 0:
        if not test_ids and val_ids:
            test_ids = [val_ids.pop()]
        if not val_ids and train_ids:
            val_ids = [train_ids.pop()]
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    train_descriptions = [desc for s_id in train_ids for desc in filtered_video_descriptions.get(s_id, [])]
    vocab = Vocabulary(VOCAB_MIN_FREQ)
    vocab.build_vocabulary(train_descriptions)
    print(f"Vocabulary size: {len(vocab)}")
    bert_embedding_matrix = load_bert_embeddings_for_vocab(vocab, BERT_MODEL_PATH, DEVICE) if Path(
        BERT_MODEL_PATH).exists() else None
    train_dataset = TrafficCaptionDataset(CSV_FEATURE_PATH, filtered_video_descriptions, train_ids, vocab,
                                         sample_id_to_app_type, SBERT_EMBEDDINGS_PATH, MAX_TRAFFIC_SEQ_LEN)
    val_dataset = TrafficCaptionDataset(CSV_FEATURE_PATH, filtered_video_descriptions, val_ids, vocab,
                                       sample_id_to_app_type, SBERT_EMBEDDINGS_PATH, MAX_TRAFFIC_SEQ_LEN)
    test_dataset = TrafficCaptionDataset(CSV_FEATURE_PATH, filtered_video_descriptions, test_ids, vocab,
                                        sample_id_to_app_type, SBERT_EMBEDDINGS_PATH, MAX_TRAFFIC_SEQ_LEN)
    actual_train_batch_size = min(BATCH_SIZE, len(train_dataset)) if len(train_dataset) > 0 else 1
    train_loader = DataLoader(train_dataset, batch_size=actual_train_batch_size, shuffle=True, collate_fn=collate_fn,
                              drop_last=(len(train_dataset) >= BATCH_SIZE))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    encoder = TransformerEncoder(input_dim=EMBEDDING_DIM_TRAFFIC, emb_dim=EMBEDDING_DIM_TRAFFIC, hidden_dim=HIDDEN_DIM,
                                num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT_RATE,
                                num_app_types=NUM_APP_TYPES, app_embedding_dim=APP_EMBEDDING_DIM)
    attention = Attention(HIDDEN_DIM)
    decoder = Decoder(len(vocab), TEXT_EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE, attention,
                      bert_embedding_matrix, action_embedding_dim=ACTION_EMBEDDING_DIM)
    model = Seq2Seq(encoder, decoder, DEVICE, NUM_APP_TYPES, ACTION_EMBEDDING_DIM, NUM_ACTION_PROTOTYPES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    caption_criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    app_type_criterion = nn.CrossEntropyLoss()
    coco_scorers = {"Bleu": Bleu(4), "METEOR": Meteor(), "ROUGE_L": Rouge(), "CIDEr": Cider()}
    best_val_cider = 0.0
    best_val_app_acc = 0.0
    best_val_epoch = -1
    if len(train_dataset) > 0:
        for epoch in range(NUM_EPOCHS):
            train_caption_loss, train_app_loss, train_distance_loss, train_contrastive_loss, train_total_loss = train_epoch(
                model, train_loader, optimizer, caption_criterion, app_type_criterion, clip=1.0,
                device=DEVICE, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            val_scores = evaluate_model(model, val_loader, vocab, DEVICE, coco_scorers, APP_TYPES)
            print(
                f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | Train Caption Loss: {train_caption_loss:.4f} | "
                f"Train App Loss: {train_app_loss:.4f} | Train Distance Loss: {train_distance_loss:.4f} | "
                f"Train Contrastive Loss: {train_contrastive_loss:.4f} | Train Total Loss: {train_total_loss:.4f}")
            for metric, score in val_scores.items():
                print(f"Val {metric}: {score:.4f}")
            current_cider = val_scores.get("CIDEr", 0.0)
            current_app_acc = val_scores.get("App_Accuracy", 0.0)
            scheduler.step(current_cider)
            if current_cider > best_val_cider:
                best_val_cider = current_cider
                best_val_app_acc = current_app_acc
                best_val_epoch = epoch
                torch.save(model.state_dict(), BASE_DATA_PATH / "best_traffic_caption_model_multitask.pth")
                print(f"Saved best model, CIDEr: {best_val_cider:.4f}, App Accuracy: {best_val_app_acc:.4f}")
    if test_loader and len(test_dataset) > 0:
        print("\nTest evaluation...")
        best_model_path = BASE_DATA_PATH / "best_traffic_caption_model_multitask.pth"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
            test_scores = evaluate_model(model, test_loader, vocab, DEVICE, coco_scorers, APP_TYPES)
            for metric, score in test_scores.items():
                print(f"Test {metric}: {score:.4f}")
        else:
            print("Warning: Best model not found, skipping test evaluation")
    print("\nTest sample generations:")
    model.eval()
    if len(test_ids) > 0:
        sample_ids_to_show = random.sample(test_ids, min(100, len(test_ids)))
        for sample_id in sample_ids_to_show:
            found_sample = None
            for tf_item, _, sid_item, app_id_item, _ in test_dataset.data_pairs:
                if str(sid_item) == str(sample_id):
                    found_sample = (tf_item, app_id_item)
                    break
            if found_sample:
                tf, true_app_id = found_sample
                true_app_id_val = true_app_id.item() if isinstance(true_app_id, torch.Tensor) else true_app_id
                with torch.no_grad():
                    app_type_pred = model.predict_app_type(tf.unsqueeze(0).to(DEVICE))
                    predicted_app_id = app_type_pred.argmax(1).item()
                    generated_text, _ = model.generate_caption(tf.unsqueeze(0).to(DEVICE), vocab, predicted_app_id)
                    clean_text = generated_text.replace("<sos>", "").replace("<eos>", "").strip()
                print(f"\nSample ID: {sample_id}")
                print(f"Generated: {clean_text}")
                print(f"True App Type: {APP_TYPES[true_app_id_val]} (ID: {true_app_id_val})")
                print(f"Predicted App Type: {APP_TYPES[predicted_app_id]} (ID: {predicted_app_id})")
                print("Groundtruths:")
                descs = filtered_video_descriptions.get(sample_id, [])
                for gt in descs:
                    print(f"- {gt}")
            else:
                print(f"\nSample ID: {sample_id} not found in test data, skipping")
    else:
        print("No test samples available for generation")