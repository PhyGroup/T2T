from tqdm import tqdm
import torch
from preprocess import Vocabulary

def evaluate_model(model, dataloader, vocab, device, coco_scorers, app_type_itos):
    model.eval()
    references_corpus = {}
    candidates_corpus = {}
    temp_gts = {
        s_id: [" ".join(Vocabulary.preprocess_sentence(d)) for d in dataloader.dataset.descriptions_dict.get(s_id, [])
               if isinstance(d, str) and d.strip()] for s_id in dataloader.dataset.sample_ids}
    correct_app_predictions = total_app_samples = 0
    unique_traffic_samples = {}
    for traffic_features, _, sample_ids, app_type_ids, _ in dataloader:
        if traffic_features is None:
            continue
        for i, s_id in enumerate(sample_ids):
            s_id_str = str(s_id)
            if s_id_str not in unique_traffic_samples:
                unique_traffic_samples[s_id_str] = (traffic_features[i:i + 1], app_type_ids[i])
    for s_id, (src_feat, true_app_id_tensor) in tqdm(unique_traffic_samples.items(), desc="Evaluating", leave=False):
        src_feat = src_feat.to(device)
        true_app_id_val = true_app_id_tensor.item()
        with torch.no_grad():
            app_type_pred = model.predict_app_type(src_feat)
            predicted_app_id = app_type_pred.argmax(1).item()
            if predicted_app_id == true_app_id_val:
                correct_app_predictions += 1
            total_app_samples += 1
            generated_caption, _ = model.generate_caption(src_feat, vocab, predicted_app_id)
            clean_caption = generated_caption.replace("<sos>", "").replace("<eos>", "").strip()
            candidates_corpus[s_id] = [clean_caption]
            references_corpus[s_id] = temp_gts.get(s_id, [])
    scores = {}
    if coco_scorers and references_corpus and candidates_corpus:
        eval_ids = set(references_corpus.keys()) & set(candidates_corpus.keys())
        if eval_ids:
            filtered_gts = {img_id: references_corpus[img_id] for img_id in eval_ids if references_corpus[img_id]}
            filtered_res = {img_id: candidates_corpus[img_id] for img_id in eval_ids if candidates_corpus[img_id]}
            if filtered_gts and filtered_res:
                for scorer_name, scorer_obj in coco_scorers.items():
                    try:
                        score, _ = scorer_obj.compute_score(filtered_gts, filtered_res)
                        if isinstance(score, list):
                            for i in range(4):
                                scores[f"{scorer_name}_{i + 1}"] = score[i]
                        else:
                            scores[scorer_name] = score
                    except Exception as e:
                        print(f"Warning: {scorer_name} computation failed ({e})")
                        if scorer_name.lower().startswith("bleu"):
                            for i in range(4):
                                scores[f"{scorer_name}_{i + 1}"] = 0.0
                        else:
                            scores[scorer_name] = 0.0
    scores["App_Accuracy"] = correct_app_predictions / total_app_samples if total_app_samples > 0 else 0.0
    return scores