from pathlib import Path
import torch

# Configuration
BASE_DATA_PATH = Path("D:/PythonTest/TrafficCaptioning/trafficcaptioning")
CSV_FEATURE_PATH = BASE_DATA_PATH / "csv"
VIDEO_DESCRIPTIONS_PATH = BASE_DATA_PATH / "video_descriptions_final.pkl"
BERT_MODEL_PATH = "D:/PythonTest/TrafficCaptioning/bert-base-uncased"
SBERT_EMBEDDINGS_PATH = BASE_DATA_PATH / "sbert_embeddings.pkl"
VOCAB_MIN_FREQ = 2
MAX_CAPTION_LEN = 20
EMBEDDING_DIM_TRAFFIC = 123
EMBEDDING_DIM_TEXT = 768
TEXT_EMBEDDING_DIM = EMBEDDING_DIM_TEXT
HIDDEN_DIM = 512
NUM_LAYERS = 2
NUM_HEADS = 8
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 10
TEACHER_FORCING_RATIO = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 65
MAX_TRAFFIC_SEQ_LEN = 50
APP_TYPES = ["messaging app", "video streaming app", "social media app", "music app", "shopping app"]
NUM_APP_TYPES = len(APP_TYPES)
APP_LOSS_WEIGHT = 0.2
APP_EMBEDDING_DIM = 64
ACTION_EMBEDDING_DIM = 256
NUM_ACTION_PROTOTYPES = 5
BEAM_WIDTH = 5
CONTRASTIVE_WEIGHT = 1.0
DISTANCE_WEIGHT = 0.5
CONTRASTIVE_TEMPERATURE = 0.1
RESULTS_CSV_PATH = BASE_DATA_PATH / "training_results.csv"
COCO_CAPTION_PATH_BASE = Path("D:/PythonTest/TrafficCaptioning/")