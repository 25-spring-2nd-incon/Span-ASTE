import sys
sys.path.append("aste")

from wrapper import SpanModel
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


RANDOM_SEED = 42
PATH_TRAIN = "aste/data/triplet_data/ably/jsonl/train.jsonl"
PATH_DEV = "aste/data/triplet_data/ably/jsonl/dev.jsonl"
PATH_TEST = "aste/data/triplet_data/ably/jsonl/test.jsonl"
SAVE_DIR = f"outputs/sample/seed_{RANDOM_SEED}_{timestamp}"


def train_from_scratch(path_train, path_dev, save_dir):
    model = SpanModel(save_dir=save_dir, random_seed=RANDOM_SEED)
    model.fit(path_train, path_dev)

train_from_scratch(PATH_TRAIN, PATH_DEV, SAVE_DIR)
