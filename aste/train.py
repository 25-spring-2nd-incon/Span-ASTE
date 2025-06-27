import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 파이썬 모듈 검색 경로에 프로젝트 루트를 추가합니다.
sys.path.append(str(PROJECT_ROOT))
# --- [수정 완료] ---

# 이제 wrapper는 프로젝트 루트 기준으로 import 됩니다.
# from wrapper import SpanModel -> from aste.wrapper import SpanModel
from aste.wrapper import SpanModel
# from wrapper import SpanModel
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

dir = "jsonl" # jsonl or fake
RANDOM_SEED = 42
PATH_TRAIN = f"aste/data/triplet_data/ably/{dir}/train.jsonl"
PATH_DEV = f"aste/data/triplet_data/ably/{dir}/dev.jsonl"
PATH_TEST = f"aste/data/triplet_data/ably/{dir}/test.jsonl"
SAVE_DIR = f"outputs/sample/seed_{RANDOM_SEED}_{timestamp}"


def train_from_scratch(path_train, path_dev, save_dir):
    model = SpanModel(save_dir=save_dir, random_seed=RANDOM_SEED)
    model.fit(path_train, path_dev)

train_from_scratch(PATH_TRAIN, PATH_DEV, SAVE_DIR)
