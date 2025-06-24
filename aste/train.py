import sys
sys.path.append("aste")

from wrapper import SpanModel

RANDOM_SEED = 5
PATH_TRAIN = "aste/data/triplet_data/korean_sample/train.txt"
PATH_DEV = "aste/data/triplet_data/korean_sample/dev.txt"
PATH_TEST = "aste/data/triplet_data/korean_sample/test.txt"
# PATH_TRAIN = "aste/data/triplet_data/14lap/train.txt"
# PATH_DEV = "aste/data/triplet_data/14lap/dev.txt"
# PATH_TEST = "aste/data/triplet_data/14lap/test.txt"
SAVE_DIR = f"outputs/sample/seed_{RANDOM_SEED}"


def train_from_scratch(path_train, path_dev, save_dir):
    model = SpanModel(save_dir=save_dir, random_seed=RANDOM_SEED, train_data_path=PATH_TRAIN, validation_data_path=PATH_DEV)
    model.fit(path_train, path_dev)

train_from_scratch(PATH_TRAIN, PATH_DEV, SAVE_DIR)
