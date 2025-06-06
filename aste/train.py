import sys
sys.path.append("aste")

from wrapper import SpanModel

def train_from_scratch():
    random_seed = 2
    path_train = "aste/data/triplet_data/sample/train.txt"
    path_dev = "aste/data/triplet_data/sample/dev.txt"
    path_test = "aste/data/triplet_data/sample/test.txt"
    save_dir = f"outputs/sample/seed_{random_seed}"

    model = SpanModel(save_dir=save_dir, random_seed=random_seed)
    model.fit(path_train, path_dev)

train_from_scratch()
