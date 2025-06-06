data_name = "14lap"

import sys
sys.path.append("aste")
from data_utils import Data

path = f"aste/data/triplet_data/{data_name}/train.txt"
data = Data.load_from_full_path(path)

for s in data.sentences[:3]:
    print("tokens:", s.tokens)
    for t in s.triples:
        print("target:", (t.t_start, t.t_end))
        print("opinion:", (t.o_start, t.o_end))
        print("label:", t.label)
    print()