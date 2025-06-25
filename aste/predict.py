# Use pretrained SpanModel weights for prediction
import sys
sys.path.append("aste")
from pathlib import Path
from data_utils import Data, Sentence, SplitEnum
from wrapper import SpanModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")


def predict_sentence(text: str, model: SpanModel) -> Sentence:
    path_in = "sample_input.txt"
    path_out = "sample_output.txt"
    
    sent = Sentence(tokens=tokenizer.tokenize(text), triples=[], pos=[], is_labeled=False, weight=1, id=0)
    data = Data(root=Path(), data_split=SplitEnum.test, sentences=[sent])
    data.save_to_path(path_in)
    
    model.predict(path_in, path_out)
    data = Data.load_from_full_path(path_out)
    print("[DEBUG] dat.sentences:", data.sentences)
    return data.sentences[0]

# text = "I love the shape , but the fan is too noisy ."
# text = "Did not enjoy the new Windows 8 and touchscreen functions ."
text = "디자인이 예쁘고 속도도 빨라요 ."
# text = "말라보이긴 한데 좀 애매해서 살 빼고 입겟읍니덩"
# text = "색상은 같은데, 핏이 별로예요."
# text = "색 개다름.."
model = SpanModel(save_dir="pretrained_dir", random_seed=0)
sent = predict_sentence(text, model)

print("[DEBUG] sent.triples", sent.triples)

for t in sent.triples:
    target = " ".join(sent.tokens[t.t_start:t.t_end+1])
    opinion = " ".join(sent.tokens[t.o_start:t.o_end+1])
    print(dict(target=target, opinion=opinion, sentiment=t.label))
