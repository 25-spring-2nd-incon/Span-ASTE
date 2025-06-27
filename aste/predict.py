# Use pretrained SpanModel weights for prediction
import sys
sys.path.append("aste")
from pathlib import Path
from data_utils import Data, Sentence, SplitEnum
from wrapper import SpanModel
from transformers import AutoTokenizer
import csv

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

model = SpanModel(save_dir="pretrained_dir", random_seed=42)

sents = []
pred_results = []

with open("aste/sentences.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row:
            sents.append(row)

with open("a_prediction_result.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["product_ID", "cleaned_sentence", "aspect_span", "opinion_span", "sentiment"])
    for sent in sents:
        product_ID = sent[0]
        cleaned_sentence = sent[1]

        pred = predict_sentence(cleaned_sentence, model)
        print("[DEBUG] pred.triples", pred.triples)

        for t in pred.triples:
            aspect_span = tokenizer.convert_tokens_to_string(pred.tokens[t.t_start:t.t_end+1])
            opinion_span = tokenizer.convert_tokens_to_string(pred.tokens[t.o_start:t.o_end+1])
            sentiment = t.label.value
            print("[DEBUG]", dict(target=aspect_span, opinion=opinion_span, sentiment=sentiment))
            if aspect_span and opinion_span and sentiment is not None:
                writer.writerow([product_ID, cleaned_sentence, aspect_span, opinion_span, sentiment])


