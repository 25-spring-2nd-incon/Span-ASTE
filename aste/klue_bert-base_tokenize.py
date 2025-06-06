from transformers import AutoTokenizer

def get_span_indices(sentence, span_text, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    span_tokens = tokenizer.tokenize(span_text)

    # 슬라이딩 윈도우 방식으로 부분 문자열 검색
    for i in range(len(tokens) - len(span_tokens) + 1):
        if tokens[i:i + len(span_tokens)] == span_tokens:
            return i, i + len(span_tokens) - 1

    raise ValueError(f"Span '{span_text}' not found in tokenized sentence.")

# 예시
sentence = "재질도 넘 좋구 밑이 라운드 모양으로 마감 된게 너무 이뻐여..!"
aspect = "재질"
opinion = "넘 좋구"

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 결과 확인
aspect_span = get_span_indices(sentence, aspect, tokenizer)
opinion_span = get_span_indices(sentence, opinion, tokenizer)

print("Sentence tokens:", tokenizer.tokenize(sentence))
print("Aspect span indices:", aspect_span)
print("Opinion span indices:", opinion_span)

# 데이터 예시
# 데이터 예시 (KLUE/BERT tokenizer 기준 indexing 적용)
sentences = [
    {
        "sentence": "말라보이긴 한데 좀 애매해서 살 빼고 입겟읍니덩",
        "triples": [
            {
                "aspect_span": "말라보이긴",
                "opinion_span": "애매",
                "polarity": "NEG",
            }
        ]
    },
    {
        "sentence": "무난하고 완전 굳",
        "triples": [
            {
                "aspect_span": "무난하고",
                "opinion_span": "완전 굳",
                "polarity": "POS",
            }
        ]
    },
    {
        "sentence": "재질도 넘 좋구 밑이 라운드 모양으로 마감 된게 너무 이뻐여 ..!",
        "triples": [
            {
                "aspect_span": "재질",
                "opinion_span": "좋구",
                "polarity": "POS",
            },
            {
                "aspect_span": "밑",
                "opinion_span": "이뻐여",
                "polarity": "POS",
            }
        ]
    }
]

results = []

def label_indices(results, sentences, tokenizer):
    for s in sentences:
        sentence = s["sentence"]
        indexed_result = sentence + "#### #### ####"
        indexed_triples = []

        for t in s["triples"]:
            aspect_span = t["aspect_span"]
            opinion_span = t["opinion_span"]
            polarity = t["polarity"]

            indexed_spans = (
                get_span_indices(sentence, aspect_span, tokenizer),
                get_span_indices(sentence, opinion_span, tokenizer),
                polarity
            )
            indexed_triples.append(indexed_spans)

        indexed_result += str(indexed_triples)
        results.append(indexed_result)

label_indices(results, sentences, tokenizer)

for r in results:
    print(r)