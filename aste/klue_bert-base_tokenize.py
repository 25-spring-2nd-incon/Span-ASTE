from transformers import AutoTokenizer

def get_span_indices(tokens, span_tokens, span_text):
    """
        부분 문자열 검색을 통해 span 인덱스 찾기
        처음 일치하는 span에 대한 index만 반환하는 점이 한계임
        예: "너무 좋아요" → [2, 3]
    """

    for i in range(len(tokens) - len(span_tokens) + 1):
        if tokens[i:i + len(span_tokens)] == span_tokens:
            return list(range(i, i + len(span_tokens)))

    raise ValueError(f"Span '{span_text}' not found in tokenized sentence.")


def label_indices(results, sentences, tokenizer):
    """
        args
            - results:span indexing 결과를 저장할 리스트
            - sentences: 리뷰 데이터셋
            - tokenizer: 토크나이징에 사용할 모델 (현재는 klue/bert-base 사용 가정)
    """

    # 각 문장을 순회하면서
    for s in sentences:
        sentence = s["sentence"]
        indexed_result = sentence + "#### #### ####"
        indexed_triples = []

        tokens = tokenizer.tokenize(sentence) # 문장 전체 토큰

        # 문장에 존재하는 복수의 triples에 대해
        for t in s["triples"]:
            aspect_span = t["aspect_span"]
            opinion_span = t["opinion_span"]
            polarity = t["polarity"]

            # aspect span과 opinion span의 index 범위를 추출
            aspect_span_tokens = tokenizer.tokenize(aspect_span)
            opinion_span_tokens = tokenizer.tokenize(opinion_span)
            

            indexed_spans = (
                get_span_indices(tokens, aspect_span_tokens, aspect_span),
                get_span_indices(tokens, opinion_span_tokens, opinion_span),
                polarity
            )
            indexed_triples.append(indexed_spans)

        indexed_result += str(indexed_triples)
        results.append({
            "sentence": sentence,
            "train": indexed_result,
            "tokens": tokens            # 수기로 확인하기 위해 한 문장 전체의 tokenized 결과도 함께 저장
        })


if __name__ == "__main__":

    # 데이터 예시 (KLUE/BERT tokenizer 기준 indexing) -> csv to json 형식으로 바꿔서 한 번에 적용하기
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
            "sentence": "재질도 넘 좋구 밑이 라운드 모양으로 마감 된게 너무 이뻐여..!",
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

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    results = []
    label_indices(results, sentences, tokenizer)

    for r in results:
        print(r)