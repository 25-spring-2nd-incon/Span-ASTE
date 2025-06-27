from transformers import AutoTokenizer
import json

def get_span_indices(tokens, span_tokens, span_text):
    print(f"\n[DEBUG][get_span_indices]")
    print(f" - sentence tokens: {tokens}")
    print(f" - span text: {span_text}")
    print(f" - span tokens: {span_tokens}")

    for i in range(len(tokens) - len(span_tokens) + 1):
        if tokens[i:i + len(span_tokens)] == span_tokens:
            print(f" - matched index: {list(range(i, i + len(span_tokens)))}")
            return list(range(i, i + len(span_tokens)))

    raise ValueError(f"Span '{span_text}' not found in tokenized sentence.")


def label_indices(results, error_cases, sentences, tokenizer):
    for idx, s in enumerate(sentences):
        sentence = s["sentence"]
        print(f"\n{'='*30}\n[INFO] Sentence {idx+1}: {sentence}")

        indexed_result = sentence + "#### #### ####"
        indexed_triples = []

        tokens = tokenizer.tokenize(sentence)
        print(f"[DEBUG] Tokenized sentence:\n  {tokens}")
        print(f"[DEBUG] Decoded sentence:\n  {tokenizer.convert_tokens_to_string(tokens)}")

        for t in s["triples"]:
            aspect_span = t["aspect_span"]
            opinion_span = t["opinion_span"]
            polarity = t["polarity"]

            aspect_span_tokens = tokenizer.tokenize(aspect_span)
            opinion_span_tokens = tokenizer.tokenize(opinion_span)

            print(f"\n[INFO] Triple:")
            print(f"  - Aspect: '{aspect_span}' → {aspect_span_tokens}")
            print(f"  - Opinion: '{opinion_span}' → {opinion_span_tokens}")

            # 비교 강조
            aspect_match = any(
                tokens[i:i+len(aspect_span_tokens)] == aspect_span_tokens
                for i in range(len(tokens) - len(aspect_span_tokens) + 1)
            )
            opinion_match = any(
                tokens[i:i+len(opinion_span_tokens)] == opinion_span_tokens
                for i in range(len(tokens) - len(opinion_span_tokens) + 1)
            )

            print(f"  ✅ Aspect match in sentence tokens: {aspect_match}")
            print(f"  ✅ Opinion match in sentence tokens: {opinion_match}")

            try:
                indexed_spans = (
                    get_span_indices(tokens, aspect_span_tokens, aspect_span),
                    get_span_indices(tokens, opinion_span_tokens, opinion_span),
                    polarity
                )
                indexed_triples.append(indexed_spans)
            except ValueError as e:
                print(f"[ERROR] {e}")
                error_cases.append({
                    "sentence": sentence,
                    "aspect_span": aspect_span,
                    "opinion_span": opinion_span,
                    "polarity": polarity,
                    "error": str(e)
                })
                continue  # 다음 triple로 넘어감

        indexed_result += str(indexed_triples)
        results.append({
            "sentence": sentence,
            "train": indexed_result,
            "tokens": tokens
        })


if __name__ == "__main__":
    input_path = "aste/data/triplet_data/korean_sample/jsonl/test.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        formatted_data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"[INFO] ✅ 데이터 {len(formatted_data)}개 로드 완료.")

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    results = []
    error_cases = []

    label_indices(results, error_cases, formatted_data, tokenizer)

    print("\n\n[INFO] ✅ 최종 라벨링 결과:")
    for r in results:
        print(r, end="\n\n")

    # 정상 라벨링된 결과 저장
    with open('aste/data/triplet_data/korean_sample/final/test.txt', 'w', encoding='utf-8') as w:
        w.writelines([r["train"] + "\n" for r in results])

    # 오류 케이스 별도 저장
    if error_cases:
        import json
        with open('./error_cases.json', 'w', encoding='utf-8') as ew:
            json.dump(error_cases, ew, ensure_ascii=False, indent=2)
        print(f"[INFO] ⚠️ 오류 케이스 {len(error_cases)}건 저장 완료 → 'error_cases.json'")
    else:
        print("[INFO] ❌ 오류 케이스 없음.")