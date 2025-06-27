# aste/bulk_predict.py

import sys
import json
import csv
from pathlib import Path
from tqdm import tqdm  # 진행 상황을 보여주기 위한 라이브러리 (pip install tqdm)

# --- [1. 경로 설정 및 모듈 Import] ---
# 프로젝트 루트 경로를 추가하여 aste.wrapper를 찾을 수 있게 합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from aste.wrapper import SpanModel
from transformers import AutoTokenizer

# --- [2. 이전 단계에서 완성한 예측 및 후처리 함수] ---
# 이 함수들은 이제 단일 문장을 처리하는 부품(component) 역할을 합니다.

def predict_sentence(text: str, model: SpanModel) -> dict:
    """하나의 문장 텍스트를 입력받아, 모델 예측을 수행하고 원본 결과를 반환합니다."""
    model_save_dir = Path(model.save_dir)
    path_in = model_save_dir / "temp_predict_input.jsonl"
    path_out = model_save_dir / "temp_predict_output.jsonl"
    
    input_data = {"sentence": text}
    
    with open(path_in, 'w', encoding='utf-8') as f:
        f.write(json.dumps(input_data, ensure_ascii=False) + '\n')
        
    model.predict(str(path_in), str(path_out))
    
    try:
        with open(path_out, 'r', encoding='utf-8') as f:
            prediction_result = json.loads(f.readline())
        path_in.unlink()
        path_out.unlink()
        return prediction_result
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"오류: 예측 결과 파일을 읽는 데 실패했습니다. 문장: '{text}', 에러: {e}")
        if path_in.exists():
            path_in.unlink()
        return None

def post_process_prediction(prediction: dict, tokenizer: AutoTokenizer):
    """모델의 원본 예측 결과(인덱스 기반)를 텍스트 기반 결과로 변환합니다."""
    tokens = prediction.get('tokens', [])
    predicted_triplets = prediction.get('predicted_triples', [])
    
    if not tokens or not predicted_triplets:
        return []

    processed_results = []
    for t in predicted_triplets:
        o_span_indices = t.get('o_span')
        t_span_indices = t.get('t_span')
        
        if not o_span_indices or not t_span_indices:
            continue
            
        opinion_tokens = tokens[o_span_indices[0] : o_span_indices[1] + 1]
        target_tokens = tokens[t_span_indices[0] : t_span_indices[1] + 1]
        
        opinion_text = tokenizer.convert_tokens_to_string(opinion_tokens)
        target_text = tokenizer.convert_tokens_to_string(target_tokens)
        
        processed_results.append({
            "aspect_span": target_text,
            "opinion_span": opinion_text,
            "sentiment": t.get('label')
        })
        
    return processed_results

# --- [3. 메인 실행 블록] ---
if __name__ == '__main__':
    # --- 설정 값 ---
    # 훈련된 모델이 저장된 디렉토리 경로
    MODEL_SAVE_DIRECTORY = "outputs/sample/seed_42_20250627_023239_best" # <--- 사용자님의 실제 경로로 수정!
    
    # 입력/출력 CSV 파일 경로
    INPUT_CSV_PATH = "aste/sentences.csv"
    OUTPUT_CSV_PATH = "b_prediction_result.csv"

    # --- 실행 ---
    # 모델과 토크나이저를 한 번만 로드합니다.
    print(f"모델 로딩 중... 경로: {MODEL_SAVE_DIRECTORY}")
    try:
        model = SpanModel(save_dir=MODEL_SAVE_DIRECTORY, random_seed=0)
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    except Exception as e:
        print(f"모델 로딩에 실패했습니다. 경로를 확인해주세요. 에러: {e}")
        sys.exit(1) # 모델 로딩 실패 시 스크립트 종료

    # 입력 CSV 파일을 읽습니다.
    try:
        with open(INPUT_CSV_PATH, "r", encoding="utf-8") as f_in:
            reader = csv.reader(f_in)
            header = next(reader)  # 헤더 읽기
            sentences_to_process = [row for row in reader if row]
    except FileNotFoundError:
        print(f"입력 파일 '{INPUT_CSV_PATH}'를 찾을 수 없습니다.")
        sys.exit(1)
        
    print(f"'{INPUT_CSV_PATH}' 파일에서 총 {len(sentences_to_process)}개의 문장을 발견했습니다.")
    print(f"예측을 시작합니다... 결과는 '{OUTPUT_CSV_PATH}'에 저장됩니다.")

    # 출력 CSV 파일을 열고 결과를 씁니다.
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["product_ID", "cleaned_sentence", "aspect_span", "opinion_span", "sentiment"])
        
        # tqdm을 사용하여 진행 상황을 시각적으로 표시합니다.
        for row in tqdm(sentences_to_process, desc="문장 처리 중"):
            product_ID = row[0]
            cleaned_sentence = row[1]

            # 1. 단일 문장 예측
            raw_prediction = predict_sentence(cleaned_sentence, model)
            
            if not raw_prediction:
                continue

            # 2. 결과 후처리 (인덱스 -> 텍스트)
            processed_results = post_process_prediction(raw_prediction, tokenizer)
            
            # 3. 후처리된 결과를 CSV에 저장
            if processed_results:
                for result in processed_results:
                    writer.writerow([
                        product_ID,
                        cleaned_sentence,
                        result["aspect_span"],
                        result["opinion_span"],
                        result["sentiment"]
                    ])
            else:
                # 예측된 triplet이 없는 경우, 빈 행을 쓸 수도 있습니다 (선택 사항)
                writer.writerow([product_ID, cleaned_sentence, "", "", ""])

    print("모든 예측이 완료되었습니다!")