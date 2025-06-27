import sys
from pathlib import Path
import json

# 프로젝트 루트 경로를 추가하여 aste.wrapper를 찾을 수 있게 합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from aste.wrapper import SpanModel


def predict_sentence(text: str, model: SpanModel) -> dict:
    """
    하나의 문장 텍스트를 입력받아, 모델 예측을 수행하고 결과를 딕셔너리로 반환합니다.
    """
    # 1. 임시 파일 경로를 지정합니다.
    model_save_dir = Path(model.save_dir)
    path_in = model_save_dir / "temp_predict_input.jsonl"
    path_out = model_save_dir / "temp_predict_output.jsonl"
    
    # --- [핵심] 입력 파일을 생성하는 부분 ---
    # 2. 모델이 읽을 수 있는 jsonlines 형식으로 데이터를 만듭니다.
    input_data = {"sentence": text}
    
    # 3. 이 데이터를 jsonlines 파일에 씁니다.
    print(f"임시 입력 파일 생성 중: {path_in}")
    with open(path_in, 'w', encoding='utf-8') as f:
        f.write(json.dumps(input_data, ensure_ascii=False) + '\n')
    # --- [파일 생성 완료] ---
        
    # 4. 모델의 predict 메소드를 호출합니다.
    #    이제 model.predict는 존재하는 path_in 파일을 읽을 수 있습니다.
    model.predict(str(path_in), str(path_out))
    
    # 5. 예측 결과를 읽어옵니다.
    try:
        with open(path_out, 'r', encoding='utf-8') as f:
            prediction_result = json.loads(f.readline())
        
        # 6. 임시 파일 삭제
        #path_in.unlink()
        #path_out.unlink()
        
        return prediction_result
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"예측 결과 파일을 읽는 데 실패했습니다: {e}")
        # 임시 입력 파일이라도 삭제해줍니다.
        if path_in.exists():
            path_in.unlink()
        return None

# --- 스크립트 실행 부분 ---
if __name__ == '__main__':
    # 예측에 사용할 훈련된 모델이 저장된 디렉토리 경로
    MODEL_SAVE_DIRECTORY = "outputs/sample/seed_42_20250627_023239" # 실제 경로로 수정

    # 예측할 문장
    text_to_predict = "겉에는 보들보들한 면소재에요!"

    # 모델 로드
    print(f"모델 로딩 중... 경로: {MODEL_SAVE_DIRECTORY}")
    model = SpanModel(save_dir=MODEL_SAVE_DIRECTORY, random_seed=0)

    # 예측 함수 실행
    prediction = predict_sentence(text_to_predict, model)

    print("prediction: ", prediction)

    # 결과 출력
    if prediction:
        print("\n--- 예측 결과 ---")
        print(f"입력 문장: {prediction.get('sentence')}")
        
        predicted_triplets = prediction.get('predicted_triples', [])
        
        if not predicted_triplets:
            print("  -> 예측된 Triplet이 없습니다.")
        else:
            print("예측된 Triplet(s):")
            for t in predicted_triplets:
                print(f"  - {t}")