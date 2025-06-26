import json
from pathlib import Path

import sys
sys.path.append("aste")

TRAIN_INPUT_JSON = 'aste/data/triplet_data/ably/json/train.json'
DEV_INPUT_JSON = 'aste/data/triplet_data/ably/json/dev.json'
TEST_INPUT_JSON = 'aste/data/triplet_data/ably/json/test.json'

TRAIN_OUTPUT_JSONL = 'aste/data/triplet_data/ably/jsonl/train.jsonl'
DEV_OUTPUT_JSONL = 'aste/data/triplet_data/ably/jsonl/dev.jsonl'
TEST_OUTPUT_JSONL = 'aste/data/triplet_data/ably/jsonl/test.jsonl'


def convert_json_to_jsonl(input_json_path: str, output_jsonl_path: str):    
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        if not isinstance(data, list):
            raise TypeError("입력된 JSON 파일의 최상위 요소가 리스트(배열)가 아닙니다.")

        Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

        num_lines = 0
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for item in data:
                json_string = json.dumps(item, ensure_ascii=False)
                f_out.write(json_string + '\n')
                num_lines += 1
        
        print(f"총 {num_lines}개의 객체가 '{output_jsonl_path}' 파일에 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_json_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")
    except json.JSONDecodeError:
        print(f"오류: '{input_json_path}' 파일이 올바른 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")

d = {0: (TRAIN_INPUT_JSON, TRAIN_OUTPUT_JSONL), 1: (DEV_INPUT_JSON, DEV_OUTPUT_JSONL), 2:(TEST_INPUT_JSON, TEST_OUTPUT_JSONL)}

for _, t in d.items():
    convert_json_to_jsonl(t[0], t[1])