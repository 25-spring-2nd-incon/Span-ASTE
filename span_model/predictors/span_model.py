# span_model/predictors/span_model.py

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register("span_model")
class SpanModelPredictor(Predictor):
    """
    `span_model`을 위한 Predictor (가장 단순하고 안정적인 최종 버전).
    JSON 입력을 받아 모델의 예측 결과를 JSON으로 반환합니다.
    """
    
    # 이 클래스가 해야 할 유일한 일은, 입력 JSON을 Instance로 바꾸는 방법을 알려주는 것입니다.
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        예측 시 입력으로 들어온 JSON 딕셔너리를 AllenNLP의 Instance 객체로 변환합니다.
        훈련 때 사용한 DatasetReader의 로직을 그대로 재사용합니다.
        """
        # self._dataset_reader는 Predictor가 내부적으로 가지고 있는,
        # 훈련 시 사용했던 바로 그 SpanModelReader 객체입니다.
        return self._dataset_reader.text_to_instance(doc_json=json_dict)

    # 나머지 predict_instance, predict, dump_line 등의 메소드는 모두 삭제합니다.
    # 부모 클래스인 Predictor가 이 _json_to_instance를 사용하여 모든 것을 자동으로 처리합니다.
    # 모델의 make_output_human_readable이 반환한 dict를 그대로 최종 결과로 사용합니다.