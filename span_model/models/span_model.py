import logging
from typing import Dict, List, Optional, Union, Any
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from span_model.models.ner import NERTagger
from span_model.models.relation_proper import ProperRelationExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("span_model")
class SpanModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        modules,  # TODO: Add type.
        feature_size: int,
        max_span_width: int,
        target_task: str,
        feedforward_params: Dict[str, Union[int, float]],
        loss_weights: Dict[str, float],
        initializer: InitializerApplicator = InitializerApplicator(),
        module_initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        display_metrics: List[str] = None,
        # New
        span_extractor_type: str = None,
        use_span_width_embeds: bool = None,
        **kwargs,
    ) -> None:
        super(SpanModel, self).__init__(vocab, regularizer)
        print(dict(span_model_unused_keys=kwargs.keys()))

        # New
        info = dict(
            span_extractor_type=span_extractor_type,
            use_span_width_embeds=use_span_width_embeds,
        )
        for k, v in info.items():
            print(dict(locals=(k, v)))
            assert v is not None, k

        ####################

        assert span_extractor_type in {"endpoint"}
        # Create span extractor.
        if use_span_width_embeds:
            self._endpoint_span_extractor = EndpointSpanExtractor(
                embedder.get_output_dim(),
                combination="x,y",
                num_width_embeddings=max_span_width,
                span_width_embedding_dim=feature_size,
                bucket_widths=False,
            )
        # New
        else:
            self._endpoint_span_extractor = EndpointSpanExtractor(
                embedder.get_output_dim(),
                combination="x,y",
            )
        self._visualize_outputs = []

        ####################

        # Set parameters.
        self._embedder = embedder
        self._loss_weights = loss_weights
        self._max_span_width = max_span_width
        self._display_metrics = self._get_display_metrics(target_task)
        span_emb_dim = self._endpoint_span_extractor.get_output_dim()

        # New
        self._feature_size = feature_size
        ####################

        # Create submodules.

        modules = Params(modules)

        # Helper function to create feedforward networks.
        def make_feedforward(input_dim):
            return FeedForward(
                input_dim=input_dim,
                num_layers=feedforward_params["num_layers"],
                hidden_dims=feedforward_params["hidden_dims"],
                activations=torch.nn.ReLU(),
                dropout=feedforward_params["dropout"],
            )

        # Submodules

        self._ner = NERTagger.from_params(
            vocab=vocab,
            make_feedforward=make_feedforward,
            span_emb_dim=span_emb_dim,
            feature_size=feature_size,
            params=modules.pop("ner"),
        )

        params = dict(
            vocab=vocab,
            make_feedforward=make_feedforward,
            span_emb_dim=span_emb_dim,
            feature_size=feature_size,
            params=modules.pop("relation"),
        )
        self._relation = ProperRelationExtractor.from_params(**params)

        ####################

        # Initialize text embedder and all submodules
        for module in [self._ner, self._relation]:
            module_initializer(module)

        initializer(self)

    @staticmethod
    def _get_display_metrics(target_task):
        """
        The `target` is the name of the task used to make early stopping decisions. Show metrics
        related to this task.
        """
        lookup = {
            "ner": [
                f"MEAN__{name}" for name in ["ner_precision", "ner_recall", "ner_f1"]
            ],
            "relation": [
                f"MEAN__{name}"
                for name in ["relation_precision", "relation_recall", "relation_f1"]
            ],
        }
        if target_task not in lookup:
            raise ValueError(
                f"Invalied value {target_task} has been given as the target task."
            )
        return lookup[target_task]

    @staticmethod
    def _debatch(x):
        # TODO: Get rid of this when I find a better way to do it.
        return x if x is None else x.squeeze(0)

    def text_to_span_embeds(self, text_embeddings: torch.Tensor, spans):
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)
        return span_embeddings

    # span_model/models/span_model.py 의 SpanModel 클래스 내부

    @overrides
    def forward(
        self,
        text: Dict[str, torch.LongTensor],
        spans: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        ner_labels: torch.LongTensor = None,
        relation_labels: torch.LongTensor = None,
    ):
        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        if relation_labels is not None:
            relation_labels = relation_labels.long()

        # -------------------- [핵심 수정 시작] --------------------
        # 1. '문서(document)' 기반의 모든 로직을 제거합니다.
        #    - _debatch, num_wrapping_dims, squeeze() 등은 더 이상 필요 없습니다.
        #    - 이제 모든 텐서는 (batch_size, ...) 모양을 유지합니다.

        # (batch_size, num_tokens, embedding_dim)
        text_embeddings = self._embedder(text)

        # (batch_size, num_tokens)
        text_mask = util.get_text_field_mask(text).float()
        
        # (batch_size)
        sentence_lengths = text_mask.sum(dim=-1).long()

        # 2. 에러가 발생했던 부분을 텐서의 실제 모양에 맞게 수정합니다.
        #    - spans 텐서의 모양은 (batch_size, num_spans, 2) 입니다.
        #    - 따라서 `spans[:, :, 0]`은 올바른 3차원 인덱싱입니다.
        #      이전의 에러는 다른 코드와의 상호작용으로 인한 차원 감소 때문이었을 수 있습니다.
        #      이제는 파이프라인 전체를 일관되게 배치 기준으로 처리합니다.
        
        # (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).float()
        
        # (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # -------------------- [핵심 수정 종료] --------------------

        # New: 두 종류의 임베딩 생성 (기존 로직 유지)
        # 4. 스팬 임베딩 추출
        # (batch_size, num_spans, span_embedding_dim)
        span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)
        
        # 5. NER 모듈 호출
        # 이 모듈은 loss, predictions, 그리고 relation 모듈이 사용할 점수들을 반환합니다.
        output_ner = self._ner(
            spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata
        )
        
        # 6. Relation 모듈 호출
        # Relation 모듈에 필요한 정보를 명시적으로 전달합니다.
        self._relation._opinion_scores = output_ner.get("opinion_scores")
        self._relation._target_scores = output_ner.get("target_scores")
        
        output_relation = self._relation(
            spans,
            span_mask,
            span_embeddings, # NER과 동일한 스팬 임베딩을 사용합니다.
            sentence_lengths,
            relation_labels,
            metadata,
        )

        # 7. 손실 계산
        loss = (
            self._loss_weights["ner"] * output_ner.get("loss", 0)
            + self._loss_weights["relation"] * output_relation.get("loss", 0)
        )
        
        # 8. 최종 출력 딕셔너리 구성
        output_dict = {
            "ner": output_ner,
            "relation": output_relation,
            "loss": loss,
            "metadata": metadata, # metadata를 그대로 전달하여 다음 단계에서 사용
        }

        return output_dict

    def update_span_embeddings(
        self,
        span_embeddings,
        span_mask,
        top_span_embeddings,
        top_span_mask,
        top_span_indices,
    ):
        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if (
                    top_span_mask[sample_nr, top_span_nr] == 0
                    or span_mask[sample_nr, span_nr] == 0
                ):
                    break
                new_span_embeddings[sample_nr, span_nr] = top_span_embeddings[
                    sample_nr, top_span_nr
                ]
        return new_span_embeddings

    # @overrides
    # def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
    #     """
    #     Converts the list of spans and predicted antecedent indices into clusters
    #     of spans for each element in the batch.

    #     Parameters
    #     ----------
    #     output_dict : ``Dict[str, torch.Tensor]``, required.
    #         The result of calling :func:`forward` on an instance or batch of instances.
    #     """

    #     doc = copy.deepcopy(output_dict["metadata"])

    #     if self._loss_weights["ner"] > 0:
    #         for predictions, sentence in zip(output_dict["ner"]["predictions"], doc):
    #             sentence.predicted_ner = predictions

    #     if self._loss_weights["relation"] > 0:
    #         for predictions, sentence in zip(
    #             output_dict["relation"]["predictions"], doc
    #         ):
    #             sentence.predicted_relations = predictions

    #     return doc
    # span_model.py 의 SpanModel 클래스 내부

    # span_model.py의 SpanModel 클래스 내부

    # span_model.py의 SpanModel 클래스 내부

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        모델의 예측 결과를 사람이 읽기 쉬운 딕셔너리 형태로 변환합니다.
        어떤 경우에도 항상 딕셔너리를 반환하도록 보장하는 최종 버전.
        """
        # 1. 최종적으로 반환할 기본 딕셔너리 구조를 먼저 만듭니다.
        #    metadata에서 원본 문장 정보를 가져옵니다.
        #    배치 내 첫 번째 인스턴스의 metadata를 사용합니다.
        metadata_item = output_dict["metadata"][0]
        # 우리가 SpanModelReader에서 저장한 `original_json`을 사용합니다.
        original_json = metadata_item.get("original_json", {})
        final_output = {
            "sentence": original_json.get("sentence", "Sentence not found."),
            "predicted_ner": [],
            "predicted_triples": []
        }

        # 2. NER 예측 결과가 있다면, 최종 출력에 추가합니다.
        ner_predictions = output_dict.get("ner", {}).get("predictions")
        if ner_predictions and isinstance(ner_predictions, list):
            # ner_predictions는 [[...], [...]] 형태의 리스트이므로 첫 번째 항목을 사용
            final_output["predicted_ner"] = ner_predictions[0]

        # 3. Relation 예측 결과가 있다면, 최종 출력에 추가합니다.
        relation_predictions = output_dict.get("relation", {}).get("predictions")
        if relation_predictions and isinstance(relation_predictions, list):
            # relation_predictions는 [{'sentence':..., 'predicted_triples':...}] 형태의 리스트
            # 여기서 'predicted_triples'만 가져와서 덮어씁니다.
            final_output["predicted_triples"] = relation_predictions[0].get("predicted_triples", [])

        return final_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = list(metrics_ner.keys()) + list(metrics_relation.keys())
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_ner.items()) + list(metrics_relation.items()))

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
