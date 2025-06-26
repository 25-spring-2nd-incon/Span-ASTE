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
        text_embeds_b = text_embeddings
        kwargs = dict(spans=spans)
        span_embeddings = self.text_to_span_embeds(text_embeddings, **kwargs)
        span_embeds_b = self.text_to_span_embeds(text_embeds_b, **kwargs)

        # 하위 모듈 호출
        output_ner = {"loss": 0}
        output_relation = {"loss": 0}

        if self._loss_weights["ner"] > 0:
            output_ner = self._ner(
                spans,
                span_mask,
                span_embeddings,
                sentence_lengths,
                ner_labels,
                metadata,
            )
            ner_scores = output_ner.pop("ner_scores")
        
        if self._loss_weights["relation"] > 0:
            if getattr(self._relation, "use_ner_scores_for_prune", False):
                self._relation._ner_scores = ner_scores
            self._relation._opinion_scores = output_ner["opinion_scores"]
            self._relation._target_scores = output_ner["target_scores"]
            self._relation._text_mask = text_mask
            self._relation._text_embeds = text_embeddings
            if getattr(self._relation, "use_span_loss_for_pruners", False):
                self._relation._ner_labels = ner_labels
            output_relation = self._relation(
                spans,
                span_mask,
                span_embeds_b,
                sentence_lengths,
                relation_labels,
                metadata,
            )

        # 손실 계산
        loss = (
            self._loss_weights["ner"] * output_ner.get("loss", 0) +
            self._loss_weights["relation"] * output_relation.get("loss", 0)
        )
        
        # weight 처리는 일단 단순화하거나, metadata 구조 확인 후 적용
        if metadata and metadata[0].get("weight") is not None:
            loss *= metadata[0].get("weight")

        output_dict = dict(
            relation=output_relation,
            ner=output_ner,
            loss=loss,
            metadata=metadata,
        )

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

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        모델의 예측 결과를 사람이 읽기 쉬운 딕셔너리 형태로 변환합니다.
        레거시 document 객체 대신 파이썬 기본 자료형을 사용하도록 수정된 버전.
        """
        # metadata는 이미 human-readable한 dict의 리스트입니다.
        # 예측 결과를 이 metadata 리스트의 각 딕셔너리에 추가합니다.
        batch_metadata = output_dict["metadata"]
        
        # NER 예측 결과 추가
        if "predictions" in output_dict.get("ner", {}):
            for i, ner_preds in enumerate(output_dict["ner"]["predictions"]):
                # 배치 크기를 벗어나는 경우를 방지
                if i < len(batch_metadata):
                    batch_metadata[i]["predicted_ner"] = ner_preds

        # Relation 예측 결과 추가
        if "predictions" in output_dict.get("relation", {}):
            for i, rel_preds in enumerate(output_dict["relation"]["predictions"]):
                if i < len(batch_metadata):
                    # 예측 결과에서 triplet 정보를 재구성하여 추가할 수 있습니다.
                    # 예를 들어, relation 모듈의 예측 결과가 [o_start, o_end, t_start, t_end, sentiment] 라면...
                    predicted_triples = []
                    for p in rel_preds:
                        # 이 부분은 relation 모듈의 실제 출력 형식에 따라 수정해야 합니다.
                        # 여기서는 p가 [o_start, o_end, t_start, t_end, sentiment] 라고 가정합니다.
                        predicted_triples.append({
                            "opinion_indices": [p[0], p[1]],
                            "target_indices": [p[2], p[3]],
                            "sentiment": p[4]
                        })
                    batch_metadata[i]["predicted_relations"] = predicted_triples
        
        # AllenNLP의 predict 커맨드는 이 딕셔너리(또는 딕셔너리의 리스트)를
        # json.dumps하여 파일에 한 줄씩 씁니다.
        # 배치 크기가 1일 때와 아닐 때를 모두 처리하기 위해,
        # 배치 전체를 리스트로 반환하는 것이 가장 안전합니다.
        return batch_metadata

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
