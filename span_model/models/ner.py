import logging
from typing import Any, Dict, List, Optional, Callable

import torch
from torch.nn import functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import util, RegularizerApplicator

from span_model.models.shared import BiAffineSingleInput
from span_model.training.ner_metrics import NERMetrics
# from span_model.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NERTagger(Model):
    """
    Named entity recognition module

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        make_feedforward: Callable,
        span_emb_dim: int,
        regularizer: Optional[RegularizerApplicator] = None,
        name: str = "ner_labels",
    ) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self._name = name
        self._namespaces = [
            entry for entry in vocab.get_namespaces() if self._name in entry
        ]

        # Number of classes determine the output dimension of the final layer
        self._n_labels = {name: vocab.get_vocab_size(name) for name in self._namespaces}

        # Null label is needed to keep track of when calculating the metrics
        for namespace in self._namespaces:
            null_label = vocab.get_token_index("", namespace)
            assert (
                null_label == 0
            )  # If not, the dummy class won't correspond to the null label.

        # The output dim is 1 less than the number of labels because we don't score the null label;
        # we just give it a score of 0 by default.

        # Create a separate scorer and metric for each dataset we're dealing with.
        self._ner_scorers = torch.nn.ModuleDict()
        self._ner_metrics = {}

        for namespace in self._namespaces:
            self._ner_scorers[namespace] = self.make_scorer(
                make_feedforward, span_emb_dim, self._n_labels[namespace]
            )

            self._ner_metrics[namespace] = NERMetrics(
                self._n_labels[namespace], null_label
            )

            self.i_opinion = vocab.get_token_index("OPINION", namespace)
            self.i_target = vocab.get_token_index("TARGET", namespace)

        self._active_namespace = None
        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")
        print(dict(ner_loss_fn=self._loss))

    def make_scorer(self, make_feedforward, span_emb_dim, n_labels):
        mention_feedforward = make_feedforward(input_dim=span_emb_dim)
        scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(
                torch.nn.Linear(mention_feedforward.get_output_dim(), n_labels)
            ),
        )
        return scorer

    @overrides
    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,
        span_mask: torch.IntTensor,
        span_embeddings: torch.IntTensor,
        sentence_lengths: torch.Tensor,
        ner_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        TODO: Write documentation.
        """

        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings
        dataset_name = "ably"
        self._active_namespace = f"{dataset_name}__{self._name}"
        scorer = self._ner_scorers[self._active_namespace]
        ner_scores = scorer(span_embeddings)

        # Give large positive scores to "null" class in masked-out elements
        ner_scores[..., 0] = util.replace_masked_values(
            ner_scores[..., 0], span_mask.bool(), 1e20
        )
        _, predicted_ner = ner_scores.max(2)

        predictions = self.predict(
            ner_scores.detach().cpu(),
            spans.detach().cpu(),
            span_mask.detach().cpu(),
            metadata,
        )
        output_dict = {"predictions": predictions}
        # New
        output_dict.update(ner_scores=ner_scores)
        output_dict.update(
            opinion_scores=ner_scores.softmax(dim=-1)[..., [self.i_opinion]]
        )
        output_dict.update(
            target_scores=ner_scores.softmax(dim=-1)[..., [self.i_target]]
        )

        if ner_labels is not None:
            metrics = self._ner_metrics[self._active_namespace]
            metrics(predicted_ner, ner_labels, span_mask)
            ner_scores_flat = ner_scores.view(
                -1, self._n_labels[self._active_namespace]
            )
            ner_labels_flat = ner_labels.view(-1)
            mask_flat = span_mask.view(-1).bool()

            loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])

            output_dict["loss"] = loss

        return output_dict

    def predict(self, ner_scores, spans, span_mask, metadata):
        # TODO: Make sure the iteration works in documents with a single sentence.
        # Zipping up and iterating iterates over the zeroth dimension of each tensor; this
        # corresponds to iterating over sentences.
        predictions = []
        zipped = zip(ner_scores, spans, span_mask, metadata)
        for ner_scores_sent, spans_sent, span_mask_sent, sentence in zipped:
            predicted_scores_raw, predicted_labels = ner_scores_sent.max(dim=1)
            softmax_scores = F.softmax(ner_scores_sent, dim=1)
            predicted_scores_softmax, _ = softmax_scores.max(dim=1)
            ix = (predicted_labels != 0) & span_mask_sent.bool()

            predictions_for_sentence = []
            
            # 예측된 레이블이 있는 스팬들만 순회합니다.
            spans_with_labels = spans_sent[ix]
            labels_with_labels = predicted_labels[ix]
            
            for span, label_index in zip(spans_with_labels, labels_with_labels):
                span_start, span_end = span.tolist() # [start, end]
                label_str = self.vocab.get_token_from_index(
                    label_index.item(), self._active_namespace
                )
                
                # 예측 결과를 [start, end, label] 형태의 리스트로 저장합니다.
                predictions_for_sentence.append([span_start, span_end, label_str])
            
            predictions.append(predictions_for_sentence)

        return predictions

    # TODO: This code is repeated elsewhere. Refactor.
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._ner_metrics.items():
            precision, recall, f1 = metrics.get_metric(reset)
            prefix = namespace.replace("_labels", "")
            to_update = {
                f"{prefix}_precision": precision,
                f"{prefix}_recall": recall,
                f"{prefix}_f1": f1,
            }
            res.update(to_update)

        res_avg = {}
        for name in ["precision", "recall", "f1"]:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__{self._name.replace('_labels', '')}_{name}"] = (
                sum(values) / len(values) if values else 0
            )
            res.update(res_avg)

        return res
