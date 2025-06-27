import json
import logging
import pickle as pkl
import warnings
from typing import Any, Dict, List, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import (
    AdjacencyField,
    LabelField,
    ListField,
    MetadataField,
    SpanField,
    TextField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from span_model.data.dataset_readers.document import Document, Sentence

from transformers import AutoTokenizer

from pathlib import Path # 파일 경로를 다루기 위해 추가


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SpanModelDataException(Exception):
    pass


@DatasetReader.register("span_model")
class SpanModelReader(DatasetReader):
    """
    [FROM]
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    
    [TO]
    JSON Lines 형식의 파일을 읽어 ASTE(Aspect Sentiment Triplet Extraction)를 위한
    Instance를 생성하는 DatasetReader입니다.
    모든 토큰화와 인덱싱은 이 클래스 내부에서 동적으로 수행됩니다.
    """

    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer_name: str = "klue/bert-base", # config 파일로부터 전달 
        error_log_path: str = None, 
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # New
        self.is_train = False

        print("#" * 80)

        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        # tokenizer (Single Source of Truth)        
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # --- [핵심 수정 2] 오류 로그 파일 핸들러 설정 ---

        # self._error_log_file = None
        # if error_log_path:
        #     # 파일이 쓰일 디렉토리가 없으면 생성합니다.
        #     Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)
        #     # 파일을 추가(append) 모드로 엽니다.
        #     self._error_log_file = open(error_log_path, "a", encoding="utf-8")
        # --- [수정 완료] ---

    # --- [핵심 수정 3] 클래스가 소멸될 때 파일을 안전하게 닫도록 합니다. ---
    # def __del__(self):
    #     if self._error_log_file:
    #         self._error_log_file.close()
    # --- [수정 완료] ---

    @overrides
    def _read(self, file_path: str):    
        """
        .jsonl 파일을 한 줄씩 읽어서 json 객체로 파싱하고, text_to_instance로 넘깁니다.
        """
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        self.is_train = "train" in file_path  # New
        # for line in lines:
        #     # Loop over the documents.
        #     doc_text = json.loads(line)
        #     instance = self.text_to_instance(doc_text)
        #     yield instance

        # jsonl 파일을 한 줄씩 읽어서 json 객체로 변환 -> text to instance()로 넘김
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                doc_json = json.loads(line)
                yield self.text_to_instance(doc_json)

    def _find_subsequence_indices(self, main_list: List, sub_list: List) -> Tuple[int, int]:
        """헬퍼 함수: 메인 리스트에서 서브 리스트의 시작/끝 인덱스를 찾습니다."""
        main_len, sub_len = len(main_list), len(sub_list)
        if (sub_len == 0):
            return -1, -1
        for i in range(main_len - sub_len + 1):
            if main_list[i: i + sub_len] == sub_list:
                return i, i + sub_len - 1
        return -1, -1

    def _too_long(self, span):
        return span[1] - span[0] + 1 > self._max_span_width

    # def _process_ner(self, span_tuples, sent):
    #     ner_labels = [""] * len(span_tuples)

    #     for span, label in sent.ner_dict.items():
    #         if self._too_long(span):
    #             continue
    #         # New
    #         if span not in span_tuples:
    #             continue
    #         ix = span_tuples.index(span)
    #         ner_labels[ix] = label

    #     return ner_labels

    # def _process_relations(self, span_tuples, sent):
    #     relations = []
    #     relation_indices = []

    #     # Loop over the gold spans. Look up their indices in the list of span tuples and store
    #     # values.
    #     for (span1, span2), label in sent.relation_dict.items():
    #         # If either span is beyond the max span width, skip it.
    #         if self._too_long(span1) or self._too_long(span2):
    #             continue
    #         # New
    #         if (span1 not in span_tuples) or (span2 not in span_tuples):
    #             continue
    #         ix1 = span_tuples.index(span1)
    #         ix2 = span_tuples.index(span2)
    #         relation_indices.append((ix1, ix2))
    #         relations.append(label)

    #     return relations, relation_indices

    # def _process_sentence(self, sent: Sentence, dataset: str):
    #     # Get the sentence text and define the `text_field`.
    #     sentence_text = [self._normalize_word(word) for word in sent.text]
    #     text_field = TextField(
    #         [Token(word) for word in sentence_text], self._token_indexers
    #     )

    #     # Enumerate spans.
    #     spans = []
    #     for start, end in enumerate_spans(
    #         sentence_text, max_span_width=self._max_span_width
    #     ):
    #         spans.append(SpanField(start, end, text_field))

    #     # New
    #     # spans = spans[:len(spans)//2]  # bug: deliberately truncate
    #     # labeled:Set[Tuple[int, int]] = set([span for span,label in sent.ner_dict.items()])
    #     # for span_pair, label in sent.relation_dict.items():
    #     #     labeled.update(span_pair)
    #     # existing:Set[Tuple[int, int]] = set([(s.span_start, s.span_end) for s in spans])
    #     # for start, end in labeled:
    #     #     if (start, end) not in existing:
    #     #         spans.append(SpanField(start, end, text_field))

    #     span_field = ListField(spans)
    #     span_tuples = [(span.span_start, span.span_end) for span in spans]

    #     # Convert data to fields.
    #     # NOTE: The `ner_labels` and `coref_labels` would ideally have type
    #     # `ListField[SequenceLabelField]`, where the sequence labels are over the `SpanField` of
    #     # `spans`. But calling `as_tensor_dict()` fails on this specific data type. Matt G
    #     # recognized that this is an AllenNLP API issue and suggested that represent these as
    #     # `ListField[ListField[LabelField]]` instead.
    #     fields = {}
    #     fields["text"] = text_field
    #     fields["spans"] = span_field

    #     if sent.ner is not None:
    #         ner_labels = self._process_ner(span_tuples, sent)
    #         fields["ner_labels"] = ListField(
    #             [
    #                 LabelField(entry, label_namespace=f"{dataset}__ner_labels")
    #                 for entry in ner_labels
    #             ]
    #         )
    #     if sent.relations is not None:
    #         relation_labels, relation_indices = self._process_relations(
    #             span_tuples, sent
    #         )
    #         fields["relation_labels"] = AdjacencyField(
    #             indices=relation_indices,
    #             sequence_field=span_field,
    #             labels=relation_labels,
    #             label_namespace=f"{dataset}__relation_labels",
    #         )

    #     return fields

    # def _process_sentence_fields(self, doc: Document):
    #     # Process each sentence.
    #     sentence_fields = [
    #         self._process_sentence(sent, doc.dataset) for sent in doc.sentences
    #     ]

    #     # Make sure that all sentences have the same set of keys.
    #     first_keys = set(sentence_fields[0].keys())
    #     for entry in sentence_fields:
    #         if set(entry.keys()) != first_keys:
    #             raise SpanModelDataException(
    #                 f"Keys do not match across sentences for document {doc.doc_key}."
    #             )

    #     # For each field, store the data from all sentences together in a ListField.
    #     fields = {}
    #     keys = sentence_fields[0].keys()
    #     for key in keys:
    #         this_field = ListField([sent[key] for sent in sentence_fields])
    #         fields[key] = this_field

    #     return fields

    @overrides
    def text_to_instance(self, doc_json: Dict[str, Any]) -> Instance:
        sentence_text = doc_json["sentence"]
        triples = doc_json.get("triples", [])
        dataset = "ably"

        tokenization_output = self._tokenizer(
            sentence_text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        

        token_strings = self._tokenizer.convert_ids_to_tokens(tokenization_output["input_ids"])
        char_offsets = tokenization_output["offset_mapping"]

        tokens = [Token(text) for text in token_strings]
        text_field = TextField(tokens, self._token_indexers)

        spans = [
            SpanField(start, end, text_field)
            for start, end in enumerate_spans(token_strings, max_span_width=self._max_span_width)
        ]
        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]
        
        ner_dict = {}
        relation_dict = {}

        for triple in triples:
            aspect_text = triple["aspect_span"]
            opinion_text = triple["opinion_span"]
            sentiment = triple.get("sentiment") or triple.get("polarity")

            try:
                a_char_start = sentence_text.find(aspect_text)
                if a_char_start == -1: raise ValueError(f"Aspect '{aspect_text}'")
                a_char_end = a_char_start + len(aspect_text)
                
                o_char_start = sentence_text.find(opinion_text)
                if o_char_start == -1: raise ValueError(f"Opinion '{opinion_text}'")
                o_char_end = o_char_start + len(opinion_text)
            except ValueError as e:
                logger.warning(f"Span text {e} not found in sentence: '{sentence_text}'")
                continue

            aspect_token_indices = []
            opinion_token_indices = []
            
            for i, (token_char_start, token_char_end) in enumerate(char_offsets):
                
                if token_char_start >= a_char_start and token_char_end <= a_char_end:
                    aspect_token_indices.append(i)

                if token_char_start >= o_char_start and token_char_end <= o_char_end:
                    opinion_token_indices.append(i)

            if not aspect_token_indices or not opinion_token_indices:
                logger.warning(f"Could not map char span to token span. Triple: {triple}, Sentence: {sentence_text}")
                continue

            # 변환된 토큰 인덱스로 최종 스팬 튜플 생성
            aspect_span = (aspect_token_indices[0], aspect_token_indices[-1])
            opinion_span = (opinion_token_indices[0], opinion_token_indices[-1])
            
            ner_dict[aspect_span] = "TARGET"
            ner_dict[opinion_span] = "OPINION"
            relation_dict[(opinion_span, aspect_span)] = sentiment
        
        fields = {"text": text_field, "spans": span_field}
        ner_labels = [""] * len(span_tuples)
        for span, label in ner_dict.items():
            if span in span_tuples:
                ix = span_tuples.index(span)
                ner_labels[ix] = label
        fields["ner_labels"] = ListField([LabelField(l, f"{dataset}__ner_labels") for l in ner_labels])
        
        relation_labels, relation_indices = [], []
        for (span1, span2), label in relation_dict.items():
            if span1 in span_tuples and span2 in span_tuples:
                ix1 = span_tuples.index(span1)
                ix2 = span_tuples.index(span2)
                relation_indices.append((ix1, ix2))
                relation_labels.append(label)
        fields["relation_labels"] = AdjacencyField(
            indices=relation_indices, sequence_field=span_field,
            labels=relation_labels, label_namespace=f"{dataset}__relation_labels"
        )
        
        metadata_to_pass = {
            "original_json": doc_json,
            "relation_dict": relation_dict
        }
        fields["metadata"] = MetadataField(metadata_to_pass)
        
        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, "rb") as f:
            for entry in pkl.load(f):
                yield entry

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        with open(cache_filename, "wb") as f:
            pkl.dump(instances, f, protocol=pkl.HIGHEST_PROTOCOL)

    # @staticmethod
    # def _normalize_word(word):
    #     if word == "/." or word == "/?":
    #         return word[1:]
    #     else:
    #         return word
