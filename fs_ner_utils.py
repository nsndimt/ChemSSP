import gc
import itertools
import json
import logging
import os
import random
import re
import sys
from argparse import ArgumentParser, Namespace
from collections import Counter, OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass, field
from itertools import chain, islice, product
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from CDEv2_tokenize import ChemSentenceTokenizer, ChemWordTokenizer
from datasets import Dataset, Features, Sequence, Value
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import BertTokenizerFast
from pybrat.parser import BratParser

import pulp


def total_size(o):

    def _total_size(o, handlers={}):
        """Returns the approximate memory footprint an object and all of its contents.

        Automatically finds the contents of the following builtin containers and
        their subclasses:  tuple, list, deque, dict, set and frozenset.
        To search other containers, add handlers to iterate over their contents:

            handlers = {SomeContainerClass: iter,
                        OtherContainerClass: OtherContainerClass.get_elements}

        """
        dict_handler = lambda d: chain.from_iterable(d.items())
        all_handlers = {
            tuple: iter,
            list: iter,
            dict: dict_handler,
            set: iter,
        }
        all_handlers.update(handlers)  # user handlers take precedence
        seen = set()  # track which object id's have already been seen
        default_size = sys.getsizeof(0)  # estimate sizeof object without __sizeof__

        def sizeof(o):
            if id(o) in seen:  # do not double count the same object
                return 0
            seen.add(id(o))
            s = sys.getsizeof(o, default_size)

            for typ, handler in all_handlers.items():
                if isinstance(o, typ):
                    s += sum(map(sizeof, handler(o)))
                    break
            return s

        return sizeof(o)

    ts = _total_size(o)
    GB, MB, KB = 1024**3, 1024**2, 1024
    if ts > GB:
        return f"{ts/GB:.2f}G"
    elif ts > MB:
        return f"{ts/MB:.2f}M"
    elif ts > KB:
        return f"{ts/KB:.2f}K"
    else:
        return f"{ts}B"


def mask_select(arr, mask):
    assert len(arr) == len(mask)
    return [item for item, flag in zip(arr, mask) if flag]


def flat(arr_of_arr):
    return list(chain(*arr_of_arr))


def window(arr, k):
    for i in range(len(arr) - k):
        yield arr[i : i + k]


def batch(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i : min(i + n, len(arr))]


def len2offset(inner_arr_len):
    end = np.cumsum(inner_arr_len).tolist()
    start = [0] + end[:-1]
    return start, end


def reconstruct(seq, inner_arr_len):
    return [seq[start:end] for start, end in zip(len2offset(inner_arr_len))]


def padding(arr, pad_length, pad_sign):
    pad_size = pad_length - len(arr)
    return arr + [pad_sign] * pad_size


def get_io_spans(labels):
    spans = []
    span_count = Counter()

    i = 0
    while i < len(labels):
        if labels[i] != "O":
            start = i
            current_label = labels[i]
            i += 1
            while i < len(labels) and labels[i] == current_label:
                i += 1
            spans.append((start, i - 1, current_label))
            span_count[current_label] += 1
        else:
            i += 1
    return tuple(spans), span_count


def tag_label_split(tag_label):
    if tag_label == "O":
        tag, label = tag_label, tag_label
    else:
        # B-creative-work does not work for labels[i].split('-')
        assert tag_label[0] in ["B", "I"] and tag_label[1] == "-", tag_label
        tag, label = tag_label[0], tag_label[2:]
        assert label != "" and label != "O"
    return tag, label


def get_bio_spans(labels):
    # B-CMT I-CMT O O B-MAT
    spans = []
    span_count = Counter()

    tag_label_tuple = list(map(tag_label_split, labels))

    i = 0
    while i < len(tag_label_tuple):
        tag, label = tag_label_tuple[i]

        if tag == "B":
            start = i
            current_label = label
            i += 1
            while i < len(labels):
                tag, label = tag_label_tuple[i]
                if tag == "I" and label == current_label:
                    i += 1
                else:
                    break
            spans.append((start, i - 1, current_label))
            span_count[current_label] += 1
        else:
            i += 1

    return tuple(spans), span_count


def valid_bio_seq(labels):
    tag_label_tuple = list(map(tag_label_split, labels))

    i = 0
    prev_tag, prev_label = "O", "O"
    while i < len(tag_label_tuple):
        tag, label = tag_label_tuple[i]

        if prev_tag == "O":
            if tag == "I":
                raise RuntimeError("entity start with I")
        elif prev_tag == "B":
            if tag == "I" and prev_label != label:
                raise RuntimeError("entity start with I")
        elif prev_tag == "I":
            if tag == "I" and prev_label != label:
                raise RuntimeError("entity start with I")

        prev_tag, prev_label = tag, label
        i += 1

    return True


def cmp_bio_io_span(bio_ys, io_ys):
    io_spans, _ = get_io_spans(io_ys)
    bio_spans, _ = get_bio_spans(bio_ys)
    if io_spans != bio_spans:
        for i, j, l in bio_spans:
            if (i, j, l) not in io_spans:
                print(f"({i},{j},{l}) is missing from bio")
        for i, j, l in io_spans:
            if (i, j, l) not in bio_spans:
                print(f"({i},{j},{l}) is added to io")


def convert_bio2io(bio_ys):
    valid_bio_seq(bio_ys)

    io_ys = []
    for i, y in enumerate(bio_ys):
        if y == "O":
            io_ys.append(y)
        else:
            tag, label = tag_label_split(y)
            io_ys.append(label)
    assert len(io_ys) == len(bio_ys)
    cmp_bio_io_span(bio_ys, io_ys)
    return tuple(io_ys)


def convert_io2bio(io_ys):
    bio_ys = []
    for i, y in enumerate(io_ys):
        if y == "O":
            bio_ys.append(y)
        else:
            if i - 1 >= 0 and io_ys[i - 1] == y:
                bio_ys.append("I-" + y)
            else:
                bio_ys.append("B-" + y)
    assert len(io_ys) == len(bio_ys)
    cmp_bio_io_span(bio_ys, io_ys)
    return tuple(bio_ys)


def get_bioes_spans(labels):
    spans = []
    span_count = Counter()

    tag_label_tuple = list(map(tag_label_split, labels))

    i = 0
    while i < len(tag_label_tuple):
        tag, label = tag_label_tuple[i]

        if tag == "B":
            start = i
            current_label = label
            i += 1
            while i < len(labels):
                tag, label = tag_label_tuple[i]
                if tag in ["I", "E"] and label == current_label:
                    i += 1
                else:
                    break
            spans.append((start, i - 1, current_label))
            span_count[current_label] += 1
        elif tag == "S":
            spans.append((i, i, label))
            span_count[label] += 1
        else:
            i += 1

    return tuple(spans), span_count


def valid_bioes_seq(labels):
    tag_label_tuple = list(map(tag_label_split, labels))

    i = 0
    prev_tag, prev_label = "O", "O"
    while i < len(tag_label_tuple):
        tag, label = tag_label_tuple[i]

        if prev_tag == "O":
            if tag in ["I", "E"]:
                raise RuntimeError("entity start with I or E")
        elif prev_tag == "B":
            if tag in ["B", "O", "S"]:
                raise RuntimeError("B B or B O or B S")
            elif tag in ["I", "E"] and prev_label != label:
                raise RuntimeError("B I or B E with different label")
        elif prev_tag == "I":
            if tag in ["B", "O", "S"]:
                raise RuntimeError("I B or I O or I S")
            elif tag in ["I", "E"] and prev_label != label:
                raise RuntimeError("I I or I E with different label")
        elif prev_tag == "S":
            if tag in ["I", "E"]:
                raise RuntimeError("entity start with I or E")
        elif prev_tag == "E":
            if tag in ["I", "E"]:
                raise RuntimeError("entity start with I or E")

        prev_tag, prev_label = tag, label
        i += 1

    return True


def cmp_bioes_bio_span(bioes_ys, bio_ys):
    bioes_spans, _ = get_bioes_spans(bioes_ys)
    bio_spans, _ = get_bio_spans(bio_ys)
    assert bioes_spans != bio_spans, "convert error"


def convert_bioes2bio(bioes_ys):
    valid_bioes_seq(bioes_ys)

    bio_ys = []
    for i, y in enumerate(bioes_ys):
        tag, label = tag_label_split(y)
        if tag == "O":
            bio_ys.append(tag)
        else:
            if tag == "E":
                bio_ys.append("I-" + label)
            elif tag == "S":
                bio_ys.append("B-" + label)
            elif tag == "B":
                bio_ys.append(y)
            elif tag == "I":
                bio_ys.append(y)
            else:
                raise Exception(f"Invalid BIOES tag found: {tag}")
    assert len(bioes_ys) == len(bio_ys)
    cmp_bioes_bio_span(bioes_ys, bio_ys)
    return tuple(bio_ys)


def convert_bio2bioes(bio_ys):
    valid_bio_seq(bio_ys)

    bioes_ys = []
    tags, labels = list(map(tag_label_split, bio_ys))
    for i, (tag, label, y) in enumerate(zip(tags, labels, bio_ys)):
        if tag == "O":
            bioes_ys.append(tag)
        else:
            if tag == "I":  # convert to E- if next tag is not I-
                if i + 1 < len(tags) and tags[i + 1] == "I":
                    bioes_ys.append(y)
                else:
                    bioes_ys.append("E-" + label)
            elif tag == "B":  # convert to S- if next tag is not I-
                if i + 1 < len(tags) and tags[i + 1] == "I":
                    bioes_ys.append(y)
                else:
                    bioes_ys.append("S-" + label)
            else:
                raise Exception(f"Invalid BIO tag found: {tag}")
    assert len(bioes_ys) == len(bio_ys)
    cmp_bioes_bio_span(bioes_ys, bio_ys)
    return tuple(bioes_ys)


def span2io(tokens, spans):
    seq = ["O"] * len(tokens)
    for start, end, type in spans:
        for i in range(start, end + 1):
            seq[i] = type
    return tuple(seq)


def span2bio(tokens, spans):
    seq = ["O"] * len(tokens)
    for start, end, type in spans:
        seq[start] = "B-" + type
        for i in range(start + 1, end + 1):
            seq[i] = "I-" + type
    return tuple(seq)


def span2bioes(tokens, spans):
    seq = ["O"] * len(tokens)
    for start, end, type in spans:
        if start == end:
            seq[start] = "S-" + type
        else:
            seq[start] = "B-" + type
            for i in range(start + 1, end - 1):
                seq[i] = "I-" + type
            seq[end] = "E-" + type
    return tuple(seq)


def check_span_overlap(spans, include_end=True):

    def _overlap(x, y):
        a, b, *_ = spans[x]
        c, d, *_ = spans[y]
        return (not (b < c or a > d)) if include_end else (not (b <= c or a >= d))

    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if _overlap(i, j):
                return True
    return False


class BaseLabelEncoder:
    ignore_label = "##Ignore##"
    ignore_label_idx = -100

    # guarantee to follow the same order as labels
    def __init__(self, labels):
        self.idx_to_item = {self.ignore_label_idx: self.ignore_label}
        self.item_to_idx = {self.ignore_label: self.ignore_label_idx}
        self.labels = labels
        for i, l in enumerate(labels):
            self.add(l, i)

    def add(self, item, idx):
        assert item not in self.item_to_idx
        self.item_to_idx[item] = idx
        self.idx_to_item[idx] = item

    def decode(self, idx):
        if isinstance(idx, (list, tuple)):
            return type(idx)(self.decode(i) for i in idx)
        else:
            return self.idx_to_item[idx]

    def encode(self, item):
        if isinstance(item, (list, tuple)):
            return type(item)(self.encode(i) for i in item)
        else:
            return self.item_to_idx[item]

    def filter(self, item):
        return [i for i in item if i != self.ignore_label]

    def __len__(self):
        # only report valid idx
        return len(self.item_to_idx)

    def __str__(self):
        return f"label: {self.label}"

    def __repr__(self):
        return f"label: {self.label} mapping: {self.item_to_idx}"

    @classmethod
    def from_dict(cls, input_dict):
        new = cls(labels=input_dict["labels"])
        assert new.idx_to_item == input_dict["idx_to_item"]
        assert new.item_to_idx == input_dict["item_to_idx"]
        return new

    def to_dict(self):
        output_dict = {
            "idx_to_item": self.idx_to_item,
            "item_to_idx": self.item_to_idx,
            "labels": self.labels,
        }
        return output_dict

    @classmethod
    def from_json(cls, input_json):
        return cls.from_dict(json.loads(input_json))

    def to_json(self):
        return json.dumps(self.to_dict())


class LabelEncoder(BaseLabelEncoder):
    default_label = "O"
    default_label_idx = 0

    # guarantee to follow the same order as labels
    def __init__(self, labels, mode):
        self.idx_to_item = {self.ignore_label_idx: self.ignore_label, 0: "O"}
        self.item_to_idx = {self.ignore_label: self.ignore_label_idx, "O": 0}
        self.mode = mode
        self.labels = labels
        if mode == "io":
            for i, l in enumerate(labels):
                self.add(l, i + 1)
        elif mode == "bio":
            for i, l in enumerate(labels):
                self.add(f"B-{l}", 2 * i + 1)
                self.add(f"I-{l}", 2 * i + 2)
        elif mode == "bioes":
            for i, l in enumerate(labels):
                self.add(f"B-{l}", 4 * i + 1)
                self.add(f"I-{l}", 4 * i + 2)
                self.add(f"E-{l}", 4 * i + 3)
                self.add(f"S-{l}", 4 * i + 4)
        else:
            raise RuntimeError("Unknown mode")

    @classmethod
    def io_mode_idx(cls, idx, mode):
        if type(idx) == int:
            assert idx != cls.ignore_label_idx
            if mode == "bio":
                return (idx + 1) // 2
            elif mode == "bioes":
                return (idx + 3) // 4
            elif mode == "io":
                return idx
            else:
                raise RuntimeError("Unknown mode")
        elif type(idx) == torch.Tensor:
            if mode == "bio":
                # round towards zero
                return torch.div((idx + 1), 2, rounding_mode="trunc")
            elif mode == "bioes":
                return torch.div((idx + 3), 4, rounding_mode="trunc")
            elif mode == "io":
                return idx
            else:
                raise RuntimeError("Unknown mode")
        else:
            raise RuntimeError("Unknown idx input type")

    def __len__(self):
        # only report valid idx
        return len(self.item_to_idx) - 1

    def __str__(self):
        return f"mode: {self.mode} label: {self.label}"

    def __repr__(self):
        return f"mode: {self.mode} label: {self.label} mapping: {self.item_to_idx}"

    @classmethod
    def from_dict(cls, input_dict):
        new = cls(labels=input_dict["labels"], mode=input_dict["mode"])
        assert new.idx_to_item == input_dict["idx_to_item"]
        assert new.item_to_idx == input_dict["item_to_idx"]
        return new

    def to_dict(self):
        output_dict = {
            "idx_to_item": self.idx_to_item,
            "item_to_idx": self.item_to_idx,
            "mode": self.mode,
            "labels": self.labels,
        }
        return output_dict


class ShiftLabelEncoder(BaseLabelEncoder):
    default_label = "O"
    # guarantee to follow the same order as labels
    def __init__(self, labels, mode, shift):
        self.shift = shift
        self.default_label_idx = shift

        self.idx_to_item = {self.ignore_label_idx: self.ignore_label,  self.shift: "O"}
        self.item_to_idx = {self.ignore_label: self.ignore_label_idx, "O":  self.shift}
        self.mode = mode
        self.labels = labels
        if mode == "io":
            for i, l in enumerate(labels):
                self.add(l, i + 1 + self.shift)
        elif mode == "bio":
            for i, l in enumerate(labels):
                self.add(f"B-{l}", 2 * i + 1 + self.shift)
                self.add(f"I-{l}", 2 * i + 2 + self.shift)
        elif mode == "bioes":
            for i, l in enumerate(labels):
                self.add(f"B-{l}", 4 * i + 1 + self.shift)
                self.add(f"I-{l}", 4 * i + 2 + self.shift)
                self.add(f"E-{l}", 4 * i + 3 + self.shift)
                self.add(f"S-{l}", 4 * i + 4 + self.shift)
        else:
            raise RuntimeError("Unknown mode")

    
    @classmethod
    def cal_encoder_len(self, labels, mode):
        if mode == "bio":
            return 2 * len(labels) + 1
        elif mode == "bioes":
            return 4 * len(labels) + 1
        elif mode == "io":
            return len(labels) + 1
        else:
            raise RuntimeError("Unknown mode")
    
    def __len__(self):
        # only report valid idx
        return len(self.item_to_idx) - 1
    
    def __str__(self):
        return f"mode: {self.mode} shift: {self.shift} label: {self.labels}"

    def __repr__(self):
        return f"mode: {self.mode} shift: {self.shift} label: {self.labels} mapping: {self.item_to_idx}"

    @classmethod
    def from_dict(cls, input_dict):
        new = cls(labels=input_dict["labels"], mode=input_dict["mode"], shift=input_dict["shift"])
        assert new.idx_to_item == input_dict["idx_to_item"]
        assert new.item_to_idx == input_dict["item_to_idx"]
        return new

    def to_dict(self):
        output_dict = {
            "idx_to_item": self.idx_to_item,
            "item_to_idx": self.item_to_idx,
            "mode": self.mode,
            "labels": self.labels,
            "shift": self.shift,
        }
        return output_dict


def list_padding(list_of_list, pad_sign):
    sizes = [len(outer_list) for outer_list in list_of_list]
    max_size = max(sizes)
    pad_sizes = [max_size - size for size in sizes]
    return [outer_list + [pad_sign] * pad_size for outer_list, pad_size in zip(list_of_list, pad_sizes)]


class CDEBertTokenizer:
    def __init__(self, backbone: str):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "uncased" in backbone:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(backbone, do_lower_case=True)
            assert self.bert_tokenizer.do_lower_case == True
        elif "cased" in backbone:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(backbone, do_lower_case=False)
            assert self.bert_tokenizer.do_lower_case == False
        else:
            raise RuntimeError("unknown casing backbone")
        self.word_tokenizer = ChemWordTokenizer()
        self.sentence_tokenizer = ChemSentenceTokenizer()

    def sentence_tokenize(self, text, return_offset=False):
        texts, offsets = [], []
        for span in self.sentence_tokenizer.span_tokenize(text):
            texts.append(text[span[0] : span[1]])
            offsets.append(span)

        if return_offset:
            return tuple(texts), tuple(offsets)
        else:
            return tuple(texts)

    # we need special treatment for roberta which use gpt2 tokenizer
    # gpt2 tokenizertreat spaces like parts of the tokens so a word will
    # be encoded differently whether it is at the beginning of the sentence (without space) or not
    # ref:https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/gpt2#transformers.GPT2Tokenizer.example
    # flair example https://github.com/flairNLP/flair/blob/093f73a68f19507113e3aca29a27540b46c27193/flair/embeddings.py#L1100
    def _subword_tokenize(self, text: str):
        result = self.bert_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        subword_ids = result.input_ids
        subwords = self.bert_tokenizer.convert_ids_to_tokens(subword_ids)
        offsets = result.offset_mapping

        # for s, (start, end) in zip(subwords, offsets):
        #     if s.startswith('##'): s = s.replace('##', '')
        #     if len(s) != (end - start):
        #         print(s, self.bert_tokenizer.tokenize(text[start:end]))

        # most of time we assume subwords == text[offset], but if subwords is [UNK]
        # they do not match, however we only use offset when build sentence

        return tuple(subwords), tuple(offsets)

    @staticmethod
    def is_subword_word_start(token: str):
        return not token.startswith("##")

    def subword_tokenize(self, text, sent_tokenize=True):
        # it's tokenizer' duty to ensure text and offset are mapped correctly
        if sent_tokenize:
            # we do not need seperate sentence boundary since we always use the
            # the first and last token to infer the boundary
            result = []
            sent_texts, sent_offsets = self.sentence_tokenize(text, True)
            for sent_text, (sent_start, sent_end) in zip(sent_texts, sent_offsets):
                subwords, offsets = self._subword_tokenize(sent_text, True)
                global_offset = tuple((s + sent_start, e + sent_start) for s, e in offsets)
                assert all(
                    sent_text[offsets[i][0] : offsets[i][1]] == text[global_offset[i][0] : global_offset[i][1]]
                    for i in range(len(offsets))
                )
                result.append((subwords, global_offset))
            return result
        else:
            result = self._subword_tokenize(text)
        return result


    def _word_tokenize(self, text, return_offset=False):
        texts, offsets = [], []
        for span in self.word_tokenizer.span_tokenize(text):
            texts.append(text[span[0]:span[1]])
            offsets.append(span)

        if return_offset:
            return tuple(texts), tuple(offsets)
        else:
            return tuple(texts)
        
    def word_tokenize(self, text, sent_tokenize=True):
        # it's tokenizer' duty to ensure text and offset are mapped correctly
        if sent_tokenize:
            result = ([], [])
            global_offset = 0
            # we do not need seperate sentence boundary since we always use the
            # the first and last token to infer the boundary
            sent_texts, sent_offsets = self.sentence_tokenize(text, True)
            for sent_text, (sent_start, sent_end) in zip(sent_texts, sent_offsets):
                texts, offsets = self._word_tokenize(sent_text, True)
                result[0].append(texts)
                global_offset = tuple((s + sent_start, e + sent_start) for s, e in offsets)
                assert all(texts[i] == text[global_offset[i][0]:global_offset[i][1]] for i in range(len(texts)))
                result[1].append(global_offset)
        else:
            result = self._word_tokenize(text, True)
        return result

@dataclass
class Token:
    text: str
    start: Optional[int] = 0
    label: Optional[str] = None
    subword: Optional[str] = None

    def __str__(self):
        return f"{self.text}({self.start},{self.end})"

    @property
    def end(self):
        return self.start + len(self.text)

    def __len__(self):
        return len(self.text)

    @classmethod
    def from_dict(cls, input_dict):
        # json does not support None as key in dict, Python does
        # but we should not have None in dict keys
        return cls(
            text=input_dict["text"],
            start=input_dict.get("start", 0),
            label=input_dict.get("label", None),
            subword=input_dict.get("subword", None),
        )

    def to_dict(self):
        output_dict = {"text": self.text}
        if self.start is not None:
            output_dict["start"] = self.start
        if self.label is not None:
            output_dict["label"] = self.label
        if self.subword is not None:
            output_dict["subword"] = self.subword
        return output_dict

    @classmethod
    def from_json(cls, input_json):
        return cls.from_dict(json.loads(input_json))

    def to_json(self):
        return json.dumps(self.to_dict())


@dataclass
class Sentence:
    text: str
    start: Optional[int] = 0
    id: Optional[str] = None
    tokens: Optional[List[Token]] = field(default=None, init=False)
    mode: Optional[str] = field(default=None, init=False)
    # entity token idx offset (start, end, label)
    entity: Optional[List[Tuple[int, int, str]]] = field(default=None, init=False)
    # entity character idx offset (start, end, label)
    entity_char_offset: Optional[List[Tuple[int, int, str]]] = field(default=None, init=False)
    entity_counter: Optional[Counter] = field(default=None, init=False)

    def __str__(self):
        return f"({self.start},{self.end})\t{self.text}"

    @property
    def end(self):
        return self.start + len(self.text)

    def __len__(self):
        return len(self.text)

    @classmethod
    def from_texts(cls, texts, start=0, labels=None):
        # mainly used for tokenized text
        text = " ".join(texts)
        spans = []
        offset = start
        for i, word in enumerate(texts):
            assert text[offset : offset + len(word)] == word
            spans.append((offset, offset + len(word)))
            offset += len(word) + 1
        new = cls(text, start)
        new.add_token_offset(spans)
        if labels:
            assert len(texts) == len(labels)
            new.labels = labels
        return new

    @staticmethod
    def reconstruct_text_from_offsets(texts, spans):
        # mainly used for tokenized text
        text, text_len = [], 0
        for word, (s, e) in zip(texts, spans):
            assert len(word) == e - s, ([ord(c) for c in word], s, e)
            if text_len < s:
                text.append(" " * (s - text_len))
                text_len += s - text_len
            text.append(word)
            text_len += len(word)
        return "".join(text)

    def add_token_offset(self, spans):
        # mainly used for raw text and need a tokenizer to split it
        # it's tokenizer' duty to ensure text and offset are mapped correctly
        # assume tokenizer always return offset that starts from 0
        prev_end = -1
        self.tokens = []
        for _start, _end in spans:
            # global start end offset
            start, end = _start + self.start, _end + self.start
            assert start >= self.start and end <= self.end and prev_end <= start
            self.tokens.append(Token(self.text[_start:_end], start))
            prev_end = end

    def split_token(self, i, offset):
        token = self.tokens[i]
        assert 0 < offset < len(token.text) or -len(token.text) < offset < 0, (token, offset)
        left_token = Token(token.text[:offset], token.start)
        if offset > 0:
            right_token = Token(token.text[offset:], token.start + offset)
        else:
            right_token = Token(token.text[offset:], token.end + offset)
        self.tokens[i] = left_token
        self.tokens.insert(i + 1, right_token)

    def merge_token(self, i):
        left_token = self.tokens[i]
        right_token = self.tokens[i + 1]
        empty = right_token.start - left_token.end
        token = Token(left_token.text + ' ' * empty + right_token.text, left_token.start)
        self.tokens.pop(i + 1)
        self.tokens[i] = token

    @property
    def token_texts(self):
        # Return token text inside the Sentence
        return [token.text for token in self.tokens]

    @property
    def labels(self):
        # Return token label inside the Sentence
        return [token.label for token in self.tokens]

    @labels.setter
    def labels(self, labels):
        # set token label inside the Sentence
        for t, l in zip(self.tokens, labels):
            t.label = l

    @property
    def subwords(self):
        # Return token label inside the Sentence
        return [token.subword for token in self.tokens]

    def add_subword_info(self, subwords):
        # set token label inside the Sentence
        for t, w in zip(self.tokens, subwords):
            t.subword = w

    def add_entity_char_offset(self, spans, nooverlap=False):
        # save character offset entity position as attibute
        self.entity_char_offset = []
        entity_pos = defaultdict(list)

        for start, end, text, label in spans:
            assert start >= self.start
            assert end <= self.end
            assert self.text[(start - self.start) : (end - self.start)] == text
            # Create a entity_pos acording with the spans
            if (start, end) not in entity_pos:
                entity_pos[(start, end)].append(label)
                self.entity_char_offset.append((start, end, text, label))
            # if the same we ignore
            elif len(entity_pos[(start, end)]) == 1 and label == entity_pos[(start, end)][0]:
                continue
            else:
                if not nooverlap:
                    print(f"entity overlap {start} {end} {label} {entity_pos[(start, end)]}")
                    entity_pos[(start, end)].append(label)
                else:
                    raise RuntimeError(f"entity overlap {start} {end} {label} {entity_pos[(start, end)]}")

    def split_at_char_offset(self):
        #Review each Token and split acoriding with character offset
        splits = 0
        i = 0
        while i < len(self.tokens):
            token = self.tokens[i]
            changed = False
            for start, end, text, label in self.entity_char_offset:
                try:
                    if token.start < start < token.end:
                        offset = start - token.start
                        self.split_token(i, offset)
                        splits += 1
                        # print(f'token split {token} at {offset}')
                        changed = True
                        break
                    # cannot happen together
                    elif token.start < end < token.end:
                        offset = end - token.start
                        self.split_token(i, offset)
                        splits += 1
                        # print(f'token split {token} at {offset}')
                        changed = True
                        break
                except AssertionError as e:
                    print(token)
                    print(f'charoffset({start}, {end}):{text}\t{label}')
                    raise e
            if not changed:
                i += 1
        return splits

    def match_charoffset(self, start, end, mode="strict"):
        start_idx, end_idx = None, None
        min_start, max_end = None, None
        for idx, token in enumerate(self.tokens):
            if start == token.start:
                start_idx = idx
            elif start > token.start:
                min_start = idx
        for idx, token in reversed(list(enumerate(self.tokens))):
            if end == token.end:
                end_idx = idx
            elif end < token.end:
                max_end = idx
        if start_idx is None:
            if mode == "strict":
                raise RuntimeError("start")
            elif mode == "include":
                start_idx = min_start
            elif mode == "ignore":
                pass
            else:
                raise RuntimeError("unknown convert mode")
        if end_idx is None:
            if mode == "strict":
                raise RuntimeError("end")
            elif mode == "include":
                end_idx = max_end
            elif mode == "ignore":
                pass
            else:
                raise RuntimeError("unknown convert mode")

        return start_idx, end_idx

    def strip_entity_char_offset(self, start, end, text):
        i = start
        while i < end and text[i - start].isspace():
            i += 1

        j = end - 1
        while j >= start and text[j - start].isspace():
            j -= 1

        if i > start or j < end - 1:
            text_strip = text.strip()
            assert i < j + 1 and text[i - start : j - start + 1] == text_strip
            return i, j + 1, text_strip
        else:
            return start, end, text

    def convert_entity_char_offset(self, mode="strict", nooverlap=False):
        # convert character offset to token labels
        # Initialize self.entity and self.entity_counter
        self.entity, self.entity_counter = [], Counter()
        entity_pos = defaultdict(list)
        for start, end, text, label in self.entity_char_offset:
            start, end, text = self.strip_entity_char_offset(start, end, text)
            start_idx, end_idx = self.match_charoffset(start, end, mode)
            if start_idx is None or end_idx is None:
                print(f"ignore offset ({start}, {end}, {text})")
                continue

            if start != self.tokens[start_idx].start or end != self.tokens[end_idx].end:
                extended_start, extended_end = self.tokens[start_idx].start, self.tokens[end_idx].end
                extened_text = self.text[extended_start:extended_end]
                print(f"extend offset ({start}, {end}, {text}) to ({extended_start}, {extended_end}, {extened_text})")

            assert start_idx >= 0
            assert start_idx <= end_idx <= len(self.tokens)
            if (start_idx, end_idx) not in entity_pos:
                entity_pos[(start_idx, end_idx)].append(label)
                self.entity.append((start_idx, end_idx, label))
                self.entity_counter[label] += 1
            # if the same we ignore
            elif len(entity_pos[(start_idx, end_idx)]) == 1 and label == entity_pos[(start_idx, end_idx)][0]:
                continue
            else:
                if not nooverlap:
                    # use the first label in this position
                    print(f"entity overlap {start_idx} {end_idx} {label} {entity_pos[(start_idx, end_idx)]}")
                    entity_pos[(start_idx, end_idx)].append(label)
                else:
                    raise RuntimeError(f"entity overlap {start_idx} {end_idx} {label} {entity_pos[(start_idx, end_idx)]}")

    def add_entity_token_idx(self, spans, nooverlap=False):
        # save token index entity position as attibute
        self.entity, self.entity_counter = [], Counter()
        entity_pos = defaultdict(list)
        for start, end, label in spans:
            assert start >= 0
            assert start <= end <= len(self.tokens)
            if (start, end) not in entity_pos:
                entity_pos[(start, end)].append(label)
                self.entity.append((start, end, label))
                self.entity_counter[label] += 1
            # if the same we ignore
            elif len(entity_pos[(start, end)]) == 1 and label == entity_pos[(start, end)][0]:
                continue
            else:
                if not nooverlap:
                    print(f"entity overlap {start} {end} {label} {entity_pos[(start, end)]}")
                    entity_pos[(start, end)].append(label)
                else:
                    raise RuntimeError(f"entity overlap {start} {end} {label} {entity_pos[(start, end)]}")

    def convert_entity_position(self, mode):
        # Generate token labels acording with the mode (bio/bios/io)
        self.mode = mode
        if mode == "bio":
            self.labels = span2bio(self.tokens, self.entity)
        elif mode == "bioes":
            self.labels = span2bioes(self.tokens, self.entity)
        elif mode == "io":
            self.labels = span2io(self.tokens, self.entity)
        else:
            raise RuntimeError("Unknown label mode")

    def add_entity_token_label(self, mode, labels=None):
        # Generate the labels acording with the mode (bio/bios/io)
        # Acording with the spans generate the labels and count how many ther is in the text
        self.mode = mode
        if not labels:
            labels = [t.label for t in self.tokens]
            assert not any([l is None for l in labels])
        assert len(labels) == len(self.tokens)
        self.labels = labels
        if mode == "bio":
            self.entity, self.entity_counter = get_bio_spans(labels)
        elif mode == "bioes":
            self.entity, self.entity_counter = get_bioes_spans(labels)
        elif mode == "io":
            self.entity, self.entity_counter = get_io_spans(labels)
        else:
            raise RuntimeError("Unknown label mode")

    @classmethod
    def from_dict(cls, input_dict):
        # json does not support None as key in dict, Python does
        # but we should not have None in dict keys
        new = cls(text=input_dict["text"], start=input_dict["start"], id=input_dict["id"])
        if input_dict["tokens"] is not None:
            new.tokens = [Token.from_dict(td) for td in input_dict["tokens"]]
        new.mode = input_dict["mode"]
        new.entity = input_dict["entity"]
        new.entity_char_offset = input_dict["entity_char_offset"]
        if input_dict["entity_counter"] is not None:
            new.entity_counter = Counter({k: v for k, v in input_dict["entity_counter"]})
        return new

    def to_dict(self):
        output_dict = {
            "text": self.text,
            "start": self.start,
            "id": self.id,
        }
        if self.tokens is not None:
            output_dict["tokens"] = [Token.to_dict(td) for td in self.tokens]
        else:
            output_dict["tokens"] = None
        output_dict["mode"] = self.mode
        output_dict["entity"] = self.entity
        output_dict["entity_char_offset"] = self.entity_char_offset
        if self.entity_counter is not None:
            output_dict["entity_counter"] = list(self.entity_counter.items())
        else:
            output_dict["entity_counter"] = None
        return output_dict

    @classmethod
    def ds_feature(cls):
        return Features(
            {
                "tokens": [Value(dtype="string")],
                "labels": [Value(dtype="string")],
                "subwords": [Value(dtype="string")],
                "entity_start": [Value(dtype="int64")],
                "entity_end": [Value(dtype="int64")],
                "entity_label": [Value(dtype="string")],
                "mode": Value(dtype="string"),
            }
        )

    def to_feature(self):
        assert self.tokens is not None and self.entity is not None and self.mode is not None
        assert all(l is not None for l in self.labels)
        entity_start, entity_end, entity_label = [], [], []
        for start, end, label in self.entity:
            entity_start.append(start)
            entity_end.append(end)
            entity_label.append(label)
        return {
            "tokens": self.token_texts,
            "labels": self.labels,
            "subwords": self.subwords,
            "entity_start": entity_start,
            "entity_end": entity_end,
            "entity_label": entity_label,
            "mode": self.mode,
        }

    @classmethod
    def from_json(cls, input_json):
        return cls.from_dict(json.loads(input_json))

    def to_json(self):
        return json.dumps(self.to_dict())


class Preprocess:

    def __init__(
        self,
        bert_tokenizer: BertTokenizerFast,
        label_encoder: LabelEncoder,
        span_label_encoder: LabelEncoder,
        max_length: int = 510,
        max_span_length: int = 10,
        neg_rate: int = 1000,
        context: int = 0,
    ):
        self.bert_tokenizer = bert_tokenizer
        self.label_encoder = label_encoder
        self.span_label_encoder = span_label_encoder
        assert context >= 0 and max_length >= 0, "neg context or max_length"
        self.context = context
        assert max_length + context + 2 <= 512, "too long"
        assert max_length >= 64, "too short"
        self.max_length = max_length
        self.max_span_length = max_span_length
        self.neg_rate = neg_rate

    def word_pad(self, ids: List[int]):
        assert len(ids) <= 512
        return padding(ids, 512, "[PAD]")

    def subword_pad_np(self, ids: List[int]):
        assert len(ids) <= 512
        return np.asarray(padding(ids, 512, self.bert_tokenizer.pad_token_id))

    def label_pad_np(self, labels: List[int]):
        assert len(labels) <= 512
        label_ids = self.label_encoder.encode(labels)
        return np.asarray(padding(label_ids, 512, self.label_encoder.ignore_label_idx))

    def zero_pad_np(self, ids: List[int]):
        assert len(ids) <= 512
        return np.asarray(padding(ids, 512, 0))

    def subword_to_dict(self, subwords: List[List[str]], labels: List[str], mask_left: int, mask_right: int):
        word_len = [len(t) for t in subwords]
        flat_subwords = flat(subwords)
        input_ids = self.bert_tokenizer.build_inputs_with_special_tokens(
            self.bert_tokenizer.convert_tokens_to_ids(flat_subwords)
        )
        input_labels = (
            [LabelEncoder.ignore_label]
            + flat([[l] * wl for l, wl in zip(labels, word_len)])
            + [LabelEncoder.ignore_label]
        )
        first_mask = flat([[1] + [0] * (wl - 1) for wl in word_len])
        last_mask = flat([[0] * (wl - 1) + [1] for wl in word_len])
        atten_masks = [1] * len(input_ids)
        for i in range(mask_left):
            first_mask[i] = 0
            last_mask[i] = 0
        for i in range(mask_right):
            first_mask[-(i + 1)] = 0
            last_mask[-(i + 1)] = 0
        first_mask = [0] + first_mask + [0]
        last_mask = [0] + last_mask + [0]
        assert len(input_ids) == len(input_labels) == len(first_mask) == len(last_mask) == len(atten_masks)
        min_len = len(input_ids)

        # gather_idx = [idx for idx, flag in enumerate(word_masks) if flag > 0]
        # scatter_idx = flat([[idx] * wl for idx, wl in enumerate(word_len)])

        return {
            "input_id": self.subword_pad_np(input_ids),
            "input_label": self.label_pad_np(input_labels),
            "raw_id": flat_subwords,
            "raw_label": input_labels,
            "word_len": word_len,
            "first_mask": self.zero_pad_np(first_mask),
            "last_mask": self.zero_pad_np(last_mask),
            "atten_mask": self.zero_pad_np(atten_masks),
            "min_len": min_len,
            "gather_idx": self.zero_pad_np([idx for idx, flag in enumerate(first_mask) if flag > 0]),
            "gather_end_idx": self.zero_pad_np([idx for idx, flag in enumerate(last_mask) if flag > 0]),
            "gather_label": self.label_pad_np(mask_select(input_labels, first_mask)),
        }

    def localize_spans(self, spans: List[Tuple[int, int, str]], context_start: int, context_end: int):
        new_spans = []
        for start, end, label in spans:
            if context_start <= start <= end <= context_end:
                new_start = start - context_start
                new_end = end - context_start
                new_spans.append((new_start, new_end, label))
        return new_spans

    def span_masking(self, spans: List[Tuple[int, int, str]], subword_len: List[int]):
        pos_span = [
            (start, end, self.span_label_encoder.encode(label))
            for start, end, label in spans
            if end - start + 1 <= self.max_span_length
        ]
        reject_set = set([(start, end) for start, end, label in spans])

        neg_span = []
        for i in range(len(subword_len)):
            for j in range(i, len(subword_len)):
                if j - i + 1 <= self.max_span_length and (i, j) not in reject_set:
                    neg_span.append((i, j, self.span_label_encoder.default_label_idx))

        if len(neg_span) > 0:
            neg_num = int(len(subword_len) * self.neg_rate) + 1
            sample_num = min(neg_num, len(neg_span))
            sampled_neg_span_mask = [True] * sample_num + [False] * (len(neg_span) - sample_num)
            assert len(sampled_neg_span_mask) == len(neg_span)
            random.shuffle(sampled_neg_span_mask)
        else:
            sampled_neg_span_mask = [True] * len(neg_span)

        max_flat_len = 512 * self.max_span_length
        gather_start = np.zeros((max_flat_len), dtype=np.int64)
        gather_end = np.zeros((max_flat_len), dtype=np.int64)
        full_span = np.ones((max_flat_len), dtype=np.int64) * self.span_label_encoder.ignore_label_idx
        sample_span = np.ones((max_flat_len), dtype=np.int64) * self.span_label_encoder.ignore_label_idx
        span_sub_len = np.zeros((max_flat_len), dtype=np.int64)
        substart, subend = len2offset(subword_len)
        idx = 0
        for i, j, l in pos_span:
            gather_start[idx] = i
            gather_end[idx] = j
            full_span[idx] = l
            sample_span[idx] = l
            span_sub_len[idx] = subend[j] - substart[i]
            idx += 1
        for (i, j, l), flag in zip(neg_span, sampled_neg_span_mask):
            gather_start[idx] = i
            gather_end[idx] = j
            full_span[idx] = l
            if flag:
                sample_span[idx] = l
            span_sub_len[idx] = subend[j] - substart[i]
            idx += 1

        # cannot do this since this will make different sentence has different lens span arr and cannot stack
        # full_span_x, full_span_y = np.nonzero(full_span != self.span_label_encoder.ignore_label_idx)
        # valid_full_span_label = full_span[full_span_x, full_span_y]
        # valid_sample_mask = sample_mask[full_span_x, full_span_y]

        return {
            "min_len": idx,
            "gather_start": gather_start[np.newaxis, :],
            "gather_end": gather_end[np.newaxis, :],
            "full_span": full_span[np.newaxis, :],
            "sample_span": sample_span[np.newaxis, :],
            "subword_len": span_sub_len[np.newaxis, :],
            "origin_span": spans,
            "pos_span": [(i, j, l) for i, j, l in spans if j - i + 1 <= self.max_span_length],
        }

    def _split(self, input_ids: List[Dict], labels: List[str], spans: List, start: int, end: int):
        # end is not included
        # we extend buf to include more words in left/right so that the mode can take more context
        # these addtitional word is ignored during loss calculation and label prediction
        sample_tokens = [input_ids[i] for i in range(start, end)]
        sample_labels = [labels[i] for i in range(start, end)]
        cur_start, cur_end = start, end - 1  # in words
        non_context_start, non_context_end = cur_start, cur_end  # save before change,  in words
        left_context, right_context, context_left = 0, 0, self.context  # in subwords
        while context_left > 0:
            if cur_start - 1 >= 0 and context_left >= len(input_ids[cur_start - 1]["bert"]):
                sample_tokens.insert(0, input_ids[cur_start - 1])
                sample_labels.insert(0, labels[cur_start - 1])
                cur_start -= 1
                left_context += len(input_ids[cur_start]["bert"])
                context_left -= len(input_ids[cur_start]["bert"])
            elif cur_end + 1 < len(input_ids) and context_left >= len(input_ids[cur_end + 1]["bert"]):
                sample_tokens.append(input_ids[cur_end + 1])
                sample_labels.append(labels[cur_end + 1])
                cur_end += 1
                right_context += len(input_ids[cur_end]["bert"])
                context_left -= len(input_ids[cur_end]["bert"])
            else:
                break

        sample = {
            "start": non_context_start,
            "end": non_context_end + 1,
            "context_start": cur_start,
            "context_end": cur_end + 1,
        }

        assert sample_tokens == input_ids[sample["context_start"] : sample["context_end"]]
        assert sample_labels == labels[sample["context_start"] : sample["context_end"]]
        sample["gather"] = {"min_len": sample["end"] - sample["start"]}

        sample["bert"] = self.subword_to_dict(
            [t["bert"] for t in sample_tokens], sample_labels, left_context, right_context
        )
        assert labels[sample["start"] : sample["end"]] == mask_select(
            self.label_encoder.decode(sample["bert"]["input_label"].tolist()), sample["bert"]["first_mask"].tolist()
        )

        assert sample["bert"]["first_mask"].sum().item() == sample["gather"]["min_len"]
        sample["gather"]["bert"] = sample["bert"].pop("gather_idx")
        sample["gather"]["bert_end"] = sample["bert"].pop("gather_end_idx")
        sample["gather"]["label"] = sample["bert"].pop("gather_label")
        assert labels[sample["start"] : sample["end"]] == self.label_encoder.filter(
            self.label_encoder.decode(sample["gather"]["label"].tolist())
        )

        sample_span = self.localize_spans(spans, non_context_start, non_context_end)
        subword_len = [len(t["bert"]) for t in input_ids[sample["start"] : sample["end"]]]
        sample["span"] = self.span_masking(sample_span, subword_len)
        return sample

    def prepare_subword(self, subwords: List[str], labels: List[str], spans: List[Tuple[int, int, str]]):
        assert len(subwords) == len(labels), "token and label have different length"

        def raw_subwords():
            res = []
            for i, subword in enumerate(subwords):
                res.append({"bert": [subword]})
            return res

        def merge_subwords():
            words = []
            word = None
            for i, subword in enumerate(subwords):
                if word is None:
                    word = {"bert": [subword], "bert_id": [i]}
                elif not CDEBertTokenizer.is_subword_word_start(subword):
                    word["bert"].append(subword)
                    word["bert_id"].append(i)
                else:
                    words.append(word)
                    word = {"bert": [subword], "bert_id": [i]}

            words.append(word)
            return words

        def align_span(words):
            new_span = []
            for start, end, label in spans:
                new_word_start = None
                new_word_end = None
                for word_i, word in enumerate(words):
                    if start in word["bert_id"]:
                        assert new_word_start is None
                        if start != word["bert_id"][0]:
                            offset = word["bert_id"].index(start)
                            assert 0 < offset < len(word["bert"]), (word, offset)
                            left_token = {"bert": word["bert"][:offset], "bert_id": word["bert_id"][:offset]}
                            right_token = {"bert": word["bert"][offset:], "bert_id": word["bert_id"][offset:]}
                            print(
                                word["bert"],
                                "split because span in the middle",
                                left_token["bert"],
                                right_token["bert"],
                            )
                            words[word_i] = left_token
                            words.insert(word_i + 1, right_token)
                        else:
                            new_word_start = word_i
                    if new_word_start is not None and end in word["bert_id"]:
                        assert new_word_end is None
                        if end != word["bert_id"][-1]:
                            offset = word["bert_id"].index(end) + 1
                            assert 0 < offset < len(word["bert"]), (word, offset)
                            left_token = {"bert": word["bert"][:offset], "bert_id": word["bert_id"][:offset]}
                            right_token = {"bert": word["bert"][offset:], "bert_id": word["bert_id"][offset:]}
                            print(
                                word["bert"],
                                "split because span in the middle",
                                left_token["bert"],
                                right_token["bert"],
                            )
                            words[word_i] = left_token
                            words.insert(word_i + 1, right_token)
                            new_word_end = word_i
                        else:
                            new_word_end = word_i
                new_span.append((new_word_start, new_word_end, label))
            return new_span

        # create word from subword and align spans to word boundary
        # input_ids = merge_subwords()
        # spans = align_span(input_ids)
        # use subwords directly
        input_ids = raw_subwords()

        samples = []
        start = 0
        bert_len = 0
        for idx, t in enumerate(input_ids):
            # when a single word has more than max_length subword
            # I do not change since in chemu this indeed happens
            # a very long chemical name can have more than 64 subwords
            # this is fine as long as we truncate it to at most 20 during tokenization
            assert len(t["bert"]) < self.max_length, f"over token length limit {len(t['bert'])} {self.max_length}"
            if bert_len + len(t["bert"]) + 2 <= self.max_length:
                bert_len += len(t["bert"])
            else:
                samples.append(self._split(input_ids, labels, spans, start, idx))
                start = idx
                bert_len = len(t["bert"])
        if start < len(input_ids):
            samples.append(self._split(input_ids, labels, spans, start, len(input_ids)))
        return samples

    def prepare_word(self, words: List[str], labels: List[str], spans: List[Tuple[int, int, str]]):
        assert len(words) == len(labels), "token and label have different length"

        def subword_tokenize():
            res = []
            for i, w in enumerate(words):
                subword = self.bert_tokenizer.tokenize(w)
                # TODO check long word filter
                if len(subword) >= 15:
                    logging.debug(subword, 'long subword', len(subword))
                    subword = subword[:15]
                if len(w) >= 50:
                    logging.debug(w, 'long word', len(w))
                if len(subword) == 0:
                    subword = [self.bert_tokenizer.unk_token]

                res.append({"bert": subword, "word": w})
            return res

        input_ids = subword_tokenize()
        
        samples = []
        start = 0
        bert_len = 0
        for idx, t in enumerate(input_ids):
            # when a single word has more than max_length subword
            # I do not change since in chemu this indeed happens
            # a very long chemical name can have more than 64 subwords
            # this is fine as long as we truncate it to at most 20 during tokenization
            assert len(t["bert"]) < self.max_length, f"over token length limit {len(t['bert'])} {self.max_length}"
            if bert_len + len(t["bert"]) + 2 <= self.max_length:
                bert_len += len(t["bert"])
            else:
                samples.append(self._split(input_ids, labels, spans, start, idx))
                start = idx
                bert_len = len(t["bert"])
        if start < len(input_ids):
            samples.append(self._split(input_ids, labels, spans, start, len(input_ids)))
        return samples

class SentenceProcessor:

    def __init__(self, labels: Optional[List[str]], mode: str, args: Namespace):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "uncased" in args.backbone:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(args.backbone, do_lower_case=True)
            assert self.bert_tokenizer.do_lower_case == True
        elif "cased" in args.backbone:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(args.backbone, do_lower_case=False)
            assert self.bert_tokenizer.do_lower_case == False
        else:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(args.backbone)
            logging.info(f"bert do lower case {self.bert_tokenizer.do_lower_case}")
        self.max_length = args.max_length
        self.max_span_length = args.max_span_length
        self.neg_rate = args.neg_rate
        self.context = args.context

        # delay label encoder construction for episode usage
        if labels is not None:
            self.label_encoder = LabelEncoder(labels, mode)
            self.span_label_encoder = LabelEncoder(labels, "io")
            self.types = labels
            self.preprocesser = Preprocess(
                self.bert_tokenizer,
                self.label_encoder,
                self.span_label_encoder,
                self.max_length,
                self.max_span_length,
                self.neg_rate,
                self.context,
            )
        self.mode = mode

    # change label encoder construction for different episode
    def reinit_label(self, labels):
        self.label_encoder = LabelEncoder(labels, self.mode)
        self.span_label_encoder = LabelEncoder(labels, "io")
        self.types = labels
        self.preprocesser = Preprocess(
            self.bert_tokenizer,
            self.label_encoder,
            self.span_label_encoder,
            self.max_length,
            self.max_span_length,
            self.neg_rate,
            self.context,
        )

    def prepare_subword(self, sent_feature: Dict, sent_id: Union[int, str]):
        sent_subwords, sent_labels, sent_mode = sent_feature["subwords"], sent_feature["labels"], sent_feature["mode"]
        sent_entity = list(zip(sent_feature["entity_start"], sent_feature["entity_end"], sent_feature["entity_label"]))
        sent_samples = []
        chunk_samples = self.preprocesser.prepare_subword(sent_subwords, sent_labels, sent_entity)
        for chunk_id, sample in enumerate(chunk_samples):
            sent_data = {
                "token": sent_subwords,
                "label": sent_labels,
                "mode": sent_mode,
                "entity": sent_entity,
                "sent_id": f"s#{sent_id}",
                "chunk_id": f"c#{chunk_id}",
                "types": self.types,
            }
            sample.update(sent_data)
            sent_samples.append(sample)
        return sent_samples

    def prepare_word(self, sent_feature: Dict, sent_id: Union[int, str]):
        sent_words, sent_labels, sent_mode = sent_feature["tokens"], sent_feature["labels"], sent_feature["mode"]
        sent_entity = list(zip(sent_feature["entity_start"], sent_feature["entity_end"], sent_feature["entity_label"]))
        sent_samples = []
        chunk_samples = self.preprocesser.prepare_word(sent_words, sent_labels, sent_entity)
        for chunk_id, sample in enumerate(chunk_samples):
            sent_data = {
                "token": sent_words,
                "label": sent_labels,
                "mode": sent_mode,
                "entity": sent_entity,
                "sent_id": f"s#{sent_id}",
                "chunk_id": f"c#{chunk_id}",
                "types": self.types,
            }
            sample.update(sent_data)
            sent_samples.append(sample)
        return sent_samples


class SentenceDataset(torch.utils.data.Dataset):

    def __init__(self, loader: Iterable[Sentence], labels: List[str], mode: str, args: Namespace):
        super().__init__()
        self.preprocessor = SentenceProcessor(labels, mode, args)
        self.samples = []
        sent_len, span_len = [], []

        for sent in loader:
            assert self.preprocessor.mode == sent.mode
            for k in sent.entity_counter:
                assert k in self.preprocessor.types, f"out of type entity {k}"
            self.samples.append(sent.to_feature())
            sent_len.append(len(self.samples[-1]["subwords"]))
            for start, end, label in sent.entity:
                span_len.append(end - start + 1)
        print(
            f"1% sent len {np.percentile(sent_len, 1)}, "
            f"2% sent len {np.percentile(sent_len, 2)}, "
            f"25% sent len {np.percentile(sent_len, 25)}"
        )
        print(
            f"75% sent len {np.percentile(sent_len, 75)}, "
            f"98% sent len {np.percentile(sent_len, 98)}, "
            f"99% sent len {np.percentile(sent_len, 99)}"
        )
        print(
            f"1% span len {np.percentile(span_len, 1)}, "
            f"2% span_len {np.percentile(span_len, 2)}, "
            f"25% span_len {np.percentile(span_len, 25)}"
        )
        print(
            f"75% span_len {np.percentile(span_len, 75)}, "
            f"98% span_len {np.percentile(span_len, 98)}, "
            f"99% span_len {np.percentile(span_len, 99)}"
        )
        # we store everything inside huggingface's dataset to avoid mem copy
        self.samples = Dataset.from_list(self.samples, features=Sentence.ds_feature())
        del loader
        gc.collect()

    def __getitem__(self, index):
        sent_feature = self.samples[index]
        random.seed(index)
        return self.preprocessor.prepare_word(sent_feature, f"sent{index}")

    def __len__(self):
        return len(self.samples)

# need to be dataclass so that we can pass this to pytorch lightning
# model and call save hyperparameter
@dataclass
class MultiSentenceProcessorTypes:
    source_id: int
    source_name: str
    dataset_labels: List[str]
    mode: str
    shift: int
    span_shift: int

    @property
    def encoder_len(self):
        return ShiftLabelEncoder.cal_encoder_len(self.dataset_labels, self.mode)
    
    @property
    def span_encoder_len(self):
        return ShiftLabelEncoder.cal_encoder_len(self.dataset_labels, "io")

    def get_label_encoder(self):
        return ShiftLabelEncoder(self.dataset_labels, self.mode, self.shift)
    
    def get_span_label_encoder(self):
        return ShiftLabelEncoder(self.dataset_labels, "io", self.span_shift)


class MultiSentenceProcessor:
    def __init__(self, all_labels: Dict[str, List[str]], mode: str, args: Namespace):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "uncased" in args.backbone:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(args.backbone, do_lower_case=True)
            assert self.bert_tokenizer.do_lower_case == True
        elif "cased" in args.backbone:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(args.backbone, do_lower_case=False)
            assert self.bert_tokenizer.do_lower_case == False
        else:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(args.backbone)
            logging.info(f"bert do lower case {self.bert_tokenizer.do_lower_case}")
        self.max_length = args.max_length
        self.max_span_length = args.max_span_length
        self.neg_rate = args.neg_rate
        self.context = args.context

        self.preprocessors = {}
        self.types = {}
        offset = 0
        span_offset = 0

        self.source_label_encoder = BaseLabelEncoder(list(all_labels.keys()))
        for data_name, data_labels in all_labels.items():
            single_type = MultiSentenceProcessorTypes(
                self.source_label_encoder.encode(data_name),
                data_name,
                data_labels,
                mode,
                offset,
                span_offset
            )
            self.types[data_name] = single_type

            # we have different O for different dataset
            label_encoder = single_type.get_label_encoder()
            span_label_encoder = single_type.get_span_label_encoder()

            preprocessor = Preprocess(
                self.bert_tokenizer,
                label_encoder,
                span_label_encoder,
                self.max_length,
                self.max_span_length,
                self.neg_rate,
                self.context,
            )
            self.preprocessors[data_name] = preprocessor

            offset += len(label_encoder)
            span_offset += len(span_label_encoder)

        self.mode = mode

    def prepare_subword(self, sent_feature, dataset_index, source, source_index):
        sent_subwords, sent_labels, sent_mode = sent_feature["subwords"], sent_feature["labels"], sent_feature["mode"]
        sent_entity = list(zip(sent_feature["entity_start"], sent_feature["entity_end"], sent_feature["entity_label"]))
        sent_samples = []

        preprocessor = self.preprocessors[source]
        assert sent_mode == preprocessor.label_encoder.mode
        chunk_samples = preprocessor.prepare_subword(sent_subwords, sent_labels, sent_entity)
        for chunk_id, sample in enumerate(chunk_samples):
            sent_data = {
                "token": sent_subwords,
                "label": sent_labels,
                "entity": sent_entity,
                "sent_id": f"s#{dataset_index}",
                "chunk_id": f"c#{chunk_id}",
                "source": source,
                "source_id": self.source_label_encoder.encode(source),
                "source_idx": source_index,
                "mode": sent_mode,
            }
            sample.update(sent_data)
            sent_samples.append(sample)
        return sent_samples
    
    def prepare_word(self, sent_feature, dataset_index, source, source_index):
        sent_words, sent_labels, sent_mode = sent_feature["tokens"], sent_feature["labels"], sent_feature["mode"]
        sent_entity = list(zip(sent_feature["entity_start"], sent_feature["entity_end"], sent_feature["entity_label"]))
        sent_samples = []

        preprocessor = self.preprocessors[source]
        assert sent_mode == preprocessor.label_encoder.mode
        chunk_samples = preprocessor.prepare_word(sent_words, sent_labels, sent_entity)
        for chunk_id, sample in enumerate(chunk_samples):
            sent_data = {
                "token": sent_words,
                "label": sent_labels,
                "entity": sent_entity,
                "sent_id": f"s#{dataset_index}",
                "chunk_id": f"c#{chunk_id}",
                "source": source,
                "source_id": self.source_label_encoder.encode(source),
                "source_idx": source_index,
                "mode": sent_mode,
            }
            sample.update(sent_data)
            sent_samples.append(sample)
        return sent_samples


class MultiSentenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        loaders: Dict[str, Iterable[Sentence]],
        all_labels: Dict[str, List[str]],
        mode: str,
        args: Namespace,
        training: bool = False,
    ):
        super().__init__()
        self.preprocessor = MultiSentenceProcessor(all_labels, mode, args)
        self.samples = []
        self.samples_info = []
        self.sample_sources = []
        for data_name in loaders:
            self.sample_sources.append(data_name)
            for sid, sent in enumerate(loaders[data_name]):
                assert self.preprocessor.mode == sent.mode
                assert all(k in self.preprocessor.types[data_name].dataset_labels for k in sent.entity_counter)
                self.samples.append(sent.to_feature())
                info = [data_name, sid]
                self.samples_info.append(info)

        self.samples = Dataset.from_list(self.samples, features=Sentence.ds_feature())
        self.samples_info = pd.DataFrame(self.samples_info, columns=["source", "source_idx"])

        if training:
            if args.multids_sampling == "per_dataset":
                self.rand_idx = []
                for source in self.sample_sources:
                    source_sample = self.samples_info[self.samples_info["source"] == source]
                    sample_idx = source_sample.sample(100000, replace=True, random_state=args.seed).index.tolist()
                    self.rand_idx.extend(sample_idx)
                random.seed(args.seed)
                random.shuffle(self.rand_idx)
            elif args.multids_sampling == "per_sentence":
                self.rand_idx = self.samples_info.sample(100000, replace=True, random_state=args.seed).index.tolist()
            elif args.multids_sampling == "none":
                self.rand_idx = self.rand_idx = list(range(len(self.samples_info)))
            else:
                raise RuntimeError("unknown multi dataset sampling")
        else:
            self.rand_idx = list(range(len(self.samples_info)))
        del loaders
        gc.collect()

    def __getitem__(self, index):
        sent_feature = self.samples[self.rand_idx[index]]
        sent_info = self.samples_info.iloc[self.rand_idx[index]]
        random.seed(index)
        sample = self.preprocessor.prepare_word(
            sent_feature, dataset_index=index, source=sent_info["source"], source_index=sent_info["source_idx"].item()
        )
        return sample

    def __len__(self):
        return len(self.rand_idx)


@dataclass
class Episode:
    query: List[Union[Sentence]]
    support: List[Union[Sentence]]
    types: List[str]
    support_counter: Counter = field(init=False)
    query_counter: Counter = field(init=False)
    mode: str = field(init=False)

    def __post_init__(self):
        self.types = sorted(self.types)
        self.support_counter = Counter()
        for sent in self.support:
            assert all(k in self.types for k in sent.entity_counter), f"out of type entity"
            self.support_counter += sent.entity_counter
        # for paragraph sampling support/query types is the entire label space, the paragraph
        # might not cover all the types
        assert set(self.support_counter.keys()).issubset(set(self.types)), "support set type mismatch"

        self.query_counter = Counter()
        for sent in self.query:
            assert all(k in self.types for k in sent.entity_counter), "out of type entity"
            self.query_counter += sent.entity_counter
        assert set(self.query_counter.keys()).issubset(set(self.types)), "query set type mismatch"

        self.mode = self.support[0].mode
        assert all(self.mode == sent.mode for sent in self.support)
        assert all(self.mode == sent.mode for sent in self.query)

    def __str__(self):
        rep = []
        rep.append("{0:<30}{1:<30}{2:<30}".format("label", "support", "query"))
        for l in self.types:
            rep.append(f"{l:<30}{self.support_counter[l]:<30}{self.query_counter[l]:<30}")
        rep.append("{0:<30}{1:<30}{2:<30}".format("total", len(self.support), len(self.query)))
        return "\n".join(rep)

    @classmethod
    def seq_from_dict(cls, input_dict):
        return Sentence.from_dict(input_dict)

    @classmethod
    def from_dict(cls, input_dict):
        return cls(
            query=[cls.seq_from_dict(seq) for seq in input_dict["query"]],
            support=[cls.seq_from_dict(seq) for seq in input_dict["support"]],
            types=input_dict["types"],
        )

    def to_dict(self):
        return {
            "query": [seq.to_dict() for seq in self.query],
            "support": [seq.to_dict() for seq in self.support],
            "types": self.types,
        }

    @classmethod
    def from_json(cls, input_json):
        return cls.from_dict(json.loads(input_json))

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def ds_feature(cls):
        return Features(
            {
                "support": [
                    {
                        "tokens": [Value(dtype="string")],
                        "labels": [Value(dtype="string")],
                        "subwords": [Value(dtype="string")],
                        "entity_start": [Value(dtype="int64")],
                        "entity_end": [Value(dtype="int64")],
                        "entity_label": [Value(dtype="string")],
                        "mode": Value(dtype="string"),
                    }
                ],
                "query": [
                    {
                        "tokens": [Value(dtype="string")],
                        "labels": [Value(dtype="string")],
                        "subwords": [Value(dtype="string")],
                        "entity_start": [Value(dtype="int64")],
                        "entity_end": [Value(dtype="int64")],
                        "entity_label": [Value(dtype="string")],
                        "mode": Value(dtype="string"),
                    }
                ],
                "types": Sequence(Value(dtype="string")),
                "mode": Value(dtype="string"),
                "support_types": Sequence(Value(dtype="string")),
            }
        )

    def to_feature(self):
        support_types = sorted(list(self.support_counter.keys()))
        return {
            "support": [sent.to_feature() for sent in self.support],
            "query": [sent.to_feature() for sent in self.query],
            "types": self.types,
            "mode": self.mode,
            "support_types": support_types,
        }


class EpisodeDataset(torch.utils.data.Dataset):

    def __init__(self, loader: Iterable[Episode], labels: List[str], mode: str, args: Namespace):
        self.types = labels
        self.preprocessor = SentenceProcessor(None, mode, args)
        self.samples = []
        for idx, episode in enumerate(loader):
            assert mode == episode.mode
            assert set(episode.types).issubset(set(labels)), f"out of episode type"
            self.samples.append(episode.to_feature())
        self.samples = Dataset.from_list(self.samples, features=Episode.ds_feature())
        del loader, labels, mode, args, idx, episode
        # print(total_size(self.samples)) around 140M
        gc.collect()

    def __getitem__(self, index):
        episode_feature = self.samples[index]
        episode_types, episode_mode, episode_support_types = (
            episode_feature["types"],
            episode_feature["mode"],
            episode_feature["support_types"],
        )
        assert self.preprocessor.mode == episode_mode
        self.preprocessor.reinit_label(episode_types)
        random.seed(index)

        samples = []
        support_span_counter = Counter()
        for i, sent_feature in enumerate(episode_feature["support"]):
            for sample in self.preprocessor.prepare_word(sent_feature, f"support{i}"):
                sample["query_mask"] = 0
                samples.append(sample)
                support_span_counter.update([l for i, j, l in sample["span"]["pos_span"]])

        for i, sent_feature in enumerate(episode_feature["query"]):
            for sample in self.preprocessor.prepare_word(sent_feature, f"query{i}"):
                sample["query_mask"] = 1
                samples.append(sample)

        samples = base_collate(samples)
        # episode types determine which label encoder to use for encode/decode label idx to label text
        # therefore episode types must cover all types in support and query
        # episode types can be determined by dataset and keeps same across dataset
        # support types and support span types indicate which token/span types exists in support
        # sometimes support does not cover all the types in episode and this will create errors in pytorch
        # having support types and support span types helps avoid this kind of errors
        # model internally always use label idx from zero and predict distribution over entire episode_types
        # non-exist types will be padded to negative infinity
        samples["episode_types"] = episode_types
        samples["support_types"] = episode_support_types
        # some span are too long and may not necessaily come into support span
        samples["support_span_types"] = [t for t in episode_support_types if support_span_counter[t] > 0]
        # types keep record of dataset entire label space
        # sentence types are wrong since we only provide episode types to processor
        # be careful episode sort types internally so here type might be different
        samples["types"] = self.types
        samples["mode"] = check_same_compress(samples["mode"], self.preprocessor.mode)
        samples["sent_id"] = [f"e#{index}:{sent_id}" for sent_id in samples["sent_id"]]

        # print('sample size', total_size(samples))
        # for k, v in samples.items():
        #     print(k, ' size', total_size(v))
        # for k, v in samples['span'].items():
        #     print(k, ' size', total_size(v))
        #     if isinstance(v, np.ndarray):
        #         print('tensor size', v.shape)
        # print(max(samples['span']['min_len']))
        return samples

    def __len__(self):
        return len(self.samples)


def np2tensor(batch, pin_memory=False):
    if isinstance(batch, np.ndarray):
        # pin memory cannot be performed outside main process
        return torch.from_numpy(batch).pin_memory() if pin_memory else torch.from_numpy(batch)
    elif isinstance(batch, int) or isinstance(batch, float) or isinstance(batch, str):
        return batch
    elif isinstance(batch, dict):
        return {key: np2tensor(batch[key]) for key in batch}
    elif isinstance(batch, list):
        return batch
    else:
        raise TypeError("unknown batch layout")


class EpisodeSampler:

    def __init__(self, K: int, Q: int, loader: Iterable[Sentence], labels: List[str], uplimit=Optional[int]):
        self.K = K
        self.Q = Q
        self.uplimit = uplimit

        self.samples = []
        self.sample_counter = []
        self.types = labels

        for sample in loader:
            self.samples.append(sample)
            self.sample_counter.append([sample.entity_counter[c] for c in labels])

        self.sample_counter = pd.DataFrame(self.sample_counter, columns=labels)
        self.sample_counter.columns = self.sample_counter.columns.str.replace("-", "_")

    def validate(self, support_sample_idxs, query_sample_idxs):
        support_cnt = Counter()
        for index in support_sample_idxs:
            support_cnt += self.samples[index].entity_counter
        assert all(support_cnt[c] >= self.K for c in self.types)

        # query_cnt = Counter()
        # for index in query_sample_idxs:
        #     query_cnt += self.samples[index].entity_counter
        # assert all(query_cnt[c] >= self.K for c in self.types)
        assert len(query_sample_idxs) == self.Q

        assert len(set(support_sample_idxs).intersection(set(query_sample_idxs))) == 0

    def solve(self, lines, minimum_occurance):
        model = pulp.LpProblem("FewShot", pulp.LpMinimize)
        lp_vars = []
        class2var = defaultdict(list)
        for idx, entity_cnt in lines.items():
            indi = pulp.LpVariable(f"use_{idx}", cat="Binary")
            lp_vars.append((indi, 1))
            for key, cnt in entity_cnt:
                class2var[key].append((indi, cnt))
        
        model += pulp.lpSum([w * indi for indi, w in lp_vars]), "NumSent"
        
        for key in class2var:
            model += pulp.lpSum([w * indi for indi, w in class2var[key]]) >= minimum_occurance
        
        if self.uplimit:
            maximum_occurance = self.uplimit * minimum_occurance
            for key in class2var:
                model += pulp.lpSum([w * indi for indi, w in class2var[key]]) <= maximum_occurance

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[model.status] != "Optimal":
            return None

        used_idxs = []
        for indi, w in lp_vars:
            if indi.varValue > 0:
                index = int(indi.name[4:])
                used_idxs.append(index)
        
        total_cost = pulp.value(model.objective)
        return used_idxs


    def sample(self, seed):
        types = self.sample_counter.columns.tolist()
        
        support_cands = {}
        support_cands_cnt = Counter()

        for key in types:
            select = self.sample_counter[key] > 0
            samples = self.sample_counter[select].sample(self.K, replace=False, random_state=seed)
            for line in samples.itertuples():
                if line.Index in support_cands:
                    continue
                support_cands[line.Index] = []
                for key in types:
                    cnt = getattr(line, key)
                    if cnt > 0:
                        support_cands_cnt[key] += 1
                        support_cands[line.Index].append((key, cnt))

        support_sample_idxs = self.solve(support_cands, self.K)
        if support_sample_idxs is None:
            return None
        
        query_cand = set(range(len(self.samples))) - set(support_sample_idxs)
        query_sample_idxs = random.choices(list(query_cand), k=self.Q)
        
        # query_cands = {}
        # query_cands_cnt = Counter()

        # for key in types:
        #     select = (self.sample_counter[key] > 0) & (~self.sample_counter.index.isin(support_sample_idxs))
        #     samples = self.sample_counter[select].sample(self.Q, replace=False, random_state=seed)
        #     for line in samples.itertuples():
        #         assert line.Index not in support_sample_idxs
        #         if line.Index in query_cands:
        #             continue
        #         query_cands[line.Index] = []
        #         for key in types:
        #             cnt = getattr(line, key)
        #             if cnt > 0:
        #                 query_cands_cnt[key] += 1
        #                 query_cands[line.Index].append((key, cnt))

        # query_sample_idxs = self.solve(query_cands, self.Q)
        # if query_sample_idxs is None:
        #     return None

        query_samples = [self.samples[index] for index in query_sample_idxs]
        support_samples = [self.samples[index] for index in support_sample_idxs]

        self.validate(support_sample_idxs, query_sample_idxs)
        episode = Episode(query_samples, support_samples, self.types)
        assert len(episode.support) > 0 and len(episode.query) > 0, "empty support or query"

        return episode


def base_collate(batch):
    elem = batch[0]
    if not all([type(e) == type(elem) for e in batch]):
        raise RuntimeError("inconsistent batch structure")
    if isinstance(elem, np.ndarray):
        if elem.ndim == 0:
            return np.stack(batch)
        elif elem.ndim == 1:
            return np.vstack(batch)
        else:
            return np.concatenate(batch, 0)
    elif isinstance(elem, int) or isinstance(elem, float) or isinstance(elem, str):
        return batch
    elif isinstance(elem, list) or isinstance(elem, tuple):
        return batch
    elif isinstance(elem, dict):
        return {key: base_collate([d[key] for d in batch]) for key in elem}
    else:
        raise RuntimeError(f"unknown batch layout {type(elem)}")


def inspect_batch_structure(batch, prefix=""):
    if isinstance(batch, dict):
        for key in batch:
            inspect_batch_structure(batch[key], prefix + " " + key)
    elif isinstance(batch, list) or isinstance(batch, tuple):
        elem = batch[0]
        if not all([type(e) == type(elem) for e in batch]):
            raise RuntimeError("inconsistent batch structure")
        print(prefix.strip(), f"{type(batch)} of {type(elem)}")
    elif isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        print(prefix.strip(), f"{type(batch)} dtype {batch.dtype} shape {batch.shape}")
    elif isinstance(batch, int) or isinstance(batch, float) or isinstance(batch, str):
        print(prefix.strip(), f"{type(batch)} val {batch}")
    else:
        raise RuntimeError(f"unknown batch layout {type(elem)}")


def move_batch_tensor(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif type(batch) == dict:
        return {key: move_batch_tensor(batch[key], device) for key in batch}
    elif type(batch) == OrderedDict:
        return OrderedDict((key, move_batch_tensor(batch[key], device)) for key in batch)
    elif type(batch) == list:
        return [move_batch_tensor(elem, device) for elem in batch]
    elif type(batch) == tuple:
        return tuple(move_batch_tensor(elem, device) for elem in batch)
    else:
        return batch


def check_same_compress(batch, elem=None):
    if elem is None:
        elem = batch[0]
    if not all([e == elem for e in batch]):
        raise RuntimeError("not the same inside batch")
    return elem


def compress_same_obj(batch):
    prev_id = None
    unique = []
    for c in batch:
        if prev_id is None or prev_id != id(c):
            unique.append(c)
        prev_id = id(c)
    return unique


def compress_same_id(batch_id, batch=None):
    prev_id = None
    if batch:
        unique, unique_id = [], []
        for c, cid in zip(batch, batch_id):
            if prev_id is None or prev_id != cid:
                unique.append(c)
                unique_id.append(cid)
            prev_id = cid
        return unique, unique_id
    else:
        unique_id = []
        for cid in batch_id:
            if prev_id is None or prev_id != cid:
                unique_id.append(cid)
            prev_id = cid
        return unique_id


def slice_np_to_min_len(ndarray, min_len):
    # np.array copy numpy arr
    # first dim is batch
    if ndarray.ndim == 2:
        return np.array(ndarray[:, :min_len])
    elif ndarray.ndim == 3:
        return np.array(ndarray[:, :min_len, :min_len])
    else:
        raise RuntimeError(">=4d tensor")


def sentence_dataset_collate(batch):
    bs = len(batch)
    batch = base_collate(flat(batch))
    batch["types"] = check_same_compress(batch["types"])
    batch["mode"] = check_same_compress(batch["mode"])
    for name in ["bert", "word", "gather", "span"]:
        if name in batch:
            min_len = max(batch[name]["min_len"])
            for k, v in batch[name].items():
                if isinstance(v, np.ndarray):
                    batch[name][k] = slice_np_to_min_len(batch[name][k], min_len)
    assert "bs" not in batch
    batch["bs"] = bs
    return np2tensor(batch)


def multi_sentence_dataset_collate(batch):
    bs = len(batch)
    batch = base_collate(flat(batch))
    batch["source_id"] = np.array(batch["source_id"], dtype=np.int64)
    batch["mode"] = check_same_compress(batch["mode"])
    for name in ["bert", "word", "gather", "span"]:
        if name in batch:
            min_len = max(batch[name]["min_len"])
            for k, v in batch[name].items():
                if isinstance(v, np.ndarray):
                    batch[name][k] = slice_np_to_min_len(batch[name][k], min_len)
    assert "bs" not in batch
    batch["bs"] = bs
    # pin memory cannot be called outside main
    return np2tensor(batch)


def episode_dataset_collate(batch):
    bs = len(batch)
    batch = base_collate(batch)
    batch["mode"] = check_same_compress(batch["mode"])
    batch["is_query"] = batch["query_mask"]
    batch["batch_id"] = np.array(flat([[i] * len(mask) for i, mask in enumerate(batch["query_mask"])]))
    batch["query_mask"] = np.array(flat(batch["query_mask"]))
    for name in ["bert", "word", "gather", "span"]:
        if name in batch:
            min_len = max(flat(batch[name]["min_len"]))
            for k, v in batch[name].items():
                if isinstance(v, np.ndarray):
                    # if name == 'span' and k == 'full_span':
                    #     truncated = batch[name][k][:, min_len:]
                    #     assert np.all(truncated < 0)
                    batch[name][k] = slice_np_to_min_len(batch[name][k], min_len)
    assert "bs" not in batch
    batch["bs"] = bs
    return np2tensor(batch)


class SolidState:
    mode = "io"
    labels = ["APL", "CMT", "DSC", "MAT", "PRO", "SMT", "SPL"]
    DOMAIN = "Material Science"
    FULLNAME = {
        "MAT": "Material",
        "DSC": "Sample descriptor",
        "PRO": "Property",
        "SPL": "Symmetry/phase label",
        "SMT": "Synthesis method",
        "CMT": "Characterization method",
        "APL": "Application",
    }
    GUIDELINE = {
        "Material": "Material: Any inorganic solid or alloy, any non-gaseous element (at RT), e.g., “BaTiO3”, “titania”, “Fe”.",
        "Symmetry/phase label": "Symmetry/phase label: Names for crystal structures/phases, e.g., “tetrago- nal”, “fcc”, “rutile”, “perovskite”; or, any symmetry label such as “P bnm”, or “P nma”.",
        "Sample descriptor": "Sample descriptor: Special descriptions of the type/shape of the sample. Examples inlcude “single crystal”, “nanotube”, “quantum dot”.",
        "Property": "Property: Anything measurable that can have a unit and a value, e.g., “conductivity”, “band gap”; or, any qualitative property or phenomenon exhibited by a material, e.g., “ferroelectric”, “metallic”.",
        "Application": "Application: Any high-level application such as “photovoltaics”, or any specific device such as “field-effect transistor”.",
        "Synthesis method": "Synthesis method: Any technique for synthesising a material, e.g., “pulsed laser deposition”, “solid state reaction”, or any other step in sample production such as “annealing” or “etching”.",
        "Characterization method": "Characterization method: Any method used to characterize a material, experiment or theory: e.g., “photoluminescence”, “XRD”, “tight binding”, “DFT”. It can also be a name for an equation or model. such “Bethe-Salpeter equation”.'",
    }

    @staticmethod
    def raw_splits():
        dir_path = os.path.join(os.path.dirname(__file__), "data", "ceder")
        return {
            "all": os.path.join(dir_path, "ner_annotations.json"),
        }

    @staticmethod
    def load_raw(fn):
        # new version which does not have the <num> token
        with open(fn) as f:
            for line in f:
                line = json.loads(line.strip())
                if line["user"] == "leighmi6":
                    doi = line["doi"]
                    for sent_i, sent in enumerate(line["tokens"]):
                        words = [t["text"] for t in sent]
                        labels = [t["annotation"] if t["annotation"] else "O" for t in sent]
                        offsets = [(t["start"], t["end"]) for t in sent]
                        local_offset = [(s - offsets[0][0], e - offsets[0][0]) for s, e in offsets]
                        io_labels = [l if l not in ["PVL", "PUT"] else "O" for l in labels]
                        assert all([l in SolidState.labels for l in io_labels if l != "O"]), io_labels
                        sent_text = Sentence.reconstruct_text_from_offsets(words, local_offset)
                        sent = Sentence(sent_text, id=f"{doi}#s{sent_i}")
                        spans, _ = get_io_spans(io_labels)
                        span_offset = [
                            (
                                local_offset[i][0],
                                local_offset[j][1],
                                sent_text[local_offset[i][0] : local_offset[j][1]],
                                l,
                            )
                            for i, j, l in spans
                        ]
                        sent.add_entity_char_offset(span_offset)
                        yield sent
    
    @staticmethod
    def splits():
        dir_path = os.path.join(os.path.dirname(__file__), "data", "ceder")
        return {
            "all": os.path.join(dir_path, "ner_annotations.json"),
        }
    
    @staticmethod
    def load(fn):
        with open(fn) as f:
            for line in f:
                line = json.loads(line.strip())
                if line['user'] == 'leighmi6':
                    doi = line['doi']
                    for sent in line['tokens']:
                        words = [t['text'] for t in sent]
                        labels = [t['annotation'] if t['annotation'] else 'O' for t in sent]
                        io_labels = [l if l not in ['PVL', 'PUT'] else 'O' for l in labels]
                        sent = Sentence.from_texts(words)
                        spans, span_cnt = get_io_spans(io_labels)
                        sent.add_entity_token_idx(spans)
                        yield sent


class Catalysis:
    mode = "bio"
    labels = ["Catalyst", "Product", "Reaction", "Reactant", "Treatment", "Characterization"]
    DOMAIN = "Chemical Science"
    FULLNAME = {
        "Catalyst": "Catalyst",
        "Product": "Product",
        "Reaction": "Reaction",
        "Reactant": "Reactant",
        "Treatment": "Catalyst Synthesis Treatment",
        "Characterization": "Catalyst Characterization method",
    }
    GUIDELINE = {
        "Catalyst": "Catalyst: Metals and Supports that catalyze chemical reaction e.g., Ru/CeO2, CeO2-supported metal catalysts, and Pt/H-USY (Pt:1 wt %) catalysts.",
        "Reactant": "Reactant: Species that interact with the catalyst to create a product (e.g., polyethylene, plastics, and PE).",
        "Product": "Product: Species that are produced from a chemical reaction between the reactant and the catalyst (e.g., C1−C4, coke, and liquid fuel).",
        "Reaction": "Reaction: Processes that involve the transformation of a chemical species via interactions with a catalyst (e.g., hydrogenation, isomerization, and hydrocracking).",
        "Treatment": "Treatment: Any technique that is used to yield useful information about the catalyst (e.g., gas chromatography mass spectroscopy (GC−MS), powder X-ray diffraction (XRD), and infrared spectrometry (IR)).",
        "Characterization": "Characterization: Any intermediate steps taken to prepare the catalyst (e.g., heating, calcination, and refluxing).",
    }

    @staticmethod
    def raw_splits():
        dir_path = os.path.join(os.path.dirname(__file__), "data", "catalysis")
        return {
            "all": os.path.join(dir_path, "ner", "combined_detokenize.json"),
        }

    @staticmethod
    def load_raw(fn):
        with open(fn) as f:
            for doi, sents in json.loads(f.read()).items():
                for sent_i, sent in enumerate(sents):
                    text = sent["text"]
                    words = sent["tokens"]
                    labels = sent["labels"]
                    offsets = sent["token_offsets"]
                    sent = Sentence(text, id=f"{doi}#s{sent_i}")
                    spans, _ = get_bio_spans(labels)
                    span_offset = [
                        (offsets[i][0], offsets[j][1], text[offsets[i][0] : offsets[j][1]], l)
                        for i, j, l in spans
                    ]
                    sent.add_entity_char_offset(span_offset)
                    yield sent

    @staticmethod
    def splits():
        dir_path = os.path.join(os.path.dirname(__file__), "data", "catalysis")
        return {
            "all": os.path.join(dir_path, "ner", "combined.json"),
        }

    @staticmethod
    def load(fn):
        with open(fn) as f:
            for doi, sents in json.loads(f.read()).items():
                for sent in sents:
                    words = [t['text'] for t in sent]
                    labels = [t.get('span_label', 'O') for t in sent]
                    assert len(words) == len(labels)
                    sent = Sentence.from_texts(words)
                    spans, span_cnt = get_bio_spans(labels)
                    sent.add_entity_token_idx(spans)
                    yield sent

class CHEMU:
    mode = "bio"
    labels = [
        "EXAMPLE_LABEL",
        "REACTION_PRODUCT",
        "STARTING_MATERIAL",
        "REAGENT_CATALYST",
        "SOLVENT",
        "OTHER_COMPOUND",
        "TIME",
        "TEMPERATURE",
        "YIELD_OTHER",
        "YIELD_PERCENT",
    ]

    @staticmethod
    def raw_splits():
        dir_path = os.path.join(os.path.dirname(__file__), "data", "chemu")
        return {
            "train": os.path.join(dir_path, "task1a-ner-train-dev", "train"),
            "dev": os.path.join(os.path.dirname(__file__), "task1a-ner-train-dev", "dev"),
        }

    @staticmethod
    def load_raw(input_file):
        # file has duplicate entity for same type
        brat = BratParser(error="ignore")
        examples = brat.parse(input_file)
        splitter = ChemSentenceTokenizer()
        for example in examples:
            text = example.text

            texts, offsets, ner_ents = [], [], []
            for span in splitter.span_tokenize(text):
                texts.append(text[span[0] : span[1]])
                offsets.append(span)
                ner_ents.append([])

            for e in example.entities:
                for idx, (sent_start, sent_end) in enumerate(offsets):
                    if sent_start <= e.start <= e.end <= sent_end:
                        local_offset = e.start - sent_start, e.end - sent_start
                        # AssertionError: ('chloroform\xa0', 'chloroform')
                        try:
                            assert texts[idx][local_offset[0] : local_offset[1]] == e.mention
                        except AssertionError:
                            assert texts[idx][local_offset[0] : local_offset[1]].strip() == e.mention
                            e.mention = texts[idx][local_offset[0] : local_offset[1]]
                        ner_ents[idx].append((*local_offset, e.mention, e.type))
                        break

            for sent_id, (sent_text, sent_ent) in enumerate(zip(texts, ner_ents)):
                sent = Sentence(sent_text, id=f"{example.id}#{sent_id}")
                sent.add_entity_char_offset(sent_ent)
                yield sent

    @staticmethod
    def splits():
        return {
            'train': os.path.join(os.path.dirname(__file__), "data/chemu/task1-train.json"),
            'test': os.path.join(os.path.dirname(__file__), "data/chemu/task1-dev.json")
        }

    @staticmethod
    def load(input_file):
        with open(input_file) as f:
            for line in f:
                data = json.loads(line)
                words = data['tokens']
                spans = data['spans']
                sent = Sentence.from_texts(words)
                sent.add_entity_token_idx(spans)
                yield sent


class PcMSP:
    mode = "bio"
    labels = [
        "Descriptor",
        "Material-target",
        "Operation",
        "Material-recipe",
        "Value",
        "Brand",
        "Material-intermedium",
        "Device",
        "Property-pressure",
        "Property-temperature",
        "Property-rate",
        "Property-time",
        "Material-others",
    ]

    @staticmethod
    def raw_splits():
        dir_path = os.path.join(os.path.dirname(__file__), "data", "pcmsp", "PcMSP-main")
        return {
            "train": os.path.join(dir_path, "original", "train"),
            "dev": os.path.join(dir_path, "original", "dev"),
            "test": os.path.join(dir_path, "original", "test"),
        }

    @staticmethod
    def load_raw(path):
        arr = os.listdir(path)
        for file_name in arr:
            # Read in the file contents
            if file_name == ".DS_Store":
                continue
            # if file_name == '103390ma11060903.tsv':
            #     print('f')
            with open(os.path.join(path, file_name)) as file:
                lines = file.readlines()
            current_line = 0
            parsed_data = []
            while current_line < len(lines):
                # Check if the current line is a Text label
                if lines[current_line].startswith("#Text"):
                    rawSentence = {}
                    rawSentence["rawText"] = []
                    current_line += 1
                    while current_line < len(lines) and lines[current_line] != "\n":
                        if not (lines[current_line].startswith("#Text")):
                            rawSentence["rawText"].append(lines[current_line])
                        current_line += 1
                    # Append the parsed data to the result list
                    parsed_data.append(rawSentence)
                current_line += 1
            for p in range(len(parsed_data)):
                word_w = []
                labels_w = []
                offsets_w = []
                for x in parsed_data[p]["rawText"]:
                    split_res = x.split("\t")
                    word_w.append(split_res[2])
                    label = split_res[3].replace("_", "O")
                    label = label.split("|")[0]
                    label = re.sub(r"(.+)(\[\d+\])", r"\1", label)
                    if label == "Operationf":
                        print("fix label: Operationf -> Operation")
                        label = "Operation"
                    labels_w.append(label)
                    start, end = map(int, split_res[1].split("-"))
                    offsets_w.append((start, end))
                local_offset = [(s - offsets_w[0][0], e - offsets_w[0][0]) for s, e in offsets_w]
                word_w = [" " * (e - s - len(word)) + word for word, (s, e) in zip(word_w, local_offset)]
                sent_text = Sentence.reconstruct_text_from_offsets(word_w, local_offset)
                sent = Sentence(sent_text, id=f"Sentence{p}#s{p}")
                spans, _ = get_io_spans(labels_w)
                span_offset = [
                    (local_offset[i][0], local_offset[j][1], sent_text[local_offset[i][0] : local_offset[j][1]], l)
                    for i, j, l in spans
                ]
                sent.add_entity_char_offset(span_offset)
                yield sent

    @staticmethod
    def splits():
        dir_path = os.path.join(os.path.dirname(__file__), 'data', 'pcmsp', 'PcMSP-main')
        return {
            'train': os.path.join(dir_path, "mat_train.json"),
            'dev': os.path.join(dir_path, "mat_dev.json"),
            'test': os.path.join(dir_path, "mat_test.json")
        }

    @staticmethod
    def load(input_file):
        with open(input_file) as f:
            data = json.loads(f.read())
            for paragraph in data:
                words = paragraph['tokens']
                spans = []
                for v in paragraph['vertexSet']:
                    start, end = v['tokenpositions'][0], v['tokenpositions'][-1]
                    label = v['kbID']
                    spans.append((start, end, label))
                sent = Sentence.from_texts(words)
                sent.add_entity_token_idx(spans)
                yield sent


class MSMention:
    mode = "bio"
    labels = [
        "Nonrecipe-operation",
        "Meta",
        "Operation",
        "Target",
        "Material",
        "Unspecified-Material",
        "Nonrecipe-Material",
        "Sample",
        "Number",
        "Amount-Unit",
        "Condition-Unit",
        "Property-Unit",
        "Synthesis-Apparatus",
        "Apparatus-Unit",
    ]

    @staticmethod
    def raw_splits():
        dir_path = os.path.join(os.path.dirname(__file__), "data", "msmention")
        return {
            "train": os.path.join(dir_path, "msmentions2020", "train"),
            "dev": os.path.join(dir_path, "msmentions2020", "dev"),
            "test": os.path.join(dir_path, "msmentions2020", "original", "test"),
        }

    @staticmethod
    def load_raw(path):
        brat = BratParser(error="ignore")
        examples = brat.parse(path)
        splitter = ChemSentenceTokenizer()
        for example in examples:
            doi, text = example.text.split("\n", 1)
            doi_offset = len(doi) + 1

            texts, offsets, ner_ents = [], [], []
            for span in splitter.span_tokenize(text):
                texts.append(text[span[0] : span[1]])
                offsets.append(span)
                ner_ents.append([])

            for e in example.entities:
                for idx, (sent_start, sent_end) in enumerate(offsets):
                    no_doi_start = e.start - doi_offset
                    no_doi_end = e.end - doi_offset
                    if sent_start <= no_doi_start <= no_doi_end <= sent_end:
                        ner_ents[idx].append((no_doi_start - sent_start, no_doi_end - sent_start, e.mention, e.type))
                        break

            for sent_text, sent_ent in zip(texts, ner_ents):
                sent = Sentence(sent_text, id=f"{example.id}#")
                sent.add_entity_char_offset(sent_ent)
                yield sent

    @staticmethod
    def splits():
        dir_path = os.path.join(os.path.dirname(__file__), 'data', 'msmention')
        return {
            'train': os.path.join(dir_path, "ner-train.json"),
            'dev': os.path.join(dir_path, "ner-dev.json"),
            'test': os.path.join(dir_path, "ner-test.json")
        }

    @staticmethod
    def load(input_file):
        with open(input_file) as f:
            for line in f:
                data = json.loads(line)
                words = data['tokens']
                spans = data['spans']
                sent = Sentence.from_texts(words)
                sent.add_entity_token_idx(spans)
                yield sent


class WNUT:
    mode = "bio"
    labels = [
        "Method",
        "Action",
        "Modifier",
        "Reagent",
        "Measure-Type",
        "Concentration",
        "Amount",
        "Location",
        "Speed",
        "Time",
        "Numerical",
        "Temperature",
        "Generic-Measure",
        "Size",
        "Device",
        "Seal",
        "Mention",
        "pH",
    ]

    @staticmethod
    def raw_splits():
        dir_path = os.path.join(os.path.dirname(__file__), "data", "wnut", "data")
        return {
            "train": os.path.join(dir_path, "train_data", "Standoff_Format"),
            "dev": os.path.join(dir_path, "dev_data", "Standoff_Format"),
            "test": [
                os.path.join(dir_path, "test_data", "Standoff_Format"),
                os.path.join(dir_path, "test_data_2020", "Standoff_Format"),
            ],
        }

    @staticmethod
    def load_raw(input_file):
        brat = BratParser(error="ignore")
        if type(input_file) == str:
            examples = brat.parse(input_file)
            splitter = ChemSentenceTokenizer()
            for example in examples:
                text = example.text

                texts, offsets, ner_ents = [], [], []
                for span in splitter.span_tokenize(text):
                    texts.append(text[span[0] : span[1]])
                    offsets.append(span)
                    ner_ents.append([])

                for e in example.entities:
                    for idx, (sent_start, sent_end) in enumerate(offsets):
                        if sent_start <= e.start <= e.end <= sent_end:
                            local_offset = e.start - sent_start, e.end - sent_start
                            # AssertionError: ('chloroform\xa0', 'chloroform')
                            try:
                                assert texts[idx][local_offset[0] : local_offset[1]] == e.mention
                            except AssertionError:
                                assert texts[idx][local_offset[0] : local_offset[1]].strip() == e.mention
                                e.mention = texts[idx][local_offset[0] : local_offset[1]]
                            ner_ents[idx].append((*local_offset, e.mention, e.type))
                            break

                for sent_id, (sent_text, sent_ent) in enumerate(zip(texts, ner_ents)):
                    sent = Sentence(sent_text, id=f"{example.id}#{sent_id}")
                    sent.add_entity_char_offset(sent_ent)
                    yield sent
        elif type(input_file) == list:
            iterators = [WNUT.load_raw(fn) for fn in input_file]
            for sent in itertools.chain(*iterators):
                yield sent

    @staticmethod
    def splits():
        dir_path = os.path.join(os.path.dirname(__file__), 'data', 'wnut')
        return {
            'train': os.path.join(dir_path, "train.jsonl"),
            'dev': os.path.join(dir_path, "dev.jsonl"),
            'test': os.path.join(dir_path, "test.jsonl")
        }

    @staticmethod
    def load(input_file):
        with open(input_file) as f:
            for line in f:
                data = json.loads(line)
                words = data['words']
                spans = data['spans']
                sent = Sentence.from_texts(words)
                sent.add_entity_token_idx(spans)
                yield sent


def init_argparser(parser):
    parser.add_argument("--backbone", default="models/s2orc-scibert-cased", type=str, help="bert encoder name")
    parser.add_argument("--mode", default="io", type=str, help="entity tagging schema")
    parser.add_argument("--max_length", default=100, type=int, help="max length")
    parser.add_argument("--max_span_length", default=10, type=int, help="max span length")
    parser.add_argument("--neg_rate", default=5.6, type=float, help="neg span sample rate")
    # context helps for crf
    parser.add_argument("--context", default=50, type=int, help="max length")
    return parser


def add_fewshot_specific_args(parser: ArgumentParser):
    parser.add_argument("--source", default="", help="source dataset")
    parser.add_argument("--target", default="", help="target dataset")
    parser.add_argument("--Qtest", default=10, type=int, help="Q test")
    parser.add_argument("--Kshot", default=10, type=int, help="K shot")
    return parser



def add_multidataset_specific_args(parser: ArgumentParser):
    parser.add_argument("--multids_sampling", default="none", type=str, 
                    help="how to sample sentence from different dataset for multi dataset pretraining")
    return parser


def add_train_specific_args(parser):
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--train_step", default=10000, type=int, help="num of step in training")
    parser.add_argument("--bs", default=1, type=int, help="eval batch size")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="Enter gpu num")
    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--bf16", action="store_true", help="bf16")
    parser.add_argument("--tf32", action="store_true", help="tf32")
    parser.add_argument("--val_times", default=10, type=int, help="num of validation times")
    parser.add_argument("--grad_acc", default=1, type=int, help="grad acc step")
    parser.add_argument("--output_dir", default="", help="output dir")
    parser.add_argument("--checkpoint", action="store_true", help="save last checkpoint")
    parser.add_argument("--best_checkpoint", action="store_true", help="save and test with best checkpoint")
    parser.add_argument("--wandb_proj", default="fewshotchemical", help="wandb project")
    parser.add_argument("--wandb_entity", default="nsndimt", help="wandb entity")
    parser.add_argument("--logging_level", default="info", help="logger level")
    parser.add_argument("--profile", action="store_true", help="profile code")
    parser.add_argument("--nan_detect", action="store_true", help="detect nan")
    parser.add_argument("--debug", action="store_true", help="debug code, not generate output")
    return parser


def set_logging(level="info"):
    root = logging.getLogger()
    if level == "error":
        root.setLevel(logging.ERROR)
    elif level == "warning":
        root.setLevel(logging.WARNING)
    elif level == "info":
        root.setLevel(logging.INFO)
    elif level == "debug":
        root.setLevel(logging.DEBUG)
    else:
        raise RuntimeError("unknown logging level")

    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    return root, formatter


NAME2CLASS = {
    "Catalysis": Catalysis,
    "SolidState": SolidState,
    "PcMSP": PcMSP,
    "MSMention": MSMention,
    "CHEMU": CHEMU,
    "WNUT": WNUT,
}


def convert_sentence(mode):
    # Catalysis, SolidState, PcMSP, MSMention, CHEMU, WNUT
    for dsname, dsclass in NAME2CLASS.items():
        data = []
        for split, fn in dsclass.splits().items():
            for sent in dsclass.load(fn):
                sent.convert_entity_position(mode)
                # print(sent)
                data.append(sent)
        print(f"{dsname} num of sentence {len(data)}")
        
        with open(os.path.join("episode", f"{dsname}.jsonl"), "w") as f:
            for sent in data:
                f.write(sent.to_json() + "\n")


def split_sentence(dsnames, test_size=0.3, seed=12345):
    if type(dsnames) == str:
        dsnames = [dsnames]

    for dsname in dsnames:
        with open(os.path.join("episode", f"{dsname}.jsonl")) as f:
            data = [line.strip() for line in f]
        
        train_sent, test_sent = train_test_split(data, test_size=test_size, random_state=seed)
        
        with open(os.path.join("episode", f"{dsname}_train.jsonl"), "w") as f:
            for sent in train_sent:
                f.write(sent + "\n")
        
        with open(os.path.join("episode", f"{dsname}_test.jsonl"), "w") as f:
            for sent in test_sent:
                f.write(sent + "\n")


def sample_episode(mode, N_episode, K_shot, Q_test, uplimit=None, just_sample=True):
    # Catalysis, SolidState, PcMSP, MSMention, CHEMU, WNUT
    # dsnames = [("WNUT", WNUT)]
    for dsname, dsclass in NAME2CLASS.items():
        data = []
        error_times = 0
        for split, fn in dsclass.splits().items():
            for sent in dsclass.load(fn):
                sent.convert_entity_position(mode)
                # print(sent)
                data.append(sent)
        print(f"{dsname} num of sentence {len(data)}")
        print(f"total convert error {error_times}")

        sampler = EpisodeSampler(K_shot, Q_test, data, dsclass.labels, uplimit=uplimit)
        pbar = tqdm()
        success_data = []
        support_sizes = []
        pbar.set_description(f"success: 0 fail: 0 total: 0")
        num_iter, iter_MAX = 0, 1000000
        assert N_episode <= iter_MAX
        while len(success_data) < N_episode and num_iter < iter_MAX:
            ret = sampler.sample(num_iter)
            if ret is not None:
                success_data.append(ret)
                support_sizes.append(len(ret.support))

            num_iter += 1
            pbar.set_description(f"success: {len(success_data)} fail: {num_iter - len(success_data)} total: {num_iter}")

        print(
            f"1% support size {np.percentile(support_sizes, 1)}, "
            f"2% support size {np.percentile(support_sizes, 2)}, "
            f"25% support size {np.percentile(support_sizes, 25)}"
        )
        print(
            f"75% support size {np.percentile(support_sizes, 75)}, "
            f"98% support size {np.percentile(support_sizes, 98)}, "
            f"99% support size {np.percentile(support_sizes, 99)}"
        )

        if not just_sample:
            with open(os.path.join("episode", f"{dsname}_K{K_shot}_Q{Q_test}.jsonl"), "w") as f:
                for episode in success_data:
                    f.write(episode.to_json() + "\n")


def load_episode(dsnames, args, training=False, debug_firstk=None):
    mode = args.mode
    if type(dsnames) == str:
        dsnames = [dsnames]

    datasets = []
    for dsname in dsnames:
        dsclass = NAME2CLASS[dsname]
        with open(os.path.join("episode", f"{dsname}_K{args.Kshot}_Q{args.Qtest}.jsonl")) as f:
            data = []
            for line in tqdm(f):
                episode = Episode.from_json(line)
                data.append(episode)

                if debug_firstk is not None and len(data) == debug_firstk:
                    break

        dataset = EpisodeDataset(data, dsclass.labels, mode, args)
        datasets.append(dataset)

    datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        datasets,
        collate_fn=episode_dataset_collate,
        batch_size=args.bs if training else args.bs * 2,
        shuffle=training and not args.nan_detect,
        num_workers=2 if not args.nan_detect else 0,
        drop_last=training,
    )
    return datasets, dataloader



def load_sent(dsnames, part, args, training=False, debug_firstk=None):
    if type(dsnames) == str:
        dsnames = [dsnames]

    datas = {}
    labels = {}
    for dsname in dsnames:
        dsclass = NAME2CLASS[dsname]
        if part == "train":
            jsonfile = f"{dsname}_train.jsonl"
        elif part == "test":
            jsonfile = f"{dsname}_test.jsonl"
        elif part == "all":
            jsonfile = f"{dsname}.jsonl"
        else:
            raise RuntimeError("unknown sentence split")
        with open(os.path.join("episode", jsonfile)) as f:
            datas[dsname] = []
            labels[dsname] = dsclass.labels
            for line in tqdm(f):
                sent = Sentence.from_json(line)
                datas[dsname].append(sent)
                if debug_firstk is not None and len(datas[dsname]) == debug_firstk:
                    break

    dataset = MultiSentenceDataset(datas, labels, args.mode, args, training)
    types = dataset.preprocessor.types

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=multi_sentence_dataset_collate,
        batch_size=args.bs if training else args.bs * 2,
        shuffle=False, #MultiSentenceDataset do the sample internally if set to training
        num_workers=2 if not args.nan_detect else 0,
        drop_last=training,
    )
    return dataset, dataloader, types



if __name__ == "__main__":
    parser = ArgumentParser()
    parser = init_argparser(parser)
    parser = add_fewshot_specific_args(parser)
    parser = add_train_specific_args(parser)
    args, _ = parser.parse_known_args()

    convert_sentence(args.mode)
    split_sentence(NAME2CLASS.keys())
    sample_episode(args.mode, 1000, args.Kshot, args.Qtest, just_sample=False)

    # dataset, dataloader = load_episode(["Catalysis"], args, debug_firstk=100)
    # inspect_batch_structure(next(iter(dataloader)))

    # dataset, dataloader, types = load_sent(NAME2CLASS.keys(), "all", args, training=True, debug_firstk=100)
    # inspect_batch_structure(next(iter(dataloader)))
