import json
import logging
import math
import os
from typing import Dict, List, Tuple
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
from fs_ner_utils import (
    ShiftLabelEncoder,
    compress_same_id,
    convert_io2bio,
    flat,
    LabelEncoder,
    get_bio_spans,
    get_io_spans,
    mask_select,
    check_same_compress,
    span2bio,
    span2io,
)
from transformers import BertModel, BertConfig
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score


def all_avg_eval(y_true, y_pred, mode):
    if mode == "bio":
        # we do not use strict mode since others use conll.pl
        # in this case entity start with I-xx will also be a correct entity
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    elif mode == "io":
        # do not know why it does not work with list
        y_true = [list(convert_io2bio(sent_target)) for sent_target in y_true]
        y_pred = [list(convert_io2bio(sent_pred)) for sent_pred in y_pred]
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        raise RuntimeError("unsupport mode")
    return report, precision, recall, f1


def FewNERD_eval(y_true, y_pred, mode):
    pred_cnt = 0  # pred entity cnt
    label_cnt = 0  # true label entity cnt
    correct_cnt = 0  # correct predicted entity cnt

    for episode_preds, episode_targets in zip(y_pred, y_true):
        assert len(episode_preds) == len(episode_targets)
        for sent_pred, sent_target in zip(episode_preds, episode_targets):
            assert len(sent_pred) == len(sent_target) > 0
            if mode == "io":
                pred_spans, _ = get_io_spans(sent_pred)
                target_spans, _ = get_io_spans(sent_target)
            elif mode == "bio":
                pred_spans, _ = get_bio_spans(sent_pred)
                target_spans, _ = get_bio_spans(sent_target)
            else:
                raise RuntimeError("unsupported labeling mode")

            pred_cnt += len(pred_spans)
            label_cnt += len(target_spans)

            for pi, pj, pl in pred_spans:
                for ti, tj, tl in target_spans:
                    if pi == ti and pj == tj and pl == tl:
                        correct_cnt += 1

    precision = correct_cnt / (pred_cnt + 1e-6)
    recall = correct_cnt / (label_cnt + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


def SNIPS_eval(y_true, y_pred, mode):
    episode_f1 = []
    episode_precision = []
    episode_recall = []
    for episode_preds, episode_targets in zip(y_pred, y_true):
        assert len(episode_preds) == len(episode_targets)
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        for sent_pred, sent_target in zip(episode_preds, episode_targets):
            assert len(sent_pred) == len(sent_target) > 0
            if mode == 'io':
                pred_spans, _ = get_io_spans(sent_pred)
                target_spans, _ = get_io_spans(sent_target)
            elif mode == 'bio':
                pred_spans, _ = get_bio_spans(sent_pred)
                target_spans, _ = get_bio_spans(sent_target)
            else:
                raise RuntimeError('unsupported labeling mode')

            pred_cnt += len(pred_spans)
            label_cnt += len(target_spans)

            for pi, pj, pl in pred_spans:
                for ti, tj, tl in target_spans:
                    if pi == ti and pj == tj and pl == tl:
                        correct_cnt += 1

        precision = correct_cnt / (pred_cnt + 1e-6)
        recall = correct_cnt / (label_cnt + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        episode_f1.append(f1)
        episode_precision.append(precision)
        episode_recall.append(recall)

    episode_avg_f1 = sum(episode_f1) / len(episode_f1)
    episode_avg_precision = sum(episode_precision) / len(episode_precision)
    episode_avg_recall = sum(episode_recall) / len(episode_recall)

    return episode_avg_precision, episode_avg_recall, episode_avg_f1


class NERBaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def assemble_sentence(self, output):
        test_preds, test_targets = OrderedDict(), OrderedDict()
        batch_preds, batch_targets = output["pred"], output["target"]
        assert list(map(len, batch_preds)) == list(map(len, batch_targets))
        batch_sent_id = output["input"]["sent_id"]
        batch_chunk_id = output["input"]["chunk_id"]
        # Fill a list with the values the sentence predict and sentence tarjet
        for chunk_pred, chunk_target, sent_id, chunk_id in zip(
            batch_preds, batch_targets, batch_sent_id, batch_chunk_id
        ):
            test_preds.setdefault(sent_id, []).append((chunk_id, chunk_pred))
            test_targets.setdefault(sent_id, []).append((chunk_id, chunk_target))

        result = OrderedDict()
        assert test_preds.keys() == test_targets.keys()
        for sent_id in test_preds:
            # default sorted on first element ascending
            # Organize the the dict
            para_preds = flat(arr for chunk_id, arr in sorted(test_preds[sent_id]))
            para_targets = flat(arr for chunk_id, arr in sorted(test_targets[sent_id]))
            assert len(para_preds) == len(para_targets)
            # join in a same dictionary
            result[sent_id] = (para_preds, para_targets)

        return result

    def eval_epoch_end(self, eval_step_outputs: List, prefix: str, log_metric=True):
        if self.task in ["all_avg"]:
            y_true, y_pred = [], []
            mode = check_same_compress([output["input"]["mode"] for output in flat(eval_step_outputs)])
            for output in flat(eval_step_outputs):
                result = self.assemble_sentence(output)
                y_true.append([para_targets for _, para_targets in result.values()])
                y_pred.append([para_preds for para_preds, _ in result.values()])
        elif self.task in ["multi_supervised"]:
            y_true, y_pred = defaultdict(list), defaultdict(list)
            mode = check_same_compress([output["input"]["mode"] for output in eval_step_outputs])
            for output in eval_step_outputs:
                sent_sources, sent_ids = compress_same_id(output["input"]["sent_id"], output["source"])
                _sent_ids, sent_preds_targets = zip(*self.assemble_sentence(output).items())
                assert list(_sent_ids) == sent_ids, (_sent_ids, sent_ids)
                for source, (sent_preds, sent_targets) in zip(sent_sources, sent_preds_targets):
                    y_true[source].append(sent_targets)
                    y_pred[source].append(sent_preds)
        else:
            raise RuntimeError("unknown mode")

        if self.task in ["all_avg"]:
            precision, recall, f1 = FewNERD_eval(y_true, y_pred, mode)
            logging.info("precision: {0:3.4f}, recall: {1:3.4f}, f1: {2:3.4f}".format(precision, recall, f1))
            if log_metric:
                self.log(f"{prefix}/precision", precision)
                self.log(f"{prefix}/recall", recall)
                self.log(f"{prefix}/f1", f1)
            else:
                return {
                    f"{prefix}/precision": precision,
                    f"{prefix}/recall": recall,
                    f"{prefix}/f1": f1
                }
        elif self.task in ["multi_supervised"]:
            macro_precision, macro_recall, macro_f1 = [], [], []
            if len(y_true.keys()) > 1:
                ret = {}
                for source in sorted(list(y_true.keys())):
                    report, precision, recall, f1 = all_avg_eval(y_true[source], y_pred[source], mode)
                    logging.info(f"\n# source dataset: {source}\n" + \
                                f"# precision: {precision:3.4f}, recall: {recall:3.4f}, f1: {f1:3.4f}\n" + \
                                report
                    )
                    macro_precision.append(precision)
                    macro_recall.append(recall)
                    macro_f1.append(f1)
                    if log_metric:
                        self.log(f"{prefix}/{source}/precision", precision)
                        self.log(f"{prefix}/{source}/recall", recall)
                        self.log(f"{prefix}/{source}/f1", f1)
                    else:
                        ret.update({
                            f"{prefix}/precision": precision,
                            f"{prefix}/recall": recall,
                            f"{prefix}/f1": f1}
                        )
            
                macro_precision = np.mean(macro_precision).item()
                macro_recall = np.mean(macro_recall).item()
                macro_f1 = np.mean(macro_f1).item()
                logging.info(f"\n# all dataset macro average:\n" + \
                            f"# precision: {macro_precision:3.4f}, recall: {macro_recall:3.4f}, f1: {macro_f1:3.4f}\n")
                if log_metric:
                    self.log(f"{prefix}/precision", macro_precision)
                    self.log(f"{prefix}/recall", macro_recall)
                    self.log(f"{prefix}/f1", macro_f1)
                else:
                    ret.update({
                        f"{prefix}/precision": precision,
                        f"{prefix}/recall": recall,
                        f"{prefix}/f1": f1}
                    )
                    return ret
            else:
                source = list(y_true.keys())[0]
                report, precision, recall, f1 = all_avg_eval(y_true[source], y_pred[source], mode)
                logging.info(f"# precision: {precision:3.4f}, recall: {recall:3.4f}, f1: {f1:3.4f}\n" + report)
                if log_metric:
                    self.log(f"{prefix}/precision", precision)
                    self.log(f"{prefix}/recall", recall)
                    self.log(f"{prefix}/f1", f1)
                else:
                    return {
                        f"{prefix}/precision": precision,
                        f"{prefix}/recall": recall,
                        f"{prefix}/f1": f1
                    }
        else:
            raise RuntimeError("unknown mode")

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs.append(self(batch))

    def on_validation_epoch_end(self):
        logging.info("## Validation Result ##")
        self.eval_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self(batch))

    def on_test_epoch_end(self):
        logging.info("## Test Result ##")
        self.eval_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self, params_lr_wd):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        grouped_param = []
        for params, lr, wd in params_lr_wd:
            grouped_param.append(
                {"params": [p for n, p in params if not any(nd in n for nd in no_decay)], "weight_decay": wd, "lr": lr}
            )
            grouped_param.append(
                {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": lr}
            )
        # small speedup really does not matter
        # try:
        #     from apex.optimizers import FusedAdam
        #     optimizer = FusedAdam(grouped_param, lr=self.hparams.lr)
        # except ImportError:
        #     logging.info("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        # huggingface/fairseq all use adamw to replace adam
        # bert setting eps=1e-6 beta=(0.9, 0.999) weight decay 0.01 clipping 1.0
        # bert setting eps=1e-6 beta=(0.9, 0.98) weight decay 0.1 clipping 0
        # we use bert style
        # pytorch support fused adamw after 2.0 but we do not use it for compatibility
        # enable fusion does not allow gradient clipping and foreach is by default true
        # so nothing need to do here.
        optimizer = torch.optim.AdamW(grouped_param, lr=self.hparams.lr, eps=1e-6)
        warmup_steps = int(self.hparams.train_step * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, self.hparams.train_step)

        return [{"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}]


def big_atten(query_emb: torch.Tensor, in_class_emb: torch.Tensor, atten_temp: float):
    query_proto = []
    for query_chunk in query_emb.split(10240):
        query_chunk_sim = []
        for in_class_chunk in in_class_emb.split(10240):
            if atten_temp != 1.0:
                query_chunk_sim.append(query_chunk.matmul(in_class_chunk.T) * atten_temp)
            else:
                query_chunk_sim.append(query_chunk.matmul(in_class_chunk.T))
        query_chunk_sim = torch.cat(query_chunk_sim, dim=-1).softmax(-1)
        query_chunk_proto = torch.zeros_like(query_chunk)
        for sim_chunk, in_class_chunk in zip(query_chunk_sim.split(1024, -1), in_class_emb.split(1024)):
            query_chunk_proto += sim_chunk.matmul(in_class_chunk)
        query_proto.append(query_chunk_proto)
    query_proto = torch.cat(query_proto, dim=0)
    return query_proto


class SpanExtractor(nn.Module):

    def __init__(self, input_dim: int, output_dim: int,
                input_type:str, output_layer: str, 
                span_len_emb: bool, zero_proj_bias: bool,
                dropout: float, max_len: int):
        super(SpanExtractor, self).__init__()
        self.gradient_checkpointing = False
        self.input_type = input_type
        if input_type == 'x,y':
            proj_dim = input_dim * 2
        elif input_type == 'x,y,|x-y|':
            proj_dim = input_dim * 3
        else:
            raise Exception("Unknown input type")
        self.span_len_emb = span_len_emb

        self.dropout = nn.Dropout(dropout)
        if output_layer == "bypass":
            self.proj = nn.Identity()
            self.output = nn.Identity()
        else:
            self.proj = nn.Linear(proj_dim, output_dim)
            if zero_proj_bias:
                nn.init.zeros_(self.proj.bias)
            if output_layer == "layernorm":
                self.output = nn.LayerNorm(output_dim)
            elif output_layer == "raw":
                self.output = nn.Identity()
            else:
                raise Exception("Unknown output layer")

        if self.span_len_emb:
            self.subword_len_emb = nn.Embedding(max_len + 1, output_dim)
            nn.init.zeros_(self.subword_len_emb.weight)
            self.word_len_emb = nn.Embedding(max_len + 1, output_dim)
            nn.init.zeros_(self.word_len_emb.weight)
            self.max_len = max_len

    def _forward_impl(
        self,
        word_repr: torch.Tensor,
        span_label: torch.Tensor,
        gather_start: torch.Tensor,
        gather_end: torch.Tensor,
        span_slen: torch.Tensor,
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        span_mask = span_label != LabelEncoder.ignore_label_idx
        batch_id, sent_idx = torch.nonzero(span_mask, as_tuple=True)
        sent_start = gather_start[batch_id, sent_idx]
        sent_end = gather_end[batch_id, sent_idx]

        start = word_repr[batch_id, sent_start, :]
        end = word_repr[batch_id, sent_end, :]

        if self.input_type == 'x,y':
            span_rep = torch.cat([start, end], dim=-1)
        elif self.input_type == 'x,y,|x-y|':
            span_rep = torch.cat([start, end, (start - end).abs()], dim=-1)
        else:
            raise Exception("Unknown input type")

        span_rep = self.dropout(span_rep)
        span_rep = self.proj(span_rep)

        if self.span_len_emb:
            subword_len = torch.clamp(span_slen.masked_select(span_mask), 0, self.max_len)
            subword_len_emb = self.subword_len_emb(subword_len)

            span_len = sent_end - sent_start + 1
            word_len = torch.clamp(span_len, 0, self.max_len)
            word_len_emb = self.word_len_emb(word_len)

            span_rep = span_rep + subword_len_emb + word_len_emb

        span_rep = self.output(span_rep)
        return span_rep, batch_id, sent_idx, sent_start, sent_end
    
    
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        word_repr: torch.Tensor,
        span_label: torch.Tensor,
        gather_start: torch.Tensor,
        gather_end: torch.Tensor,
        span_slen: torch.Tensor,
    ):  
        if self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                word_repr,
                span_label,
                gather_start,
                gather_end,
                span_slen
            )
        else:
            return self._forward_impl(
                word_repr,
                span_label,
                gather_start,
                gather_end,
                span_slen
            )

class ProtoSpan(NERBaseModel):

    def __init__(
        self,
        task,
        backbone,
        dropout,
        bert_dropout,
        span_hidden,
        span_input,
        span_output,
        max_span_length,
        no_atten,
        no_cosine,
        cosine_temp,
        atten_cosine_temp,
        no_span_len_emb,
        add_proj_bias,
        lr,
        bert_lr,
        wd,
        train_step,
        seed,
        **kwargs,
    ):
        super(ProtoSpan, self).__init__()
        self.save_hyperparameters(
            "task",
            "backbone",
            "dropout",
            "bert_dropout",
            "span_hidden",
            "span_input",
            "span_output",
            "max_span_length",
            "no_atten",
            "no_cosine",
            "cosine_temp",
            "atten_cosine_temp",
            "no_span_len_emb",
            "add_proj_bias",
            "lr",
            "bert_lr",
            "wd",
            "train_step",
            "seed",
        )

        self.task = task
        pl.seed_everything(seed)
        config = BertConfig.from_pretrained(backbone)
        config.hidden_dropout_prob = bert_dropout
        config.attention_probs_dropout_prob = bert_dropout
        self.bert = BertModel.from_pretrained(backbone, config=config, add_pooling_layer=False)
        self.bert.gradient_checkpointing_enable()
        
        self.span_extractor = SpanExtractor(
            self.bert.config.hidden_size,
            span_hidden,
            span_input,
            span_output,
            not no_span_len_emb,
            not add_proj_bias,
            dropout,
            max_span_length
        )
        # ignore -100 in label encoder
        # if we only extract sample span loss should never see ignored label
        # but to unify training and inference, we extract full span and mask sample span
        self.loss_fn = nn.NLLLoss(ignore_index=-100)
        self.atten = not no_atten
        self.cosine = not no_cosine
        self.cosine_temp = cosine_temp
        self.atten_temp = atten_cosine_temp

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout ration for extractor")
        parser.add_argument("--bert_dropout", default=0.1, type=float, help="dropout for bert")
        parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
        parser.add_argument("--bert_lr", default=2e-5, type=float, help="learning rate")
        parser.add_argument("--wd", default=0.01, type=float, help="weight decay rate")
        parser.add_argument("--clip", default=1, type=float, help="max gradient norm for clippling")
        parser.add_argument("--span_hidden", default=768, type=int, help="span hidden")
        parser.add_argument("--span_input", default="x,y,|x-y|", type=str, help="span input option")
        parser.add_argument("--span_output", default="layernorm", type=str, help="span ouput option")
        parser.add_argument("--no_atten", action="store_true", help="add attention")
        parser.add_argument('--no_cosine', action='store_true', help='normalize embedding')
        parser.add_argument('--cosine_temp', default=160, type=float, help='cosine sclaling')
        parser.add_argument('--atten_cosine_temp', default=40, type=float, help='attention cosine sclaling')
        parser.add_argument('--add_proj_bias', action='store_true', help='init span proj bias to zero')
        parser.add_argument('--no_span_len_emb', action='store_true', help='add span length embed')
        return parser

    def proto(self, batch: Dict, training=False):
        assert self.task in ["all_avg"]
        outputs = self.bert(
            batch["bert"]["input_id"],
            attention_mask=batch["bert"]["atten_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        gather_index = batch["gather"]["bert"].unsqueeze(-1).expand(-1, -1, outputs["hidden_states"][-1].size(-1))
        token_rep = torch.gather(outputs["hidden_states"][-1], dim=1, index=gather_index)

        support_mask = (batch["query_mask"] == 0).bool()
        query_mask = (batch["query_mask"] == 1).bool()
        full_span = batch["span"]["full_span"]
        span_start = batch["span"]["gather_start"]
        span_end = batch["span"]["gather_end"]
        span_slen = batch["span"]["subword_len"]
        sample_span = batch["span"]["sample_span"]

        query_sims, query_preds, query_target = [], [], []
        query_batch_ids, query_gather_starts, query_gather_ends = [], [], []
        for bid in range(batch["bs"]):
            batch_mask = batch["batch_id"] == bid
            batch_support_idx = (batch_mask & support_mask).nonzero(as_tuple=True)[0]
            support_token_rep = token_rep.index_select(0, batch_support_idx)
            support_label = full_span.index_select(0, batch_support_idx)
            support_start = span_start.index_select(0, batch_support_idx)
            support_end = span_end.index_select(0, batch_support_idx)
            support_slen = span_slen.index_select(0, batch_support_idx)
            support_emb, support_batch_id, support_sent_idx, \
                support_gather_start, support_gather_end = self.span_extractor(
                    support_token_rep,
                    support_label,
                    support_start,
                    support_end,
                    support_slen)
            
            support_label = support_label[support_batch_id, support_sent_idx]

            batch_query_idx = (batch_mask & query_mask).nonzero(as_tuple=True)[0]
            query_token_rep = token_rep.index_select(0, batch_query_idx)
            if training:
                query_label = sample_span.index_select(0, batch_query_idx)
            else:
                query_label = full_span.index_select(0, batch_query_idx)
            query_start = span_start.index_select(0, batch_query_idx)
            query_end = span_end.index_select(0, batch_query_idx)
            query_slen = span_slen.index_select(0, batch_query_idx)
            query_emb, query_batch_id, query_sent_idx, \
                query_gather_start, query_gather_end = self.span_extractor(
                query_token_rep,
                query_label,
                query_start,
                query_end,
                query_slen)
            query_batch_ids.append(query_batch_id)
            query_gather_starts.append(query_gather_start)
            query_gather_ends.append(query_gather_end)

            query_label = query_label[query_batch_id, query_sent_idx]
            
            if self.cosine:
                query_emb = F.normalize(query_emb)
                support_emb = F.normalize(support_emb)
            
            query_sim = []
            # does not matter span label is alway io
            if batch["mode"] in ["io", "bio"]:
                # labelencoder assign idx in order
                # always use io internally
                # we ignore too long spans even if it is in the support
                support_types = batch["support_span_types"][bid]
                label_encoder = LabelEncoder(batch["episode_types"][bid], "io")
                type_ids = [
                    (lid, label)
                    for lid, label in label_encoder.idx_to_item.items()
                    if label != LabelEncoder.ignore_label
                ]
                for lid, label in type_ids:
                    if label != "O" and label not in support_types:
                        # episode_types is always used to construct label encoder
                        # and decode output label ids; therefore we need to have prediction
                        # even when there is not corresponding support
                        assert lid != 0
                        in_class_sim = torch.full_like(query_sim[0], -1000)
                        query_sim.append(in_class_sim)
                        continue
                    in_class_emb = support_emb[support_label == lid]
                    assert in_class_emb.size(0) > 0
                    if self.atten:
                        atten_scaling = self.atten_temp if self.cosine else 1.0
                        if query_emb.size(0) > 10240 and not self.training:
                            proto = big_atten(query_emb, in_class_emb, atten_scaling)
                        else:
                            atten_weight = (query_emb.matmul(in_class_emb.T) * atten_scaling).softmax(-1)
                            proto = atten_weight.matmul(in_class_emb)
                    else:
                        proto = in_class_emb.mean(0, keepdim=True)
                    
                    if self.cosine:
                        in_class_sim = F.cosine_similarity(query_emb, proto) * self.cosine_temp
                    else:
                        in_class_sim = -(query_emb - proto).pow(2).sum(-1)
                    
                    query_sim.append(in_class_sim)
            else:
                raise RuntimeError("unknown mode")

            query_sim = torch.stack(query_sim, dim=-1)
            query_logit = F.log_softmax(query_sim, dim=-1)
            query_sims.append(query_logit)
            _, pred = torch.max(query_sim, dim=-1)
            query_preds.append(pred)
            query_target.append(query_label)

        return {
            "sim": query_sims,
            "pred": query_preds,
            "target": query_target,
            "batch_id": query_batch_ids,
            "gather_start": query_gather_starts,
            "gather_end": query_gather_ends,
        }

    def training_step(self, batch: Dict, batch_idx: int):
        output = self.proto(batch, training=True)
        episode_loss = []
        for query_sim, query_span_label in zip(output["sim"], output["target"]):
            loss_sim = query_sim.view(-1, query_sim.size(-1))
            loss_target = query_span_label.view(-1)
            episode_loss.append(self.loss_fn(loss_sim, loss_target))
        
        loss = torch.stack(episode_loss).mean()
        self.log("train/loss", loss, batch_size=batch["bs"], prog_bar=True)

        return loss

    def forward(self, batch):
        output = self.proto(batch, training=False)
        res = []
        
        def NMS(cand):

            def conflict_judge(line_x, line_y):
                if line_x[0] == line_y[0]:
                    return True
                if line_x[0] < line_y[0]:
                    if line_x[1] >= line_y[0]:
                        return True
                if line_x[0] > line_y[0]:
                    if line_x[0] <= line_y[1]:
                        return True
                return False

            filter_list = []
            for elem in sorted(cand, key=lambda x: -x[3]):
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append(elem)

            return filter_list        

        for bid in range(batch["bs"]):
            span_pred_logit, span_pred = output["sim"][bid].max(-1)
            batch_id = output["batch_id"][bid]
            gather_start, gather_end = output["gather_start"][bid], output["gather_end"][bid]
            location_info = torch.stack([span_pred, batch_id, gather_start, gather_end])
            word_len = mask_select(batch["gather"]["min_len"][bid], batch["is_query"][bid])
            span_cands = defaultdict(list)

            NonO_span_idx = span_pred != LabelEncoder.default_label_idx
            NonO_span_pos = location_info[:, NonO_span_idx]
            NonO_span_logit = span_pred_logit[NonO_span_idx]
            for logit, label, i, j, k in zip(NonO_span_logit.cpu().tolist(), *NonO_span_pos.cpu().tolist()):
                assert label != LabelEncoder.default_label_idx
                span_cands[i].append((j, k, label, logit))

            pred = [[LabelEncoder.default_label] * seq_len for seq_len in word_len]
            # span label encoder is always io
            label_encoder = LabelEncoder(batch["episode_types"][bid], "io")
            for i, span_cand in span_cands.items():
                decoded_cand = NMS(span_cand)
                span_span = [(j, k, label_encoder.decode(coded_label)) for j, k, coded_label, logit in decoded_cand]
                if batch["mode"] == "io":
                    pred[i] = span2io(pred[i], span_span)
                elif batch["mode"] == "bio":
                    pred[i] = span2bio(pred[i], span_span)
                else:
                    raise RuntimeError("unknown mode")

            episode_labels, episode_start, episode_end = batch["label"][bid], batch["start"][bid], batch["end"][bid]
            assert len(episode_labels) == len(episode_start) == len(episode_end)
            episode_targets = [para[s:e] for para, s, e in zip(episode_labels, episode_start, episode_end)]

            res.append(
                {
                    "pred": pred,
                    "target": mask_select(episode_targets, batch["is_query"][bid]),
                    "input": {
                        "sent_id": mask_select(batch["sent_id"][bid], batch["is_query"][bid]),
                        "chunk_id": mask_select(batch["chunk_id"][bid], batch["is_query"][bid]),
                        "mode": batch["mode"],
                    },
                }
            )
        return res

    def configure_optimizers(self):
        bert_parameters, else_parameters = [], []
        for n, p in self.named_parameters():
            if n.startswith("bert"):
                bert_parameters.append((n, p))
            else:
                else_parameters.append((n, p))
        return super().configure_optimizers(
            [
                (bert_parameters, self.hparams.bert_lr, self.hparams.wd),
                (else_parameters, self.hparams.lr, self.hparams.wd),
            ]
        )


# ref masked_log_softmax from allennlp https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py#L286
class MaskLogsoftmax(nn.Module):

    def __init__(self, types, span_encoder=False):
        super().__init__()
        self.decoders = {}
        if not span_encoder:
            self.total_mask_len = sum(dataset_types.encoder_len for dataset_types in types.values())
            mask = torch.zeros((len(types), self.total_mask_len), dtype=torch.bool)
            for source, obj in types.items():
                mask[obj.source_id, obj.shift : obj.shift + obj.encoder_len] = True
                self.decoders[source] = obj.get_label_encoder()
        else:
            self.total_mask_len = sum(dataset_types.span_encoder_len for dataset_types in types.values())
            mask = torch.zeros((len(types), self.total_mask_len), dtype=torch.bool)
            for source, obj in types.items():
                mask[obj.source_id, obj.span_shift : obj.span_shift + obj.span_encoder_len] = True
                self.decoders[source] = obj.get_span_label_encoder()
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, inputs: torch.Tensor, mask_id: torch.Tensor, training=True):
        mask = self.mask.index_select(0, mask_id)

        # mask true means keep false means ignore
        bsz_seq_len, class_num = inputs.size()
        assert mask.size(0) == bsz_seq_len and mask.size(1) == class_num, (mask.size(), inputs.size())
        if training:
            masked_inputs = inputs.masked_fill(~mask, inputs.detach().min() - 20)
            # fp16 cannot express exp(-20) it will become zero directly **underflow**
            return torch.log_softmax(masked_inputs, dim=-1, dtype=torch.float32).to(inputs.dtype)
        else:
            return inputs.masked_fill(~mask, torch.finfo(inputs.dtype).min)


class MultiSpan(NERBaseModel):
    # use **kwargs to absorb ununsed args
    def __init__(
        self,
        task,
        backbone,
        types,
        dropout,
        bert_dropout,
        span_hidden,
        span_input,
        span_output,
        max_span_length,
        no_span_len_emb,
        add_proj_bias,
        lr,
        bert_lr,
        wd,
        train_step,
        seed,
        **kwargs,
    ):
        super(MultiSpan, self).__init__()
        self.save_hyperparameters(
            "task",
            "backbone",
            "types",
            "dropout",
            "bert_dropout",
            "span_hidden",
            "span_input",
            "span_output",
            "max_span_length",
            "no_span_len_emb",
            "add_proj_bias",
            "lr",
            "bert_lr",
            "wd",
            "train_step",
            "seed",
        )
        self.task = task
        pl.seed_everything(seed)
        config = BertConfig.from_pretrained(backbone)
        config.hidden_dropout_prob = bert_dropout
        config.attention_probs_dropout_prob = bert_dropout
        self.bert = BertModel.from_pretrained(backbone, config=config, add_pooling_layer=False)

        # dropout layer for bert
        self.dropout = nn.Dropout(dropout)
        self.mask_logsoftmax = MaskLogsoftmax(types, True)
        self.span_extractor = SpanExtractor(
            self.bert.config.hidden_size,
            span_hidden,
            span_input,
            span_output,
            not no_span_len_emb,
            not add_proj_bias,
            dropout,
            max_span_length
        )
        self.classifier = nn.Linear(span_hidden, self.mask_logsoftmax.total_mask_len)
        self.loss_fn = nn.NLLLoss()  # ignore -100 in loss

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument("--dropout", default=0.1, type=float, help="dropout for final token embedding")
        parser.add_argument("--bert_dropout", default=0.1, type=float, help="dropout for bert")
        parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
        parser.add_argument("--bert_lr", default=2e-5, type=float, help="learning rate")
        parser.add_argument("--wd", default=0.01, type=float, help="weight decay rate")
        parser.add_argument("--clip", default=1, type=float, help="max gradient norm for clippling")
        parser.add_argument("--span_hidden", default=768, type=int, help="span hidden")
        parser.add_argument("--span_input", default="x,y,|x-y|", type=str, help="span input option")
        parser.add_argument("--span_output", default="layernorm", type=str, help="span ouput option")
        parser.add_argument('--add_proj_bias', action='store_true', help='init span proj bias to zero')
        parser.add_argument('--no_span_len_emb', action='store_true', help='add span length embed')
        return parser

    def predict(self, batch):
        assert self.task in ["multi_supervised"]
        outputs = self.bert(
            batch["bert"]["input_id"],
            attention_mask=batch["bert"]["atten_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        gather_index = batch["gather"]["bert"].unsqueeze(-1).expand(-1, -1, outputs["hidden_states"][-1].size(-1))
        token_rep = torch.gather(outputs["hidden_states"][-1], dim=1, index=gather_index)
        
        span_label = batch["span"]["full_span"]
        span_start = batch["span"]["gather_start"]
        span_end = batch["span"]["gather_end"]
        span_slen = batch["span"]["subword_len"]
        span_rep, batch_id, sent_idx, sent_start, sent_end = self.span_extractor(token_rep, span_label, span_start, span_end, span_slen)
        span_logit = self.classifier(span_rep)
        return span_logit, batch_id, sent_idx, sent_start, sent_end

    def training_step(self, batch, batch_idx):
        span_logit, batch_id, sent_idx, sent_start, sent_end = self.predict(batch)
        span_label = batch['span']['sample_span'][batch_id, sent_idx]
        expand_source_id = batch["source_id"][batch_id]
        masked_span_logit = self.mask_logsoftmax(span_logit, expand_source_id)
        loss_logits = masked_span_logit.view(-1, span_logit.size(-1))
        loss_targets = span_label.view(-1)
        loss = self.loss_fn(loss_logits, loss_targets)
        self.log("train/loss", loss, batch_size=batch["bs"], prog_bar=True)
        return loss

    def forward(self, batch):
        def NMS(cand):

            def conflict_judge(line_x, line_y):
                if line_x[0] == line_y[0]:
                    return True
                if line_x[0] < line_y[0]:
                    if line_x[1] >= line_y[0]:
                        return True
                if line_x[0] > line_y[0]:
                    if line_x[0] <= line_y[1]:
                        return True
                return False

            filter_list = []
            for elem in sorted(cand, key=lambda x: -x[3]):
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append(elem)

            return filter_list


        span_logit, batch_id, sent_idx, sent_start, sent_end = self.predict(batch)
        expand_source_id = batch["source_id"][batch_id]
        span_logit = self.mask_logsoftmax(span_logit, expand_source_id, False)
        span_pred_logit, span_pred = span_logit.max(dim=-1)
        location_info = torch.stack([span_pred, batch_id, sent_start, sent_end])

        span_cands = defaultdict(list)
        sent_source = batch["source"]
        span_pred_logit = span_pred_logit.cpu().tolist()
        location_info = location_info.cpu().tolist()
        for logit, coded_label, i, j, k in zip(span_pred_logit, *location_info):
            label_encoder = self.mask_logsoftmax.decoders[sent_source[i]]
            label = label_encoder.decode(coded_label)
            if label != ShiftLabelEncoder.default_label:
                span_cands[i].append((j, k, label, logit))

        word_len = batch['gather']['min_len']
        batch_preds = [[ShiftLabelEncoder.default_label] * seq_len for seq_len in word_len]
        for i, span_cand in span_cands.items():            
            decoded_cand = NMS(span_cand)
            span_span = [(j, k, label) for j, k, label, logit in decoded_cand]
            if batch["mode"] == "io":
                batch_preds[i] = span2io(batch_preds[i], span_span)
            elif batch["mode"] == "bio":
                batch_preds[i] = span2bio(batch_preds[i], span_span)
            else:
                raise RuntimeError("unknown mode")

        batch_labels, batch_start, batch_end = batch["label"], batch["start"], batch["end"]
        assert len(batch_labels) == len(batch_start) == len(batch_end)
        batch_targets = [para[s:e] for para, s, e in zip(batch_labels, batch_start, batch_end)]

        return {
            "pred": batch_preds,
            "target": batch_targets,
            "source": batch["source"],
            "input": {
                "sent_id": batch["sent_id"],
                "chunk_id": batch["chunk_id"],
                "mode": batch["mode"],
            },
        }

    def configure_optimizers(self):
        bert_parameters, else_parameters = [], []
        for n, p in self.named_parameters():
            if n.startswith("bert"):
                bert_parameters.append((n, p))
            else:
                else_parameters.append((n, p))
        return super().configure_optimizers(
            [
                (bert_parameters, self.hparams.bert_lr, self.hparams.wd),
                (else_parameters, self.hparams.lr, self.hparams.wd),
            ]
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from fs_ner_utils import (
        load_episode,
        load_sent,
        init_argparser,
        add_train_specific_args,
        add_fewshot_specific_args,
        CDEBertTokenizer
    )
    from itertools import islice

    parser = ArgumentParser()
    parser = init_argparser(parser)
    parser = add_fewshot_specific_args(parser)
    parser = add_train_specific_args(parser)
    parser.add_argument("--task", type=str, required=True, help="model mode")
    parser.add_argument("--model", type=str, required=True, help="model name")

    args, _ = parser.parse_known_args()

    if args.model == "protospan":
        model_class = ProtoSpan
    elif args.model == "multispan":
        model_class = MultiSpan
    else:
        raise Exception("Unknown model type")

    parser = model_class.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.task in ["all_avg", "episode_avg"]:
        train_dataset, train_data_loader = load_episode("SolidState", args, training=True)
        dev_dataset, dev_data_loader = load_episode("SolidState", args, training=False)
        model = model_class(**vars(args), types=train_dataset.types)
    elif args.task in ["multi_supervised"]:
        train_dataset, train_data_loader, train_types = load_sent(["Catalysis", "SolidState"], "train", args, training=True)
        dev_dataset, dev_data_loader, dev_types = load_sent(["Catalysis", "SolidState"], "test", args, training=False)
        model = model_class(**vars(args), types=train_types)
    else:
        raise RuntimeError("unknown mode")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    with torch.autograd.anomaly_mode.detect_anomaly():
        for batch in islice(train_data_loader, 0, 20):
            loss = model.training_step(batch, 0)

    model.eval()
    with torch.no_grad():
        result = []
        for batch in islice(dev_data_loader, 20):
            result.append(model.test_step(batch, 0))
        model.on_test_epoch_end()