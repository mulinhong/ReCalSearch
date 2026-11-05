"""Training utilities: dataset, collation, pooling, losses, and trainer.

Supports all experiment variants:
  - in-batch / BM25 / dense / hybrid hard negatives
  - dual-view (raw + normalized code)
  - mean / last-token / attention / query-aware pooling
"""

import json
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer

from code_normalizer import normalize_code


class Text2CodeDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _tokens_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(t) for t in value)
    raise TypeError(f"Unsupported field type: {type(value)}")


def _extract_text_code(item: Any) -> Tuple[str, str]:
    if isinstance(item, dict):
        for text_key, code_key in [
            ("text", "code"),
            ("docstring_tokens", "code_tokens"),
            ("doc", "code_tokens"),
            ("docstring", "code"),
        ]:
            if text_key in item and code_key in item:
                return _tokens_to_text(item[text_key]), _tokens_to_text(item[code_key])
        raise KeyError(f"Unsupported keys: {list(item.keys())}")
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return _tokens_to_text(item[0]), _tokens_to_text(item[1])
    raise TypeError(f"Unsupported item type: {type(item)}")


class MeanPooling(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        m = mask.unsqueeze(-1).expand(hidden.size()).float()
        return torch.sum(hidden * m, dim=1) / torch.clamp(m.sum(dim=1), min=1e-6)


class LastTokenPooling(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        seq_lengths = mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, seq_lengths.long()]


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        scores = self.attn(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)


class QueryAwarePooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        query_embed: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if query_embed is None:
            m = mask.unsqueeze(-1).expand(hidden.size()).float()
            return torch.sum(hidden * m, dim=1) / torch.clamp(m.sum(dim=1), min=1e-6)
        projected = self.projection(hidden)
        scores = torch.bmm(projected, query_embed.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)


def build_pooling(mode: str, hidden_dim: int = 768) -> nn.Module:
    if mode == "mean":
        return MeanPooling()
    if mode == "last_token":
        return LastTokenPooling()
    if mode == "attention":
        return AttentionPooling(hidden_dim)
    if mode == "query_aware":
        return QueryAwarePooling(hidden_dim)
    raise ValueError(f"Unknown pooling mode: {mode}")


def collate_fn(
    batch: List[Any],
    tokenizer: PreTrainedTokenizer,
    use_dual_view: bool = False,
    num_hard_negatives: int = 0,
    max_length: int = 512,
    query_prefix: str = "",
    code_prefix: str = "",
) -> Dict[str, Any]:
    pairs = [_extract_text_code(item) for item in batch]
    queries = [query_prefix + p[0] for p in pairs]
    codes_raw = [code_prefix + p[1] for p in pairs]

    query_tensor = tokenizer(
        queries,
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt",
    )
    code_raw_tensor = tokenizer(
        codes_raw,
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt",
    )

    result = {
        "anchor": query_tensor,
        "positive": code_raw_tensor,
        "labels": torch.arange(len(queries), dtype=torch.long),
    }

    if use_dual_view:
        codes_norm = [code_prefix + normalize_code(p[1]) for p in pairs]
        code_norm_tensor = tokenizer(
            codes_norm,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        )
        result["positive_norm"] = code_norm_tensor

    if num_hard_negatives > 0:
        hard_neg_codes: List[List[str]] = []
        hard_neg_mask: List[List[bool]] = []

        for item_idx, item in enumerate(batch):
            fallback_code = pairs[item_idx][1]
            if isinstance(item, dict) and item.get("hard_negative_codes"):
                negs = [
                    _tokens_to_text(neg)
                    for neg in item.get("hard_negative_codes", [])[:num_hard_negatives]
                    if neg
                ]
                valid = [True] * len(negs)
                while len(negs) < num_hard_negatives:
                    negs.append(fallback_code)
                    valid.append(False)
                hard_neg_codes.append(negs[:num_hard_negatives])
                hard_neg_mask.append(valid[:num_hard_negatives])
            else:
                hard_neg_codes.append([fallback_code] * num_hard_negatives)
                hard_neg_mask.append([False] * num_hard_negatives)

        for neg_idx in range(num_hard_negatives):
            neg_texts = [code_prefix + hns[neg_idx] for hns in hard_neg_codes]
            neg_tensor = tokenizer(
                neg_texts,
                truncation=True,
                max_length=max_length,
                padding="longest",
                return_tensors="pt",
            )
            result[f"hard_negative_{neg_idx}"] = neg_tensor

        result["hard_negative_mask"] = torch.tensor(hard_neg_mask, dtype=torch.bool)

    return result


class DebiasedContrastiveTrainer(Trainer):
    def __init__(
        self,
        *args,
        pooling_module: nn.Module = None,
        temperature: float = 1.0,
        num_hard_negatives: int = 0,
        lambda_hn: float = 0.25,
        use_dual_view: bool = False,
        lambda_norm: float = 1.0,
        lambda_consistency: float = 0.5,
        warmup_hn_ratio: float = 0.0,
        hn_loss_mode: str = "separate",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.pooling_module = pooling_module or MeanPooling()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives
        self.lambda_hn = lambda_hn
        self.use_dual_view = use_dual_view
        self.lambda_norm = lambda_norm
        self.lambda_consistency = lambda_consistency
        self.warmup_hn_ratio = warmup_hn_ratio
        self.hn_loss_mode = hn_loss_mode
        self.label_names = ["labels"]

    def _get_embeds(
        self,
        model: PreTrainedModel,
        batch: Dict,
        field: str,
        query_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        ids = batch[field]["input_ids"]
        mask = batch[field]["attention_mask"]
        hidden = model(ids, attention_mask=mask)[0]
        pooling = getattr(model, "pooling_head", self.pooling_module)
        return pooling(hidden, mask, query_embed=query_embed)

    def _warmup_done(self) -> bool:
        if self.warmup_hn_ratio <= 0:
            return True
        max_steps = getattr(self.state, "max_steps", 0)
        if max_steps is None or max_steps <= 0:
            return True
        return self.state.global_step >= int(max_steps * self.warmup_hn_ratio)

    def compute_loss(self, model, batch, num_items_in_batch=None, return_outputs=False):
        anchor_embeds = self._get_embeds(model, batch, "anchor")
        pos_embeds_raw = self._get_embeds(model, batch, "positive", query_embed=anchor_embeds)

        anchor_norm = F.normalize(anchor_embeds, dim=1)
        pos_raw_norm = F.normalize(pos_embeds_raw, dim=1)
        sim_raw = (anchor_norm @ pos_raw_norm.T) / self.temperature
        labels = batch["labels"].to(sim_raw.device)
        loss_raw = F.cross_entropy(sim_raw, labels)
        loss = loss_raw
        aux_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)

        if self.use_dual_view and "positive_norm" in batch:
            pos_embeds_norm = self._get_embeds(model, batch, "positive_norm", query_embed=anchor_embeds)
            pos_norm_norm = F.normalize(pos_embeds_norm, dim=1)
            sim_norm = (anchor_norm @ pos_norm_norm.T) / self.temperature
            loss_norm = F.cross_entropy(sim_norm, labels)
            loss_consistency = 1.0 - F.cosine_similarity(pos_raw_norm, pos_norm_norm, dim=1).mean()
            aux_loss = self.lambda_norm * loss_norm + self.lambda_consistency * loss_consistency
            loss = loss + aux_loss

        if self.num_hard_negatives > 0 and self._warmup_done():
            neg_fields = [
                f"hard_negative_{i}"
                for i in range(self.num_hard_negatives)
                if f"hard_negative_{i}" in batch
            ]
            if neg_fields:
                neg_embeds_list = []
                for field in neg_fields:
                    ne = self._get_embeds(model, batch, field, query_embed=anchor_embeds)
                    neg_embeds_list.append(F.normalize(ne, dim=1))

                if self.hn_loss_mode == "unified":
                    neg_scores = []
                    hard_mask = batch.get("hard_negative_mask")
                    hard_mask = hard_mask.to(anchor_norm.device) if hard_mask is not None else None
                    for neg_idx, ne in enumerate(neg_embeds_list):
                        scores = anchor_norm @ ne.T
                        if hard_mask is not None:
                            valid_cols = hard_mask[:, neg_idx].bool().unsqueeze(0)
                            scores = scores.masked_fill(~valid_cols, torch.finfo(scores.dtype).min)
                        neg_scores.append(scores)
                    logits = torch.cat([anchor_norm @ pos_raw_norm.T] + neg_scores, dim=1) / self.temperature
                    loss = loss + self.lambda_hn * F.cross_entropy(logits, labels)
                else:
                    pos_scores = torch.sum(anchor_norm * pos_raw_norm, dim=1, keepdim=True) / self.temperature
                    neg_scores = torch.cat(
                        [
                            torch.sum(anchor_norm * ne, dim=1, keepdim=True) / self.temperature
                            for ne in neg_embeds_list
                        ],
                        dim=1,
                    )
                    hard_mask = batch.get("hard_negative_mask")
                    if hard_mask is not None:
                        hard_mask = hard_mask.to(neg_scores.device).bool()
                        neg_scores = neg_scores.masked_fill(~hard_mask, torch.finfo(neg_scores.dtype).min)
                        valid_rows = hard_mask.any(dim=1)
                    else:
                        valid_rows = torch.ones(neg_scores.shape[0], dtype=torch.bool, device=neg_scores.device)

                    if valid_rows.any():
                        logits_hard = torch.cat([pos_scores, neg_scores], dim=1)[valid_rows]
                        labels_hard = torch.zeros(logits_hard.shape[0], dtype=torch.long, device=logits_hard.device)
                        loss_hard = F.cross_entropy(logits_hard, labels_hard)
                        loss = loss + self.lambda_hn * loss_hard

        return (loss, sim_raw) if return_outputs else loss
