# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

import difw


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        tasks=None,
        **kwargs,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(
            in_features, out_features, **kwargs)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            # Merge the weights and mark it
            self.linear.weight.data += (self.lora_B @
                                        self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)) * self.scaling
        return pretrained + lora


class DiTASKLinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = 0.0,
        tasks=None,
        trainable_scale_shared=False,
        trainable_scale_per_task=False,
        shared_mode=None,
        **kwargs,
    ):
        has_tasks = tasks is not None

        if isinstance(r, int):
            r = {'shared': r}
        super().__init__(
            r=r['shared'], lora_alpha=lora_shared_scale, lora_dropout=lora_dropout)

        self.linear = torch.nn.Linear(
            in_features, out_features, **kwargs)

        self.tasks = tasks
        if r['shared'] > 0:
            if has_tasks:
                self.cpab_tasks = {
                    task: difw.Cpab(tess_size=r[task], backend='pytorch', device='gpu', zero_boundary=True)
                    for task in tasks
                }
                self.lora_tasks_A = nn.ParameterDict({
                    task: nn.Parameter(torch.zeros_like(self.cpab_tasks[task].sample_transformation(1)), requires_grad=True)
                    for task in tasks
                })
                if trainable_scale_per_task:
                    self.lora_task_scale = nn.ParameterDict({
                        task: nn.Parameter(torch.FloatTensor(
                            [lora_task_scale]))
                        for task in tasks
                    })
                else:
                    self.lora_task_scale = {task: lora_task_scale[task]
                                            for task in tasks}

            self.T = difw.Cpab(tess_size=r['shared'], backend='pytorch', device='gpu' ,zero_boundary=True)
            self.lora_shared_A = nn.Parameter(torch.zeros_like(self.T.sample_transformation(1)), requires_grad=True)

            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_shared_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_shared_B)
        if hasattr(self, "lora_tasks_A"):
            for task in self.tasks:
                nn.init.kaiming_uniform_(
                    self.lora_tasks_A[task], a=math.sqrt(5))
                # nn.init.zeros_(self.lora_tasks_B[task])
    
    def transform_data(self, T, w, theta):
        U, S, V = torch.svd(w)
        with torch.autocast("cuda", dtype=torch.float32, enabled=True):
            S_t = T.transform_data(S.view(1, -1, 1), theta, outsize=S.shape[0]).view(-1)
        w_deformed = U @ torch.diag_embed(S_t) @ V.T
        return w_deformed

    def forward(self, x: torch.Tensor, x_tasks: Dict[str, torch.Tensor] = None):
        pretrained = self.linear(x)
        if self.r == 0:
            return pretrained, None
        x = self.lora_dropout(x)
        w = self.linear.weight
        do, di = w.shape
        if do == 2 * di:
            wk, wv = torch.tensor_split(w, 2)
            wk_deformed = self.transform_data(self.T, wk, self.lora_shared_A)
            wv_deformed = self.transform_data(self.T, wv, self.lora_shared_A)
            w_deformed = torch.cat([wk_deformed, wv_deformed], dim=0)
        elif do == 3 * di:
            wq, wk, wv = torch.tensor_split(w, 3)
            wq_deformed = self.transform_data(self.T, wq, self.lora_shared_A)
            wk_deformed = self.transform_data(self.T, wk, self.lora_shared_A)
            wv_deformed = self.transform_data(self.T, wv, self.lora_shared_A)
            w_deformed = torch.cat([wq_deformed, wk_deformed, wv_deformed], dim=0)
        else:
            w_deformed = self.transform_data(self.T, w, self.lora_shared_A)
        lora = F.linear(x, w_deformed, bias=self.linear.bias)
        if self.tasks is not None:
            lora_tasks = {}
            for task in self.tasks:
                x_task = x if x_tasks is None else x_tasks[task]
                T = self.cpab_tasks[task]
                theta = self.lora_tasks_A[task]
                if do == 2 * di:
                    wk, wv = torch.tensor_split(w, 2)
                    wk_deformed = self.transform_data(T, wk, theta)
                    wv_deformed = self.transform_data(T, wv, theta)
                    w_deformed = torch.cat([wk_deformed, wv_deformed], dim=0)
                elif do == 3 * di:
                    wq, wk, wv = torch.tensor_split(w, 3)
                    wq_deformed = self.transform_data(T, wq, theta)
                    wk_deformed = self.transform_data(T, wk, theta)
                    wv_deformed = self.transform_data(T, wv, theta)
                    w_deformed = torch.cat([wq_deformed, wk_deformed, wv_deformed], dim=0)
                else:
                    w_deformed = self.transform_data(T, w, theta)
                lora_task = F.linear(x_task, w_deformed, bias=self.linear.bias)
                lora_tasks[task] = lora_task
        else:
            lora_tasks = None

        return lora, lora_tasks

def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none", freeze_patch_embed: bool = False, freeze_norm: bool = False, free_relative_bias: bool = False, freeze_downsample_reduction=False) -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    def lora_filter(key): return "lora_" in key
    def patch_embed_filter(
        key): return not freeze_patch_embed and "patch_embed" in key

    def norm_filter(key): return not freeze_norm and "norm" in key

    def downsample_reduction_filter(
        key): return not freeze_downsample_reduction and "downsample.reduction" in key

    def relative_position_bias_filter(
        key): return not free_relative_bias and "relative_position_bias_table" in key

    def all_filters(key):
        return lora_filter(key) or patch_embed_filter(key) or norm_filter(key) or downsample_reduction_filter(key) or relative_position_bias_filter(key)

    print(f"LoRA bias mode: {bias}")
    print(f"LoRA Freeze patch_embed: {freeze_patch_embed}")
    print(f"LoRA Freeze norm: {freeze_norm}")
    print(f"LoRA Freeze downsample_reduction: {freeze_downsample_reduction}")
    print(f"LoRA Freeze relative_position_bias: {free_relative_bias}")
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if not all_filters(n):
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


def merge_lora_weights(model) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str, split_qkv: bool = False) -> Dict:
    unmatched_keys = []
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        print(full_checkpoint_name)
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            weights = state_dict.pop(
                full_checkpoint_name)
            last_four = ".".join(full_attribute_name.split(".")[-4:])
            if split_qkv and last_four in ["attn.qkv.linear.weight", "attn.qkv.linear.bias"]:
                w_q, w_k, w_v = torch.chunk(weights, chunks=3)
                weight_bias = last_four.split(".")[-1]
                full_attribute_name_without_suffix = ".".join(full_attribute_name.split(".")[
                    :-2])
                state_dict[f"{full_attribute_name_without_suffix}.q.linear.{weight_bias}"] = w_q
                state_dict[f"{full_attribute_name_without_suffix}.k.linear.{weight_bias}"] = w_k
                state_dict[f"{full_attribute_name_without_suffix}.v.linear.{weight_bias}"] = w_v
            else:
                state_dict[full_attribute_name] = weights
        else:
            unmatched_keys.append(checkpoint_name)
    if len(unmatched_keys) > 0:
        print(
            f"WARNING: The following keys from the checkpoint were not mapped: {unmatched_keys}")
    return state_dict