# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# NPU-native structured output bitmask application for the v1 model runner.
#
# Mirrors the convention of ``vllm_ascend/worker/v2/structured_outputs.py``:
# apply the bitmask on the NPU through a Triton kernel, with
# ``BLOCK_SIZE=8192`` and ``BLOCK_SIZE_SUB=1024`` tiling to stay within the
# NPU UB. Kept parallel to v2 so neither runner cross-imports the other.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch


@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    BLOCK_SIZE_SUB: tl.constexpr = 1024
    bitmask_idx = tl.program_id(0)
    block_id = tl.program_id(1)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    for sub_offset in tl.range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        global_token_offset = block_id * BLOCK_SIZE + sub_offset
        bitmask_word_start = global_token_offset // 32
        bitmask_offset = bitmask_word_start + tl.arange(0, BLOCK_SIZE_SUB // 32)
        packed_bitmask = tl.load(
            bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
            mask=bitmask_offset < bitmask_stride,
            other=0,
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE_SUB)

        block_offset = global_token_offset + tl.arange(0, BLOCK_SIZE_SUB)
        tl.store(
            logits_ptr + logits_idx * logits_stride + block_offset,
            -float("inf"),
            mask=bitmask & (block_offset < vocab_size),
        )


def apply_grammar_bitmask(
    scheduler_output: "SchedulerOutput",
    grammar_output: "GrammarOutput",
    input_batch: "InputBatch",
    logits: torch.Tensor,
) -> None:
    """Apply grammar bitmask to ``logits`` in-place on the NPU.

    Same signature as the upstream helper; the body runs the bitmask kernel
    fully on-device, so the original ``logits.dtype`` is preserved.
    """
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens

    # First pass: logit index of each structured request in the batch.
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    # Second pass: collect (logit_row, bitmask_row) pairs in scheduler order.
    out_indices: list[int] = []
    bitmask_row_indices: list[int] = []
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        logit_index = struct_out_req_batch_indices.get(req_id)
        if logit_index is not None:
            for i in range(1 + num_spec_tokens):
                out_indices.append(logit_index + i)
                bitmask_row_indices.append(cumulative_index + i)
        cumulative_index += 1 + num_spec_tokens

    if not out_indices:
        return

    logits_indices = torch.tensor(
        out_indices, dtype=torch.int32, pin_memory=PIN_MEMORY,
    ).to(logits.device, non_blocking=True)

    # Only the bitmask rows actually used are uploaded to the NPU.
    grammar_bitmask = torch.from_numpy(
        grammar_output.grammar_bitmask[bitmask_row_indices],
    ).to(logits.device, non_blocking=True)
    assert grammar_bitmask.is_contiguous()

    num_masks = len(out_indices)
    vocab_size = logits.shape[-1]
    BLOCK_SIZE = 8192
    grid = (num_masks, cdiv(vocab_size, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        grammar_bitmask,
        grammar_bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
