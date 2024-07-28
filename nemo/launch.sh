#!/bin/bash

# MODEL="/scripts/LibriSpeech/inter_unit_new/results/checkpoints/megatron_llama_sft.nemo"
MODEL="/checkpoints/synthetic/results/checkpoints/megatron_llama_sft.nemo"

bash -c "\
export TRANSFORMERS_OFFLINE=1 
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0
CUDA_DEVICE_MAX_CONNECTIONS=1 python launch.py \
--config-path=/opt/NeMo/examples/nlp/language_modeling/conf \
--config-name=megatron_gpt_inference \
inference.top_k=20 \
inference.top_p=0 \
inference.temperature=1 \
inference.tokens_to_generate=300 \
inference.end_strings=\"['<|begin_of_text|>', '<|eot_id|>', '<|end_of_text|>']\" \
inference.repetition_penalty=-1 \
trainer.devices=1 \
trainer.precision=bf16-mixed \
tensor_model_parallel_size=1 \
pipeline_model_parallel_size=1 \
pipeline_model_parallel_split_rank=1 \
gpt_model_file=$MODEL \
server=True \
port=8889"

# > /dev/null 2>&1