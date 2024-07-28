cd /opt/NeMo
TP=4
PP=1
MODEL=/scripts/original_llama3_instruct/llama3.nemo
GLOBAL_BATCH_SIZE=32
WARMUP=100
LR=1e-4

TRAIN_DS="[/scripts/synthetic_data/data/json/train.jsonl]"
VALID_DS="[/scripts/synthetic_data/data/json/valid.jsonl]"
TEST_DS="[/scripts/synthetic_data/data/json/test.jsonl]"
CONCAT_SAMPLING_PROBS="[1.0]"
export TOKENIZERS_PARALLELISM=false

CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -u /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
    --config-path=/scripts/ \
    --config-name=squad.yaml \
    run.base_results_dir="/checkpoints/synthetic" \
    trainer.val_check_interval=0.2 \
    trainer.max_steps=400 \
    trainer.max_epochs=2 \
    trainer.precision=bf16 \
    trainer.num_nodes=1 \
    trainer.devices=4 \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    exp_manager.resume_if_exists=True \
    exp_manager.checkpoint_callback_params.save_best_model=False \
    exp_manager.checkpoint_callback_params.save_top_k=1 \
    +exp_manager.create_tensorboard_logger=True \
    model.tensor_model_parallel_size=${TP} \
    model.pipeline_model_parallel_size=${PP} \
    model.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.micro_batch_size=1 \
    model.restore_from_path=${MODEL} \
    model.data.train_ds.file_names=${TRAIN_DS} \
    model.data.validation_ds.file_names=${VALID_DS} \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.train_ds.num_workers=4 \
    model.data.validation_ds.num_workers=4 \
    model.data.test_ds.num_workers=4 \
    model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
    model.data.train_ds.max_seq_length=8192 \
    model.data.validation_ds.max_seq_length=8192 \
    model.data.train_ds.add_eos=False \
    model.data.train_ds.add_bos=True \
    model.data.train_ds.truncation_field="output" \
    model.data.test_ds.truncation_field="output" \
    model.data.validation_ds.truncation_field="output" \
    model.data.train_ds.separate_prompt_and_response_with_newline=False \
    model.optim.lr=$LR \
    +model.peft.peft_scheme=null \
    +model.peft.restore_from_path=null \
    ++model.optim.sched.warmup_steps=${WARMUP} \
    ++model.tokenizer.type=/scripts/original_llama3_instruct \
    ++model.tokenizer.model=null \
    ++model.tokenizer.library="huggingface"
