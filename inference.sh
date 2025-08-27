
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --adapter your_path \
    --device_map mapmoe.json \
    --system fakesv.txt \
    --max_batch_size 4 \
    --torch_dtype bfloat16 \
    --infer_backend pt \
    --max_length 8192 \
    --attn_impl flash_attn \
    --split_dataset_ratio 1.0 \
    --max_new_tokens 1 \
    --metric acc \
    --dataset your_path.jsonl \
    --data_seed 2025 \
    --seed 2025 \
    --temperature 0 \
    --result_path result.jsonl \
