N=5
K=1
LR=5e-6
CUDA_VISIBLE_DEVICES=1 python main.py \
    --N_way $N \
    --K_shot $K \
    --Q_num 1 \
    --data_dir data \
    --seed 42 \
    --output_dir output \
    --max_length 128 \
    --model_name_or_path bert-base-uncased \
    --per_device_train_batch_size 4 \
    --num_train_iters 30000 \
    --num_valid_iters 1000 \
    --num_valid_steps 1000 \
    --learning_rate $LR \
    --weight_decay 0.02 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 300 \
    --test_file test_wiki_input_with_rel_id-$N-$K.json \
    --test_file_da test_pubmed_input-$N-$K.json\
    --use_rel \
    --use_cp \
    --do_test \
