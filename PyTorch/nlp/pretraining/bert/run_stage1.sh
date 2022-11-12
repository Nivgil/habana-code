#!/bin/bash

PYTHON=python
DATA_DIR=/git_lfs/data/pytorch/bert/pretraining/hdf5_lower_case_1_seq_len_128/books_wiki_en_corpus/train_packed_new
#MAX_STEPS=7038
MAX_STEPS=200
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
export INIT_HCCL_ON_ACQUIRE=true

rm -rf /tmp/log_directory
mkdir /tmp/log_directory


#/usr/local/share/openmpi/bin//mpirun -n 8 --bind-to core --map-by socket:PE=7 --rank-by core --report-bindings --allow-run-as-root \
/usr/local/share/openmpi/bin//mpirun -n 8 --bind-to core --map-by socket:PE=4 --rank-by core --report-bindings --allow-run-as-root \
$PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp --hmp_bf16=./ops_bf16_bert_pt.txt \
      --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json --use_habana \
      --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --json-summary=/tmp/log_directory/dllogger.json \
      --output_dir=/tmp/results/checkpoints --use_fused_lamb \
      --input_dir=$DATA_DIR \
      --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=$MAX_STEPS \
      --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128



# SINGLE CARD

# $PYTHON run_pretraining.py --do_train --bert_model=bert-large-uncased --hmp \
#       --hmp_bf16=./ops_bf16_bert_pt.txt --hmp_fp32=./ops_fp32_bert_pt.txt --config_file=./bert_config.json \
#       --use_habana --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
#       --json-summary=/tmp/log_directory/dllogger.json --output_dir=/tmp/results/checkpoints --use_fused_lamb \
#       --input_dir=$DATA_DIR \
#       --train_batch_size=8192 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 \
#       --warmup_proportion=0.2843 --num_steps_per_checkpoint=200 --learning_rate=0.006 --gradient_accumulation_steps=128 \
#       --enable_packed_data_mode True