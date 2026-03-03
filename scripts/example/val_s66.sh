:set ff=unix#!/bin/bash
source ~/.bashrc
OPENMM_CPU_THREADS=1
# settings
epochs=1
batch_size=1 # number of molecules
n_max_confs=64 # number of conformations, actual batch size = number of conformations × number of molecules
layer_1="SAGEConv" # GNN architecture for MM module
layer_2="TorchMD_ET" # GNN architecture for residual module
units=512 # hidden dimension
activation="relu" # activation function
config_1="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1" # MM module architecture
config_2="torch" # residual module architecture
janossy_config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1" # Janossy pooling architecture
learning_rate=1e-4 # learning rate
input_prefix="./data/" # validation data path
datasets="s66x8" # validation dataset
output_prefix="./results/" # output path
force_weight=1.0 # force loss weight
residual_weight=1.0 # residual loss weight. After completing the training of the first round for the two modules, keep residual_weight to 1 for fine-tuning.
stage="stage_2" # training stage: stage_1 (only MM) with residual_weight set to 0, stage_2 (only residual) with residual_weight set to 1. 
# run
for eval_dataset in $input_prefix/s66x8/*; do
  if [ -d "$eval_dataset" ]; then
    eval_dataset_name=$(basename "$eval_dataset")
    echo "Processing dataset: $eval_dataset_name"
    datasets="s66x8/$eval_dataset_name"
    python ./val_s66.py --epochs $epochs --batch_size $batch_size --layer_1 $layer_1 --layer_2 $layer_2 --units $units --activation $activation --config_1 "$config_1" --config_2 "$config_2" --janossy_config "$janossy_config" --learning_rate $learning_rate \
    --input_prefix $input_prefix --datasets $datasets --eval_dataset_name "$eval_dataset_name" --output_prefix $output_prefix --n_max_confs $n_max_confs --force_weight $force_weight
  fi
done