#!/bin/bash
source ~/.bashrc
OPENMM_CPU_THREADS=1

# settings
epochs=150
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
input_prefix="~/ResFF/data/" # training data path
datasets="spice-train" # training dataset
vl_input_prefix="~/ResFF/data/" # validation data path
vl_datasets="spice-val" # validation dataset
output_prefix="~/ResFF/train" # output path
force_weight=1.0 # force loss weight
residual_weight=0.0 # residual loss weight. After completing the training of the first round for the two modules, keep residual_weight to 1 for fine-tuning.
stage='stage_1' # training stage: stage_1 (only MM) with residual_weight set to 0, stage_2 (only residual) with residual_weight set to 1. 


# run
conda activate resff

#do
python ~/ResFF/train.py --epochs $epochs --batch_size $batch_size --layer_1 $layer_1 --layer_2 $layer_2 --units $units --activation $activation --config_1 "$config_1" --config_2 "$config_2" --janossy_config "$janossy_config" --learning_rate $learning_rate \
--input_prefix $input_prefix --datasets "$datasets" --vl_input_prefix $vl_input_prefix --vl_datasets "$vl_datasets" --output_prefix $output_prefix --n_max_confs $n_max_confs --force_weight $force_weight --residual_weight $residual_weight --stage $stage
#done
