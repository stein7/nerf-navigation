#!/bin/bash

set_data_path() {
    # Dataset Path
    export USER_DATA="/home/sslunder0/project/24vlsi/navi_datasets" 
    
    # Declare an associative array for datasets and their paths
    declare -gA datasets
    datasets["stonehenge"]="$USER_DATA/stonehenge"
    datasets["scannet_0000"]="$USER_DATA/scannet_0000"
    datasets["replica_room2"]="$USER_DATA/replica_room2"
    datasets["replica_office3"]="$USER_DATA/replica_office3"
    datasets["replica_hotel"]="$USER_DATA/replica_hotel"
    datasets["replica_FRL_v2"]="$USER_DATA/replica_FRL_v2"
    datasets["replica_apt1"]="$USER_DATA/replica_apt1"
    datasets["replica_apt2"]="$USER_DATA/replica_apt2"

    echo "dataset list : ${!datasets[@]}" # print all index of dataset array
}
set_model_path() {
    # NeRF model Path
    export USER_DATA="/home/sslunder0/project/24vlsi/navi_models" 
    
    # Declare an associative array for models and their paths
    declare -gA models
    models["stone_nerf"]="$USER_DATA/stone_nerf"
    models["scannet_0000"]="$USER_DATA/scannet_0000"
    models["replica_room2"]="$USER_DATA/replica_room2"
    models["replica_office3"]="$USER_DATA/replica_office3"
    models["replica_hotel"]="$USER_DATA/replica_hotel"
    models["replica_FRL_v2"]="$USER_DATA/replica_FRL_v2"
    models["replica_apt1"]="$USER_DATA/replica_apt1"
    models["replica_apt2"]="$USER_DATA/replica_apt2"

    echo "model list : ${!models[@]}" # print all index of dataset array
}
run_model_parallel () {
    local gpu_id="$1"
    local data="$2"
    local model="$3"

    #for scaling in ${array_scaling[*]} ; do
    #    run_model $gpu_id $data $model $scaling
    #done

    len=${#array_scaling[@]}
    for ((i=0; i<$len; i++)); do
        scaling="${array_scaling[$i]}"
        start_pos="${array_start_pos[$i]}"
        end_pos="${array_end_pos[$i]}"
        run_model "$gpu_id" "$data" "$model" "$scaling" "$start_pos" "$end_pos"
    done
}
run_model () {
    local gpu_id="$1"
    local data="$2"
    local model="$3"
    local scaling="$4"
    local start_pos="$5"
    local end_pos="$6"
    data_path=${datasets[$data]}
    workspace=${models[$model]}

    echo "CUDA: $gpu_id DATA: $data_path MODEL: $workspace CONFIG: lib/config.yaml "
    echo "scling: $scaling start: $start_pos end: $end_pos"
    CUDA_VISIBLE_DEVICES=$gpu_id python simulate.py $data_path --workspace $workspace --config lib/config.yaml\
                                     --start_pos $start_pos --end_pos $end_pos --a_star_approx_scaling $scaling
}

# Set Data Path
set_data_path
set_model_path


# Iterate over each GPU and dataset
gpu_id="$1"
data="$2"
model="$3"


echo "input gpu id: ${gpu_id}"
echo "input data: ${data}"
echo "input work model: ${model}"

#echo "Continue?"
#read YES

run_model "$gpu_id" "$data" "$model" "0.01" "0.2 0.5 0.2" "-0.6 -0.7 0.2"

# array_scaling=("0.01" "0.001" "0.0001")
# array_start_pos=("-0.72 0.2 0.2" "-0.72 0.2 0.3" "-0.72 0.2 0.5")
# array_end_pos=("-0.72 0.2 0.2" "-0.72 0.2 0.3" "-0.72 0.2 0.5")
# echo "array scaling ${array_scaling[@]}"
# echo "array_start_pos ${array_start_pos[@]}"
# echo "array_end_pos ${array_end_pos[@]}"
# run_model_parallel $gpu_id $data $model 


