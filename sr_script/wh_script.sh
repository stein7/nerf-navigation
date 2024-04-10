#!/bin/bash
set_data_path() {
    # Dataset Path
    if [ "$USER_DATA" = "" ]
    then
        export USER_DATA="$USER_HOME/dataset/data"
    fi
    # Declare an associative array for datasets and their paths
    declare -gA datasets
    datasets["chair"]="$USER_DATA/nerf_synthetic/chair"
    datasets["drums"]="$USER_DATA/nerf_synthetic/drums"
    datasets["ficus"]="$USER_DATA/nerf_synthetic/ficus"
    datasets["hotdog"]="$USER_DATA/nerf_synthetic/hotdog"
    datasets["lego"]="$USER_DATA/nerf_synthetic/lego"
    datasets["materials"]="$USER_DATA/nerf_synthetic/materials"
    datasets["mic"]="$USER_DATA/nerf_synthetic/mic"
    datasets["ship"]="$USER_DATA/nerf_synthetic/ship"
    echo "${!datasets[@]}" # print all index of dataset array
}
run_model () {
    local data="$1"
    local gpu_id="$2"
    local mesh="$3"
    data_path=${datasets[$data]}
    workspace="wh_work/$data"
    # echo for debugging
    work="$workspace/mesh_$mesh/"
    echo "CUDA: $gpu_id DATA: $data_path CONFIG: lib/configs/mesh_sampling.yaml mcube_reso: $mesh"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py $data_path --workspace $work --config lib/configs/mesh_sampling.yaml\
                         --mcubes_reso $mesh
}
run_model_parallel () {
    local data="$1"
    local gpu_id="$2"
    local num_gpu="$3"
    # parallel: rays
    #for ray in ${array_rays[*]} ; do
    #    run_model $gpu_id $data $ray &
    #    ((gpu_id++))
    #done
    # parallel: mesh
    avail_gpu=$num_gpu
    current_gpu=$gpu_id
    for mesh in ${array_mesh[*]} ; do
        if [ "$avail_gpu" == 1 ]
        then
            run_model $data $current_gpu $mesh
            avail_gpu=$num_gpu
            current_gpu=$gpu_id
        else
            run_model $data $current_gpu $mesh &
            ((avail_gpu--))
            ((current_gpu++))
        fi
    done
}
# Set Data Path
set_data_path
# Iterate over each GPU and dataset
data="$1"
gpu_id="$2"
num_gpu="$3"
#array_emb_q_ff=("fp32" "fp16" "fp8" "fp4")
#array_emb_q_bp=("fp32" "fp16" "fp8")
array_rays=("1024" "2048" "4096")
array_iters=("7200")
array_mesh=("256" "128" "64")
#run_model_parallel $gpu_id $data
run_model_parallel $data $gpu_id $num_gpu
#run_model $gpu_id $data
