#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#set -xe

# this script should be run from tests directory
#  bash agent/sql_agent_test/test_sql_agent.sh

WORKPATH=$(dirname "$PWD")
echo $WORKPATH
LOG_PATH="$WORKPATH/tests"

# WORKDIR is one level up from GenAIComps
WORKDIR=$(dirname "$WORKPATH")
echo $WORKDIR

export agent_image="opea/agent-langchain:comps"
export agent_container_name="test-comps-agent-endpoint"

export ip_address=$(hostname -I | awk '{print $1}')
tgi_port=8085
tgi_volume=${HF_CACHE_DIR} #$WORKPATH/data
vllm_port=8085
vllm_volume=${HF_CACHE_DIR} #$WORKPATH/data
export model=meta-llama/Meta-Llama-3.1-70B-Instruct
export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID="meta-llama/Meta-Llama-3.1-70B-Instruct"
export LLM_ENDPOINT_URL="http://${ip_address}:${tgi_port}"
export temperature=0.01
export max_new_tokens=4096
export TOOLSET_PATH=$WORKPATH/comps/agent/langchain/tools/
echo "TOOLSET_PATH=${TOOLSET_PATH}"
export recursion_limit=15
export db_name=california_schools
export db_path=/home/user/TAG-Bench/dev_folder/dev_databases/${db_name}/${db_name}.sqlite

# download the test data
function prepare_data() {
    cd $WORKDIR

    echo "Downloading data..."
    git clone https://github.com/TAG-Research/TAG-Bench.git
    cd TAG-Bench/setup
    chmod +x get_dbs.sh
    ./get_dbs.sh

    echo "Split data..."
    cd $WORKPATH/tests/agent/sql_agent_test
    bash run_data_split.sh

    echo "Data preparation done!"
}

function build_docker_images() {
    echo "Building the docker images"
    cd $WORKPATH
    echo $WORKPATH
    docker build --no-cache -t $agent_image --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -f comps/agent/langchain/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/agent-langchain built fail"
        exit 1
    else
        echo "opea/agent-langchain built successful"
    fi
}

# launch tgi-gaudi
function start_tgi_service() {
    echo "token is ${HF_TOKEN}"

    #multi cards
    echo "start tgi gaudi service"
    docker run -d --runtime=habana --name "test-comps-tgi-gaudi-service" -p $tgi_port:80 -v $tgi_volume:/data -e HF_TOKEN=$HF_TOKEN -e HABANA_VISIBLE_DEVICES=0,1,2,3 -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e http_proxy=$http_proxy -e https_proxy=$https_proxy --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.5 --model-id $model --max-input-tokens 8192 --max-total-tokens 16384 --sharded true --num-shard 4
    sleep 5s
    echo "Waiting tgi gaudi ready"
    n=0
    until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
        docker logs test-comps-tgi-gaudi-service &> ${LOG_PATH}/tgi-gaudi-service.log
        n=$((n+1))
        if grep -q Connected ${LOG_PATH}/tgi-gaudi-service.log; then
            break
        fi
        sleep 5s
    done
    sleep 5s
    echo "Service started successfully"
}

function build_vllm_docker_images() {
    echo "Building the vllm docker images"
    cd $WORKPATH
    echo $WORKPATH
    if [ ! -d "./vllm" ]; then
        git clone https://github.com/HabanaAI/vllm-fork.git
    fi
    cd ./vllm-fork
    docker build -f Dockerfile.hpu -t opea/vllm-gaudi:comps --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
    if [ $? -ne 0 ]; then
        echo "opea/vllm-gaudi:comps failed"
        exit 1
    else
        echo "opea/vllm-gaudi:comps successful"
    fi
}

function start_vllm_service() {
    # redis endpoint
    echo "token is ${HF_TOKEN}"

    #single card
    echo "start vllm gaudi service"
    echo "**************model is $model**************"
    docker run -d --runtime=habana --rm --name "test-comps-vllm-gaudi-service" -e HABANA_VISIBLE_DEVICES=0,1,2,3 -p $vllm_port:80 -v $vllm_volume:/data -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e VLLM_SKIP_WARMUP=true --cap-add=sys_nice --ipc=host opea/vllm-gaudi:comps --model ${model} --host 0.0.0.0 --port 80 --block-size 128 --max-num-seqs  4096 --max-seq_len-to-capture 8192 --tensor-parallel-size 4
    sleep 5s
    echo "Waiting vllm gaudi ready"
    n=0
    until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
        docker logs test-comps-vllm-gaudi-service &> ${LOG_PATH}/vllm-gaudi-service.log
        n=$((n+1))
        if grep -q "Uvicorn running on" ${LOG_PATH}/vllm-gaudi-service.log; then
            break
        fi
        if grep -q "No such container" ${LOG_PATH}/vllm-gaudi-service.log; then
            echo "container test-comps-vllm-gaudi-service not found"
            exit 1
        fi
        sleep 5s
    done
    sleep 5s
    echo "Service started successfully"
}
# launch the agent
function start_sql_agent_llama_service() {
    echo "Starting sql_agent_llama agent microservice"
    docker compose -f $WORKPATH/tests/agent/sql_agent_llama.yaml up -d
    sleep 5s
    docker logs test-comps-agent-endpoint
    echo "Service started successfully"
}

# run the test
function run_test() {
    echo "Running test..."
    cd $WORKPATH/tests/agent/
    python3 test.py --test-sql-agent
}

# echo "Building docker image...."
# build_docker_images

# echo "Lauching TGI-gaudi...."
# start_tgi_service

echo "Building vllm docker image...."
build_vllm_docker_images

echo "Launching vllm service...."
start_vllm_service

echo "launching sql_agent_llama service...."
start_sql_agent_llama_service

echo "Running test...."
run_test

