model=mistralai/Mistral-7B-Instruct-v0.3 #meta-llama/Meta-Llama-3-8B-Instruct #
export HF_TOKEN=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
export local_model_dir=/localdisk/minminho/hf_cache/
port=9009

docker run --rm -p $port:80 \
-v ${local_model_dir}:/data \
-e http_proxy=$http_proxy \
-e https_proxy=$https_proxy \
-e no_proxy=$no_proxy \
-e HF_TOKEN=$HF_TOKEN \
--name "tgi-service" \
--ipc=host ghcr.io/huggingface/text-generation-inference:1.4 \
--model-id $model \
