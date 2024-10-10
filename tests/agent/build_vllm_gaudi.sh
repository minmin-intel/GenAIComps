git clone https://github.com/HabanaAI/vllm-fork.git
cd ./vllm-fork; git checkout v0.5.3.post1-Gaudi-1.17.0
echo $PWD
cp ../Dockerfile.hpu ./
docker build -f Dockerfile.hpu -t opea/vllm:hpu --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy