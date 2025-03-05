echo "Building the vllm docker image"
cd $WORKDIR
echo $WORKDIR

image_name=vllm-gaudi:mh-1p19

if [ ! -d "./vllm-fork" ]; then
    git clone https://github.com/HabanaAI/vllm-fork.git
fi
cd ./vllm-fork
#git checkout v0.6.4.post2+Gaudi-1.19.2
docker build --no-cache -f Dockerfile.hpu -t  $image_name --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
if [ $? -ne 0 ]; then
    echo "$image_name failed"
    exit 1
else
    echo "$image_name successful"
fi

