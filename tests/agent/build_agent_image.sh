
export agent_image="opea/agent:comps"

WORKPATH=$(dirname "$PWD")
echo $WORKPATH

function build_docker_images() {
    echo "Building the docker images"
    cd $WORKPATH
    echo $WORKPATH
    docker build --no-cache -t $agent_image --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -f comps/agent/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/agent built fail"
        exit 1
    else
        echo "opea/agent built successful"
    fi
}

build_docker_images