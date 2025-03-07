## Start redis db
```bash
docker run --name redis-db -p 6379:6379 -p 8001:8001 -d redis/redis-stack:7.2.0-v9
docker run --name redis-kv -p 6380:6379 -p 8002:8001 -d redis/redis-stack:7.2.0-v9
```

## env setup
```bash
export ip_address=$(hostname -I | awk '{print $1}')
export REDIS_URL_VECTOR="redis://${ip_address}:6379"
export REDIS_URL_KV="redis://${ip_address}:6380"
export LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"
export LLM_ENDPOINT="http://${ip_address}:8086"
export DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_REDIS_FIANANCE"
```

## Build docker image
```bash
cd $WORKDIR/GenAIComps
docker build -t opea/dataprep:mh --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
```

## Launch dataprep container
```bash
docker compose -f compose.yaml up -d
```
-e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT 
