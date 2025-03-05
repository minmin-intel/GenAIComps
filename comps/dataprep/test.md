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
```