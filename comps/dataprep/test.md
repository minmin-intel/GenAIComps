## Start redis db
```bash
docker run -p 6379:6379 -p 8001:8001 -d redis/redis-stack:7.2.0-v9
```

## env setup
```bash
export ip_address=$(hostname -I | awk '{print $1}')
export REDIS_URL="redis://${ip_address}:6379"
```