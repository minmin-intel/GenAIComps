strategy=react
llm_endpoint=http://localhost:9009

python test.py \
--endpoint_test \
--quick_test \
--strategy $strategy \
--llm_endpoint_url $llm_endpoint
