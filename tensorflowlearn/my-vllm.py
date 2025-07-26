from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # 本地服务无需认证
    base_url="http://localhost:8888/v1"  # 端口与启动命令一致
)

response = client.chat.completions.create(
    model="qwen3-1.7b",  # 必须与 --served-model-name 一致
    messages=[
        {"role": "user", "content": "解释量子纠缠的原理"}
    ],
    max_tokens=500,
    temperature=0.7,
    top_p=0.9
)
print(response.choices[0].message.content)

# CUDA_VISIBLE_DEVICES=0 vllm serve ./qwen3-1.7b/ \
#   --port 8888 \
#   --host 0.0.0.0 \
#   --gpu-memory-utilization 0.4 \
#   --max-model-len 4096 \
#   --served-model-name qwen1.7b