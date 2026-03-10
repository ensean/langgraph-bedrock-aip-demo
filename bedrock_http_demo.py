"""
AWS Bedrock Inference Profile — 纯 HTTP 调用 Demo

使用 requests 库直接调用 Bedrock converse API，
通过 API Key (Bearer token) 认证，无需 AWS SDK 签名。

用法：
    export BEDROCK_API_KEY="your-bedrock-api-key"
    python bedrock_http_demo.py
"""

import json
import os
import sys

import requests

# ── 配置 ──────────────────────────────────────────────────────────────────────

AIP_ARN = "arn:aws:bedrock:ap-northeast-1:123456789012:application-inference-profile/qs2x0oirm4py"
REGION = "ap-northeast-1"

# Bedrock Converse API 端点
ENDPOINT = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{requests.utils.quote(AIP_ARN, safe='')}/converse"


def call_bedrock(prompt: str, api_key: str) -> str:
    """通过 HTTP 直接调用 Bedrock converse API，返回模型回复文本。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ],
        "inferenceConfig": {
            "maxTokens": 512,
            "temperature": 0.3,
        },
    }

    response = requests.post(ENDPOINT, headers=headers, json=body, timeout=60)
    response.raise_for_status()

    data = response.json()
    # converse 响应结构: output.message.content[0].text
    return data["output"]["message"]["content"][0]["text"]


def main():
    api_key = os.environ.get("BEDROCK_API_KEY", "").strip()
    if not api_key:
        print("错误：未设置环境变量 BEDROCK_API_KEY")
        print('  export BEDROCK_API_KEY="your-bedrock-api-key"')
        sys.exit(1)

    prompts = [
        "用一句话介绍 AWS Bedrock 是什么。",
        "计算 123 × 456 等于多少？",
        "写一首关于云计算的四行诗。",
    ]

    print("=" * 60)
    print("  AWS Bedrock HTTP Demo (requests + API Key)")
    print("=" * 60)
    print(f"  Endpoint : {ENDPOINT}")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] 问：{prompt}")
        print("-" * 60)
        try:
            answer = call_bedrock(prompt, api_key)
            print(f"    答：{answer}")
        except requests.HTTPError as e:
            print(f"    HTTP 错误 {e.response.status_code}: {e.response.text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
