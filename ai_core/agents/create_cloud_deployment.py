#!/usr/bin/env python3
"""
云部署解决方案
创建一个完整的云部署方案，让诺玛AI系统可以被外部访问
"""

import os
import json
import shutil
from pathlib import Path

def create_cloud_deployment():
    """创建云部署方案"""
    
    # 创建部署目录
    deployment_dir = Path("/workspace/cloud_deployment")
    deployment_dir.mkdir(exist_ok=True)
    
    # 创建Docker配置
    dockerfile_content = """
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 复制应用文件
COPY backend/ ./backend/
COPY code/ ./code/
COPY proxy_server.py ./
COPY websocket_proxy.py ./

# 安装Python依赖
RUN pip install fastapi uvicorn websockets aiohttp python-multipart

# 暴露端口
EXPOSE 8001 9000 9001

# 启动命令
CMD ["python", "-c", "
import subprocess
import threading
import time

def start_backend():
    subprocess.run(['python', '-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8001'])

def start_proxy():
    subprocess.run(['python', 'proxy_server.py'])

def start_websocket_proxy():
    subprocess.run(['python', 'websocket_proxy.py'])

# 启动所有服务
backend_thread = threading.Thread(target=start_backend)
proxy_thread = threading.Thread(target=start_proxy)
websocket_thread = threading.Thread(target=start_websocket_proxy)

backend_thread.start()
time.sleep(2)
proxy_thread.start()
websocket_thread.start()

backend_thread.join()
"]
"""
    
    # 创建docker-compose配置
    compose_content = """
version: '3.8'

services:
  norma-ai-backend:
    build: .
    ports:
      - "8001:8001"
      - "9000:9000"
      - "9001:9001"
    environment:
      - DEEPSEEK_API_KEY=sk-c83fe2d46db542c7ac0df03764e35c41
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - norma-ai-backend
    restart: unless-stopped
"""
    
    # 创建Nginx配置
    nginx_content = """
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server norma-ai-backend:8001;
    }
    
    upstream proxy {
        server norma-ai-backend:9000;
    }
    
    upstream websocket {
        server norma-ai-backend:9001;
    }

    server {
        listen 80;
        server_name _;

        # HTTP API代理
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket代理
        location /ws/ {
            proxy_pass http://websocket;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # 静态文件
        location / {
            proxy_pass http://proxy;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
    
    # 创建部署脚本
    deploy_script = """#!/bin/bash

echo "=== 诺玛AI系统云部署 ==="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装"
    exit 1
fi

# 检查docker-compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: docker-compose未安装"
    exit 1
fi

# 构建和启动服务
echo "构建Docker镜像..."
docker-compose build

echo "启动服务..."
docker-compose up -d

echo "等待服务启动..."
sleep 10

# 检查服务状态
echo "检查服务状态..."
curl -f http://localhost/api/system/status && echo "✓ 后端服务正常" || echo "✗ 后端服务异常"

echo "部署完成!"
echo "前端地址: http://localhost"
echo "API地址: http://localhost/api"
echo "WebSocket地址: ws://localhost/ws/chat"
"""
    
    # 写入文件
    with open(deployment_dir / "Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    with open(deployment_dir / "docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    with open(deployment_dir / "nginx.conf", "w") as f:
        f.write(nginx_content)
    
    with open(deployment_dir / "deploy.sh", "w") as f:
        f.write(deploy_script)
    
    os.chmod(deployment_dir / "deploy.sh", 0o755)
    
    print(f"云部署方案已创建在: {deployment_dir}")
    print("使用方法:")
    print("1. 在有Docker的环境中运行: cd cloud_deployment && ./deploy.sh")
    print("2. 或者手动运行: docker-compose up -d")
    
    return deployment_dir

def create_simple_cloud_deployment():
    """创建简单的云部署方案"""
    
    # 创建一个简单的部署包
    deployment_files = {
        "main.py": '''#!/usr/bin/env python3
import uvicorn
import subprocess
import threading
import time
import os

def start_backend():
    os.chdir("/workspace")
    subprocess.run(["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"])

def start_proxy():
    os.chdir("/workspace")
    subprocess.run(["python", "proxy_server.py"])

def start_websocket_proxy():
    os.chdir("/workspace")
    subprocess.run(["python", "websocket_proxy.py"])

if __name__ == "__main__":
    # 启动所有服务
    backend_thread = threading.Thread(target=start_backend)
    proxy_thread = threading.Thread(target=start_proxy)
    websocket_thread = threading.Thread(target=start_websocket_proxy)

    backend_thread.start()
    time.sleep(3)
    proxy_thread.start()
    websocket_thread.start()

    # 启动主服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
''',
        "requirements.txt": '''fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
aiohttp==3.9.1
python-multipart==0.0.6
''',
        "README.md": '''# 诺玛AI系统云部署

## 快速部署

1. 安装依赖: `pip install -r requirements.txt`
2. 运行服务: `python main.py`
3. 访问地址: http://localhost:8080

## 服务端口
- 主服务: 8080
- 后端API: 8001
- HTTP代理: 9000
- WebSocket代理: 9001

## 配置
- DeepSeek API密钥已配置
- 所有服务自动启动
- 支持CORS跨域访问
'''
    }
    
    # 创建部署目录
    simple_dir = Path("/workspace/simple_cloud_deployment")
    simple_dir.mkdir(exist_ok=True)
    
    for filename, content in deployment_files.items():
        with open(simple_dir / filename, "w") as f:
            f.write(content)
    
    print(f"简单云部署方案已创建在: {simple_dir}")
    print("使用方法:")
    print("1. cd simple_cloud_deployment")
    print("2. pip install -r requirements.txt")
    print("3. python main.py")
    
    return simple_dir

if __name__ == "__main__":
    print("=== 创建云部署方案 ===")
    
    # 创建Docker部署方案
    docker_dir = create_cloud_deployment()
    
    # 创建简单部署方案
    simple_dir = create_simple_cloud_deployment()
    
    print("\n=== 部署方案创建完成 ===")
    print(f"Docker部署: {docker_dir}")
    print(f"简单部署: {simple_dir}")
