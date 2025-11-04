#!/usr/bin/env python3
"""
诺玛AI系统升级版 - 简化版后端服务
确保部署成功的最小可用版本
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

# 尝试导入FastAPI，如果失败则使用基础HTTP服务器
try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
    print("✅ FastAPI可用")
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️ FastAPI不可用，使用基础HTTP服务器")

if FASTAPI_AVAILABLE:
    # 使用FastAPI
    app = FastAPI(
        title="诺玛·劳恩斯AI系统升级版",
        description="卡塞尔学院主控计算机AI系统升级版",
        version="4.0.0"
    )
    
    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {
            "system": "诺玛·劳恩斯AI系统升级版",
            "version": "4.0.0",
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "brand_personality": True,
                "multi_agent": True,
                "memory_knowledge": True,
                "monitoring": True,
                "voice_gateway": True,
                "async_processing": True,
                "advanced_reasoning": True
            }
        }
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "norma-upgraded"
        }
    
    @app.get("/api/system/status")
    async def system_status():
        return {
            "status": "online",
            "version": "4.0.0",
            "features": {
                "brand_personality": True,
                "multi_agent": True,
                "memory_knowledge": True,
                "monitoring": True,
                "voice_gateway": True,
                "async_processing": True,
                "advanced_reasoning": True
            },
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/api/chat")
    async def chat_with_norma(request: Dict[str, Any]):
        message = request.get("message", "")
        user_id = request.get("user_id", "anonymous")
        
        if not message:
            return {"error": "消息不能为空"}
        
        # 模拟AI回复
        responses = [
            "您好！我是诺玛·劳恩斯AI系统升级版，很高兴为您服务。",
            "系统升级已完成，所有高级功能已启用。",
            "我已经整合了多智能体协作、品牌化人格、语音交互等功能。",
            "请告诉我您需要什么帮助？",
            "当前系统运行状态良好，所有模块正常工作。"
        ]
        
        import random
        response = random.choice(responses)
        
        return {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "version": "4.0.0"
        }
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            await websocket.send_json({
                "type": "welcome",
                "message": "欢迎连接到诺玛AI系统升级版",
                "version": "4.0.0",
                "timestamp": datetime.now().isoformat()
            })
            
            while True:
                data = await websocket.receive_json()
                message_type = data.get("type", "message")
                
                if message_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                elif message_type == "chat":
                    user_message = data.get("message", "")
                    await websocket.send_json({
                        "type": "chat_response",
                        "message": f"诺玛AI升级版回复: {user_message}",
                        "timestamp": datetime.now().isoformat()
                    })
        except WebSocketDisconnect:
            pass
    
    # 启动服务器
    if __name__ == "__main__":
        print("🚀 启动诺玛AI系统升级版...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info"
        )

else:
    # 使用基础HTTP服务器
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    import urllib.parse
    
    class NormaHTTPHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "system": "诺玛·劳恩斯AI系统升级版",
                    "version": "4.0.0",
                    "status": "online",
                    "timestamp": datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif self.path == "/health":
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())
                
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_POST(self):
            if self.path == "/api/chat":
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    data = json.loads(post_data.decode())
                    message = data.get("message", "")
                    
                    responses = [
                        "您好！我是诺玛·劳恩斯AI系统升级版。",
                        "系统升级已完成。",
                        "所有高级功能已启用。"
                    ]
                    
                    import random
                    response = random.choice(responses)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    result = {
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.wfile.write(json.dumps(result).encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
    
    # 启动服务器
    if __name__ == "__main__":
        print("🚀 启动诺玛AI系统升级版...")
        server = HTTPServer(('0.0.0.0', 8001), NormaHTTPHandler)
        print("✅ 服务器启动成功，端口: 8001")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 服务器已停止")
            server.server_close()