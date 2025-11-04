#!/usr/bin/env python3
"""
诺玛AI系统 - 简化版后端服务（用于性能测试）
"""

import os
import sys
import json
import asyncio
import sqlite3
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# 数据模型
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None

class SystemStatusResponse(BaseModel):
    system_name: str
    version: str
    uptime: str
    status: str
    cpu_usage: str
    memory_usage: str
    memory_available: str
    network_status: str
    security_level: str
    timestamp: str

class BloodAnalysisRequest(BaseModel):
    student_name: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    type: Optional[str] = "general"

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.system_status_subscribers: List[WebSocket] = []
        self.chat_subscribers: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        await websocket.accept()
        if connection_type == "system_status":
            self.system_status_subscribers.append(websocket)
        elif connection_type == "chat":
            self.chat_subscribers.append(websocket)
        else:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.system_status_subscribers:
            self.system_status_subscribers.remove(websocket)
        if websocket in self.chat_subscribers:
            self.chat_subscribers.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, connection_type: str = "general"):
        if connection_type == "system_status":
            connections = self.system_status_subscribers
        elif connection_type == "chat":
            connections = self.chat_subscribers
        else:
            connections = self.active_connections
        
        for connection in connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# 诺玛系统服务类
class NormaSystemService:
    def __init__(self):
        self.db_path = "/workspace/data/norma_database.db"
        self.logs_path = "/workspace/data/norma_logs.db"
        self.ensure_directories()

    def ensure_directories(self):
        """确保必要目录存在"""
        directories = [
            "/workspace/data",
            "/workspace/data/knowledge_base",
            "/workspace/data/vector_db",
            "/workspace/data/dragon_blood",
            "/workspace/data/security_logs",
            "/workspace/backend/uploads"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def get_system_status(self) -> SystemStatusResponse:
        """获取系统状态"""
        try:
            # 获取真实的系统信息
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemStatusResponse(
                system_name="诺玛·劳恩斯",
                version="3.1.2 (1990-2025)",
                uptime="35年3个月",
                status="正常运行",
                cpu_usage=f"{cpu_percent:.1f}%",
                memory_usage=f"{memory.percent:.1f}%",
                memory_available=f"{memory.available / 1024**3:.1f}GB",
                network_status="活跃",
                security_level="高",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"系统状态获取失败: {str(e)}")

    async def process_chat_message(self, message: str) -> str:
        """处理聊天消息"""
        try:
            # 模拟处理时间
            await asyncio.sleep(0.1)
            return f"诺玛系统回复: 您发送的消息是 '{message}'。系统运行正常，当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            return f"处理消息时发生错误: {str(e)}"

    async def get_blood_analysis(self, student_name: Optional[str] = None) -> Dict[str, Any]:
        """获取血统分析"""
        try:
            await asyncio.sleep(0.2)  # 模拟分析时间
            if student_name:
                return {
                    "student_name": student_name,
                    "bloodline_type": "A级混血种",
                    "purity_level": "87.3%",
                    "abilities": "黄金瞳、言灵·君焰",
                    "status": "稳定",
                    "last_analysis": datetime.now().strftime("%Y-%m-%d")
                }
            else:
                return {
                    "total_registered_students": 4,
                    "bloodline_distribution": {
                        "S级混血种": 1,
                        "A级混血种": 3
                    },
                    "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"血统分析失败: {str(e)}")

    async def get_system_logs(self, limit: int = 10) -> Dict[str, Any]:
        """获取系统日志"""
        try:
            await asyncio.sleep(0.1)
            logs = []
            for i in range(limit):
                logs.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": ["INFO", "WARNING", "ERROR"][i % 3],
                    "module": ["系统", "网络", "安全"][i % 3],
                    "message": f"示例日志消息 {i+1}",
                    "details": f"详细描述 {i+1}"
                })
            return {
                "log_count": len(logs),
                "logs": logs,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"系统日志获取失败: {str(e)}")

    async def search_knowledge_base(self, query: str) -> Dict[str, Any]:
        """搜索知识库"""
        try:
            await asyncio.sleep(0.3)  # 模拟搜索时间
            return {
                "search_query": query,
                "search_engine": "DuckDuckGo (模拟)",
                "results": [
                    {
                        "title": f"关于 '{query}' 的研究资料",
                        "url": "https://example.com/research",
                        "snippet": "这是关于该主题的详细研究内容...",
                        "reliability": "高"
                    }
                ],
                "total_results": 1,
                "search_time": "0.3秒",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"知识库搜索失败: {str(e)}")

    async def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        try:
            await asyncio.sleep(0.1)
            return {
                "firewall_status": "已启用",
                "antivirus_status": "运行中",
                "intrusion_detection": "正常监控",
                "failed_login_attempts": 0,
                "suspicious_activities": "无",
                "last_security_scan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "security_score": 95,
                "recommendations": [
                    "系统安全状态良好",
                    "建议继续保持当前安全配置"
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"安全状态获取失败: {str(e)}")

    async def get_multi_agent_status(self) -> Dict[str, Any]:
        """获取多智能体状态"""
        try:
            await asyncio.sleep(0.2)
            return {
                "agents": [
                    {
                        "name": "搜索专家",
                        "status": "活跃",
                        "tasks_completed": 156,
                        "accuracy": 94.5,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    {
                        "name": "文档专家",
                        "status": "活跃",
                        "tasks_completed": 89,
                        "accuracy": 97.2,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    {
                        "name": "知识专家",
                        "status": "活跃",
                        "tasks_completed": 203,
                        "accuracy": 96.8,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ],
                "collaboration_mode": "协调模式",
                "total_tasks": 448,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"多智能体状态获取失败: {str(e)}")

# 全局服务实例
norma_service = NormaSystemService()

# 创建FastAPI应用
app = FastAPI(
    title="诺玛·劳恩斯AI系统API",
    description="卡塞尔学院主控计算机AI系统后端服务",
    version="3.1.2"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API路由
@app.get("/")
async def root():
    return {
        "message": "诺玛·劳恩斯AI系统API服务",
        "version": "3.1.2",
        "status": "running",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/api/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统状态"""
    return await norma_service.get_system_status()

@app.post("/api/chat/message")
async def chat_message(message: ChatMessage):
    """处理聊天消息"""
    response = await norma_service.process_chat_message(message.message)
    return {
        "response": response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/api/bloodline/analyze")
async def analyze_bloodline(request_body: BloodAnalysisRequest):
    """血统分析"""
    return await norma_service.get_blood_analysis(request_body.student_name)

@app.get("/api/system/logs")
async def get_logs(limit: int = 10):
    """获取系统日志"""
    return await norma_service.get_system_logs(limit)

@app.post("/api/knowledge/search")
async def search_knowledge(request_body: SearchRequest):
    """搜索知识库"""
    return await norma_service.search_knowledge_base(request_body.query)

@app.get("/api/security/status")
async def get_security_status():
    """获取安全状态"""
    return await norma_service.get_security_status()

@app.get("/api/agents/status")
async def get_agents_status():
    """获取多智能体状态"""
    return await norma_service.get_multi_agent_status()

# WebSocket端点
@app.websocket("/ws/system")
async def websocket_system_status(websocket: WebSocket):
    """系统状态WebSocket连接"""
    await manager.connect(websocket, "system_status")
    try:
        while True:
            status = await norma_service.get_system_status()
            await manager.send_personal_message(
                json.dumps({
                    "type": "system_status",
                    "data": status.dict(),
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
            await asyncio.sleep(5)  # 每5秒更新一次
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """聊天WebSocket连接"""
    await manager.connect(websocket, "chat")
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "message":
                response = await norma_service.process_chat_message(message_data.get("message", ""))
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "response",
                        "message": response,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
                await manager.broadcast(
                    json.dumps({
                        "type": "chat_message",
                        "user": "用户",
                        "message": message_data.get("message", ""),
                        "timestamp": datetime.now().isoformat()
                    }),
                    "chat"
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """日志流WebSocket连接"""
    await manager.connect(websocket)
    try:
        while True:
            logs = await norma_service.get_system_logs(5)
            await manager.send_personal_message(
                json.dumps({
                    "type": "logs_update",
                    "data": logs,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
            await asyncio.sleep(3)  # 每3秒更新一次
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 文件上传端点
@app.post("/api/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """上传文档"""
    try:
        file_path = f"/workspace/backend/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "message": "文件上传成功",
            "filename": file.filename,
            "size": len(content),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 静态文件服务
if Path("/workspace/backend/uploads").exists():
    app.mount("/uploads", StaticFiles(directory="/workspace/backend/uploads"), name="uploads")

if __name__ == "__main__":
    uvicorn.run(
        "simple_backend:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )