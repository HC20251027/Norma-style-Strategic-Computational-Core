#!/usr/bin/env python3
"""
诺玛·劳恩斯AI系统 - 升级版本主入口
整合所有升级模块和功能的完整版本

升级版本特性:
- 品牌化AI人格系统
- 多智能体协作
- 记忆和知识库管理
- 实时监控和性能分析
- 语音交互网关
- 异步任务处理
- 高级推理引擎

作者: 皇
版本: 4.0.0 Enhanced
创建时间: 2025-10-31
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import psutil

# 添加项目路径
sys.path.append('/workspace')
sys.path.append('/workspace/backend')
sys.path.append('/workspace/norma_agent_enhanced')
sys.path.append('/workspace/code')

# 尝试导入升级模块
try:
    from norma_agent_enhanced import NormaBrandAgent
    from norma_agent_enhanced.core.personality_engine import NormaPersonalityEngine
    from norma_agent_enhanced.core.user_preferences import UserPreferencesManager
    from norma_agent_enhanced.monitoring.monitoring_manager import NormaMonitoringManager
    from norma_agent_enhanced.memory_knowledge.core.memory_manager import NormaMemoryManager
    from norma_agent_enhanced.multi_agent.main_system import NormaMultiAgentSystem
    ENHANCED_MODULES_AVAILABLE = True
    print("✅ 升级模块加载成功")
except ImportError as e:
    print(f"⚠️ 升级模块导入失败: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# 尝试导入基础模块
try:
    from backend.src.agents.norma_agent import NormaAgent
    from backend.src.config.settings import NormaSettings
    BASIC_MODULES_AVAILABLE = True
    print("✅ 基础模块加载成功")
except ImportError as e:
    print(f"⚠️ 基础模块导入失败: {e}")
    BASIC_MODULES_AVAILABLE = False

# =============================================================================
# 配置管理
# =============================================================================

class NormaUpgradedConfig:
    """诺玛升级版本配置"""
    
    # 环境设置
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # 服务器配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8001"))
    
    # API配置
    API_PREFIX = "/api/v1"
    API_TITLE = "诺玛·劳恩斯AI系统升级版"
    API_VERSION = "4.0.0"
    API_DESCRIPTION = "卡塞尔学院主控计算机AI系统 - 集成多智能体、品牌化人格、语音交互等高级功能"
    
    # 数据库配置
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./norma_upgraded.db")
    
    # DeepSeek API配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-c83fe2d46db542c7ac0df03764e35c41")
    DEEPSEEK_API_BASE = "https://api.deepseek.com"
    DEEPSEEK_MODEL = "deepseek-chat"
    
    # WebSocket配置
    WS_MAX_CONNECTIONS = 100
    WS_PING_INTERVAL = 30
    WS_PING_TIMEOUT = 10
    
    # 升级功能开关
    FEATURES = {
        "brand_personality": ENHANCED_MODULES_AVAILABLE,
        "multi_agent": ENHANCED_MODULES_AVAILABLE,
        "memory_knowledge": ENHANCED_MODULES_AVAILABLE,
        "monitoring": ENHANCED_MODULES_AVAILABLE,
        "voice_gateway": ENHANCED_MODULES_AVAILABLE,
        "async_processing": ENHANCED_MODULES_AVAILABLE,
        "advanced_reasoning": ENHANCED_MODULES_AVAILABLE
    }

# =============================================================================
# 全局状态管理
# =============================================================================

class NormaUpgradedState:
    """诺玛升级版本全局状态"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.brand_agent = None
        self.personality_engine = None
        self.memory_manager = None
        self.multi_agent_system = None
        self.monitoring_manager = None
        self.user_preferences = {}
        self.active_connections: List[WebSocket] = []
        self.system_stats = {
            "uptime": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
            "active_connections": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
    
    async def initialize_modules(self):
        """初始化升级模块"""
        if not ENHANCED_MODULES_AVAILABLE:
            print("⚠️ 升级模块不可用，使用基础功能")
            return
        
        try:
            # 初始化品牌智能体
            self.brand_agent = NormaBrandAgent("system")
            print("✅ 品牌智能体初始化成功")
            
            # 初始化人格引擎
            self.personality_engine = NormaPersonalityEngine()
            print("✅ 人格引擎初始化成功")
            
            # 初始化记忆管理器
            self.memory_manager = NormaMemoryManager()
            print("✅ 记忆管理器初始化成功")
            
            # 初始化多智能体系统
            self.multi_agent_system = NormaMultiAgentSystem()
            print("✅ 多智能体系统初始化成功")
            
            # 初始化监控管理器
            self.monitoring_manager = NormaMonitoringManager()
            print("✅ 监控管理器初始化成功")
            
        except Exception as e:
            print(f"❌ 模块初始化失败: {e}")
    
    async def update_system_stats(self):
        """更新系统统计信息"""
        try:
            self.system_stats.update({
                "uptime": (datetime.now() - self.start_time).total_seconds(),
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "active_connections": len(self.active_connections)
            })
        except Exception as e:
            print(f"⚠️ 统计信息更新失败: {e}")

# 全局状态实例
norma_state = NormaUpgradedState()

# =============================================================================
# FastAPI应用初始化
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("🚀 诺玛AI系统升级版启动中...")
    await norma_state.initialize_modules()
    print("✅ 系统初始化完成")
    
    # 启动统计更新任务
    async def stats_updater():
        while True:
            await norma_state.update_system_stats()
            await asyncio.sleep(30)  # 每30秒更新一次
    
    stats_task = asyncio.create_task(stats_updater())
    
    yield
    
    # 关闭时清理
    stats_task.cancel()
    print("🛑 诺玛AI系统升级版已关闭")

# 创建FastAPI应用
app = FastAPI(
    title=NormaUpgradedConfig.API_TITLE,
    description=NormaUpgradedConfig.API_DESCRIPTION,
    version=NormaUpgradedConfig.API_VERSION,
    prefix=NormaUpgradedConfig.API_PREFIX,
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API路由定义
# =============================================================================

@app.get("/")
async def root():
    """根路径 - 系统信息"""
    return {
        "system": "诺玛·劳恩斯AI系统升级版",
        "version": NormaUpgradedConfig.API_VERSION,
        "environment": NormaUpgradedConfig.ENVIRONMENT,
        "features": NormaUpgradedConfig.FEATURES,
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "uptime": norma_state.system_stats.get("uptime", 0)
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_stats": norma_state.system_stats
    }

@app.get("/api/system/status")
async def system_status():
    """系统状态API"""
    return {
        "status": "online",
        "version": NormaUpgradedConfig.API_VERSION,
        "features": NormaUpgradedConfig.FEATURES,
        "enhanced_modules": ENHANCED_MODULES_AVAILABLE,
        "basic_modules": BASIC_MODULES_AVAILABLE,
        "system_stats": norma_state.system_stats,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_with_norma(request: Dict[str, Any]):
    """与诺玛AI对话"""
    try:
        message = request.get("message", "")
        user_id = request.get("user_id", "anonymous")
        
        if not message:
            return {"error": "消息不能为空"}
        
        # 更新请求统计
        norma_state.system_stats["total_requests"] += 1
        
        # 使用品牌智能体处理对话
        if norma_state.brand_agent:
            response = await norma_state.brand_agent.process_brand_interaction(
                message, "conversation"
            )
            content = response.get("content", "抱歉，我现在无法回复。")
        else:
            # 基础回复
            content = f"诺玛AI升级版已收到您的消息: {message}"
        
        # 更新成功统计
        norma_state.system_stats["successful_requests"] += 1
        
        return {
            "response": content,
            "timestamp": datetime.now().isoformat(),
            "version": NormaUpgradedConfig.API_VERSION
        }
        
    except Exception as e:
        # 更新失败统计
        norma_state.system_stats["failed_requests"] += 1
        return {
            "error": f"处理消息时发生错误: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/features")
async def get_features():
    """获取系统功能列表"""
    return {
        "available_features": NormaUpgradedConfig.FEATURES,
        "enhanced_modules": ENHANCED_MODULES_AVAILABLE,
        "basic_modules": BASIC_MODULES_AVAILABLE,
        "feature_details": {
            "brand_personality": "品牌化AI人格系统",
            "multi_agent": "多智能体协作系统",
            "memory_knowledge": "记忆和知识库管理",
            "monitoring": "实时监控和性能分析",
            "voice_gateway": "语音交互网关",
            "async_processing": "异步任务处理",
            "advanced_reasoning": "高级推理引擎"
        }
    }

@app.get("/api/stats")
async def get_system_stats():
    """获取系统统计信息"""
    await norma_state.update_system_stats()
    return norma_state.system_stats

# =============================================================================
# WebSocket连接管理
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    await websocket.accept()
    norma_state.active_connections.append(websocket)
    
    try:
        # 发送欢迎消息
        await websocket.send_json({
            "type": "welcome",
            "message": "欢迎连接到诺玛AI系统升级版",
            "version": NormaUpgradedConfig.API_VERSION,
            "features": NormaUpgradedConfig.FEATURES,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # 接收消息
            data = await websocket.receive_json()
            message_type = data.get("type", "message")
            
            if message_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            
            elif message_type == "chat":
                user_message = data.get("message", "")
                if user_message:
                    # 处理聊天消息
                    response = await process_chat_message(user_message)
                    await websocket.send_json({
                        "type": "chat_response",
                        "message": response,
                        "timestamp": datetime.now().isoformat()
                    })
            
    except WebSocketDisconnect:
        print("客户端断开连接")
    except Exception as e:
        print(f"WebSocket错误: {e}")
    finally:
        if websocket in norma_state.active_connections:
            norma_state.active_connections.remove(websocket)

async def process_chat_message(message: str) -> str:
    """处理聊天消息"""
    try:
        if norma_state.brand_agent:
            response = await norma_state.brand_agent.process_brand_interaction(
                message, "conversation"
            )
            return response.get("content", "抱歉，我现在无法回复。")
        else:
            return f"诺玛AI升级版回复: {message}"
    except Exception as e:
        return f"处理消息时发生错误: {str(e)}"

# =============================================================================
# 静态文件服务
# =============================================================================

# 挂载静态文件目录
if os.path.exists("/workspace/uploads"):
    app.mount("/uploads", StaticFiles(directory="/workspace/uploads"), name="uploads")

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("🚀 诺玛·劳恩斯AI系统升级版启动")
    print("=" * 60)
    print(f"版本: {NormaUpgradedConfig.API_VERSION}")
    print(f"环境: {NormaUpgradedConfig.ENVIRONMENT}")
    print(f"主机: {NormaUpgradedConfig.HOST}:{NormaUpgradedConfig.PORT}")
    print(f"升级模块: {'✅' if ENHANCED_MODULES_AVAILABLE else '❌'}")
    print(f"基础模块: {'✅' if BASIC_MODULES_AVAILABLE else '❌'}")
    print("=" * 60)
    
    # 启动服务器
    uvicorn.run(
        "norma_agent_upgraded_main:app",
        host=NormaUpgradedConfig.HOST,
        port=NormaUpgradedConfig.PORT,
        reload=NormaUpgradedConfig.DEBUG,
        log_level="info"
    )