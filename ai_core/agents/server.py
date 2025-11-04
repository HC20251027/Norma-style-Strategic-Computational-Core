#!/usr/bin/env python3
"""
诺玛Agent - 智能多模态AI助手系统
Web API服务器主入口

作者: 诺玛开发团队
版本: 2.0.0
日期: 2025-11-01
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    from dotenv import load_dotenv
except ImportError as e:
    print(f"缺少必要的依赖: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/norma_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="诺玛Agent API",
    description="智能多模态AI助手系统API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
norma_agent = None

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("🚀 诺玛Agent启动中...")
    
    # 创建必要的目录
    directories = ["data", "logs", "data/memory", "data/test_results"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    try:
        # 初始化诺玛Agent核心组件
        from src.core.norma_agent import NormaAgent
        global norma_agent
        norma_agent = NormaAgent()
        await norma_agent.initialize()
        
        logger.info("✅ 诺玛Agent初始化完成")
    except Exception as e:
        logger.error(f"❌ 诺玛Agent初始化失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("🛑 诺玛Agent关闭中...")
    if norma_agent:
        await norma_agent.cleanup()
    logger.info("✅ 诺玛Agent已关闭")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "欢迎使用诺玛Agent智能多模态AI助手系统",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/api/status")
async def api_status():
    """API状态检查"""
    return {
        "api": "诺玛Agent API",
        "status": "running",
        "version": "2.0.0",
        "features": {
            "multimodal": os.getenv("FEATURE_MULTIMODAL", "true") == "true",
            "voice": os.getenv("FEATURE_VOICE", "true") == "true",
            "image": os.getenv("FEATURE_IMAGE", "true") == "true",
            "video": os.getenv("FEATURE_VIDEO", "true") == "true",
            "knowledge": os.getenv("FEATURE_KNOWLEDGE", "true") == "true",
            "memory": os.getenv("FEATURE_MEMORY", "true") == "true",
            "monitoring": os.getenv("FEATURE_MONITORING", "true") == "true"
        }
    }

@app.post("/api/chat")
async def chat_endpoint(request: dict):
    """聊天接口"""
    if not norma_agent:
        raise HTTPException(status_code=503, detail="诺玛Agent未初始化")
    
    try:
        message = request.get("message", "")
        user_id = request.get("user_id", "anonymous")
        session_id = request.get("session_id", "default")
        
        if not message:
            raise HTTPException(status_code=400, detail="消息不能为空")
        
        # 调用诺玛Agent处理消息
        response = await norma_agent.process_message(
            message=message,
            user_id=user_id,
            session_id=session_id
        )
        
        return {
            "status": "success",
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"聊天处理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/multimodal/process")
async def multimodal_endpoint(request: dict):
    """多模态处理接口"""
    if not norma_agent:
        raise HTTPException(status_code=503, detail="诺玛Agent未初始化")
    
    try:
        data_type = request.get("type")
        data = request.get("data")
        options = request.get("options", {})
        
        if not data_type or not data:
            raise HTTPException(status_code=400, detail="数据类型和数据不能为空")
        
        # 调用诺玛Agent处理多模态数据
        result = await norma_agent.process_multimodal(
            data_type=data_type,
            data=data,
            options=options
        )
        
        return {
            "status": "success",
            "result": result,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"多模态处理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def metrics_endpoint():
    """性能指标接口"""
    try:
        import psutil
        import time
        
        return {
            "status": "success",
            "metrics": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "uptime": time.time(),
                "process_count": len(psutil.pids())
            },
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"获取指标错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 获取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # 启动服务器
    logger.info(f"🌟 启动诺玛Agent服务器: http://{host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )