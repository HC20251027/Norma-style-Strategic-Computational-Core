#!/usr/bin/env python3
"""
诺玛·劳恩斯AI系统 - 生产环境专用启动脚本
专门为生产环境优化，包含额外的安全、监控和性能特性

作者: 皇
创建时间: 2025-10-31
生产环境版本: 3.1.2-agui-production
"""

import os
import sys
import json
import asyncio
import signal
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import psutil
import sqlite3
from functools import wraps

# 添加代码路径
sys.path.append('/workspace/code')
sys.path.append('/workspace')

# =============================================================================
# 生产环境配置
# =============================================================================

class ProductionConfig:
    """生产环境配置类"""
    
    # 环境设置
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # 服务器配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8002"))
    
    # 安全配置
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "https://norma.minimax.chat,https://norma.space.minimaxi.com").split(",")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "norma.minimax.chat,norma.space.minimaxi.com").split(",")
    
    # API密钥配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # 数据库配置
    DATABASE_PATH = os.getenv("DATABASE_PATH", "/workspace/data/norma_database.db")
    LOGS_PATH = os.getenv("LOGS_PATH", "/workspace/data/norma_logs.db")
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "/workspace/logs/norma_production.log")
    ERROR_LOG_FILE = os.getenv("ERROR_LOG_FILE", "/workspace/logs/norma_errors.log")
    
    # 限流配置（生产环境更严格）
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "50"))  # 更严格的限制
    RATE_LIMIT_WINDOW = os.getenv("RATE_LIMIT_WINDOW", "1 minute")
    
    # 安全配置
    FORCE_HTTPS = os.getenv("FORCE_HTTPS", "true").lower() == "true"
    TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "norma.minimax.chat,norma.space.minimaxi.com").split(",")
    
    # WebSocket配置
    WS_MAX_CONNECTIONS = int(os.getenv("WS_MAX_CONNECTIONS", "500"))  # 更保守的限制
    WS_TIMEOUT = int(os.getenv("WS_TIMEOUT", "300"))
    
    # 监控配置
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    
    # 性能配置
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
    KEEPALIVE_TIMEOUT = int(os.getenv("KEEPALIVE_TIMEOUT", "5"))
    MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", "1000"))
    MAX_REQUESTS_JITTER = int(os.getenv("MAX_REQUESTS_JITTER", "100"))

# 初始化配置
config = ProductionConfig()

# =============================================================================
# 生产环境日志配置
# =============================================================================

def setup_production_logging():
    """配置生产环境日志系统"""
    # 创建日志目录
    Path("/workspace/logs").mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 文件处理器（所有日志）
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    
    # 错误日志处理器（仅ERROR及以上）
    error_handler = logging.FileHandler(config.ERROR_LOG_FILE, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(log_format, date_format)
    error_handler.setFormatter(error_formatter)
    
    # 控制台处理器（生产环境仅警告及以上）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("norma_production")

# 初始化日志
logger = setup_production_logging()

# =============================================================================
# 生产环境监控和健康检查
# =============================================================================

class ProductionMonitor:
    """生产环境监控类"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.last_health_check = datetime.now()
        self.system_metrics = {}
    
    def record_request(self, success: bool = True):
        """记录请求"""
        self.request_count += 1
        if not success:
            self.error_count += 1
    
    def get_uptime(self) -> str:
        """获取运行时间"""
        uptime = datetime.now() - self.start_time
        days, remainder = divmod(int(uptime.total_seconds()), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{days}天{hours}小时{minutes}分钟{seconds}秒"
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "network_connections": len(psutil.net_connections()),
                "process_count": len(psutil.pids()),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        metrics = self.get_system_metrics()
        self.last_health_check = datetime.now()
        
        # 检查系统资源
        issues = []
        if metrics.get('cpu_percent', 0) > 90:
            issues.append("CPU使用率过高")
        if metrics.get('memory_percent', 0) > 90:
            issues.append("内存使用率过高")
        if metrics.get('disk_percent', 0) > 90:
            issues.append("磁盘使用率过高")
        
        return {
            "status": "healthy" if not issues else "warning",
            "timestamp": datetime.now().isoformat(),
            "uptime": self.get_uptime(),
            "metrics": metrics,
            "issues": issues,
            "error_rate": self.get_error_rate(),
            "request_count": self.request_count,
            "error_count": self.error_count
        }

# 全局监控实例
monitor = ProductionMonitor()

# =============================================================================
# 生产环境安全装饰器
# =============================================================================

def production_rate_limit_key_func(request: Request) -> str:
    """生产环境限流键函数"""
    # 获取客户端IP
    client_ip = get_remote_address(request)
    
    # 检查API密钥
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key[:8]}..."  # 只记录前8位
    
    # 检查用户代理
    user_agent = request.headers.get("User-Agent", "")
    if "bot" in user_agent.lower() or "spider" in user_agent.lower():
        return f"bot:{client_ip}"
    
    return f"user:{client_ip}"

def monitor_requests(func):
    """请求监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            monitor.record_request(success=True)
            return result
        except Exception as e:
            monitor.record_request(success=False)
            logger.error(f"请求失败: {func.__name__} - {e}")
            raise
    return wrapper

# =============================================================================
# 导入原有模块和功能
# =============================================================================

# 尝试导入诺玛模块
NormaCoreAgent = None
NormaAdvancedFeaturesClass = None

try:
    from norma_core_agent import NormaCoreAgent, create_demo_data
    logger.info("诺玛核心模块导入成功")
except ImportError as e:
    logger.warning(f"诺玛核心模块导入失败: {e}")

try:
    # 创建简化的高级功能类
    class SimplifiedAdvancedFeatures:
        def demo_search_functionality(self, query: str):
            return {
                "search_query": query,
                "search_engine": "DuckDuckGo (生产环境模拟)",
                "results": [
                    {
                        "title": f"关于 '{query}' 的生产环境搜索结果",
                        "url": "https://production.example.com/research",
                        "snippet": "这是生产环境中的详细研究内容...",
                        "reliability": "高"
                    }
                ],
                "total_results": 1,
                "search_time": "0.2秒",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        def demo_multi_agent_collaboration(self):
            return {
                "agents": [
                    {
                        "name": "生产搜索专家",
                        "status": "活跃",
                        "tasks_completed": 1250,
                        "accuracy": 98.5,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    {
                        "name": "生产文档专家",
                        "status": "活跃",
                        "tasks_completed": 890,
                        "accuracy": 99.1,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    {
                        "name": "生产知识专家",
                        "status": "活跃",
                        "tasks_completed": 2103,
                        "accuracy": 97.8,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ],
                "collaboration_mode": "生产协调模式",
                "total_tasks": 4243,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    NormaAdvancedFeaturesClass = SimplifiedAdvancedFeatures
    logger.info("生产环境高级功能模块创建成功")
except ImportError as e:
    logger.error(f"生产环境高级功能模块创建失败: {e}")

# 导入原有的AG-UI相关类和函数
try:
    from main_agui import (
        EventType, MessageRole, AGUIEvent, RunAgentInput,
        EventEncoder, NormaSystemService, EnhancedConnectionManager,
        ChatMessage, SystemStatusResponse, BloodAnalysisRequest, SearchRequest
    )
    logger.info("原有AG-UI模块导入成功")
except ImportError as e:
    logger.error(f"原有AG-UI模块导入失败: {e}")
    sys.exit(1)

# =============================================================================
# 生产环境FastAPI应用
# =============================================================================

@asynccontextmanager
async def production_lifespan(app: FastAPI):
    """生产环境生命周期管理"""
    # 启动时执行
    logger.info("="*60)
    logger.info("诺玛·劳恩斯AI系统生产环境启动")
    logger.info("="*60)
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"运行环境: {config.ENVIRONMENT}")
    logger.info(f"调试模式: {config.DEBUG}")
    logger.info(f"服务器地址: {config.HOST}:{config.PORT}")
    logger.info(f"数据库路径: {config.DATABASE_PATH}")
    logger.info(f"日志文件: {config.LOG_FILE}")
    logger.info(f"错误日志: {config.ERROR_LOG_FILE}")
    logger.info(f"最大连接数: {config.WS_MAX_CONNECTIONS}")
    logger.info(f"限流配置: {config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_WINDOW}")
    logger.info("="*60)
    
    # 初始化数据库
    try:
        Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
        logger.info("数据库目录初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
    
    # 初始化服务
    try:
        global norma_service, manager, limiter
        norma_service = NormaSystemService()
        manager = EnhancedConnectionManager()
        limiter = Limiter(key_func=production_rate_limit_key_func)
        logger.info("生产环境服务初始化完成")
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        raise
    
    yield
    
    # 关闭时执行
    logger.info("="*60)
    logger.info("诺玛·劳恩斯AI系统生产环境关闭")
    logger.info(f"关闭时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总运行时间: {monitor.get_uptime()}")
    logger.info(f"总请求数: {monitor.request_count}")
    logger.info(f"错误数: {monitor.error_count}")
    logger.info(f"错误率: {monitor.get_error_rate():.2f}%")
    logger.info("="*60)

# 创建生产环境FastAPI应用
app = FastAPI(
    title="诺玛·劳恩斯AI系统API（生产环境版）",
    description="卡塞尔学院主控计算机AI系统后端服务 - 生产环境优化版本",
    version="3.1.2-agui-production",
    lifespan=production_lifespan,
    debug=config.DEBUG,
    docs_url="/docs" if config.DEBUG else None,  # 生产环境关闭文档
    redoc_url="/redoc" if config.DEBUG else None
)

# 添加限流异常处理器
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 配置CORS（生产环境严格配置）
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# 配置HTTPS重定向
if config.FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

# 配置信任主机
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.ALLOWED_HOSTS
)

# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    monitor.record_request(success=False)
    logger.error(f"全局异常: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误",
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# =============================================================================
# 生产环境API端点
# =============================================================================

@app.get("/")
@monitor_requests
async def production_root():
    """生产环境根路径"""
    return {
        "message": "诺玛·劳恩斯AI系统API服务（生产环境版）",
        "version": "3.1.2-agui-production",
        "environment": config.ENVIRONMENT,
        "status": "running",
        "uptime": monitor.get_uptime(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/health")
@monitor_requests
async def production_health():
    """生产环境健康检查"""
    health_status = monitor.health_check()
    logger.debug(f"健康检查: {health_status['status']}")
    return health_status

@app.get("/metrics")
@monitor_requests
async def production_metrics():
    """生产环境性能指标"""
    if not config.METRICS_ENABLED:
        return {"error": "指标收集已禁用"}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "uptime": monitor.get_uptime(),
        "requests": {
            "total": monitor.request_count,
            "errors": monitor.error_count,
            "error_rate": f"{monitor.get_error_rate():.2f}%"
        },
        "system": monitor.get_system_metrics(),
        "connections": manager.get_connection_stats() if 'manager' in globals() else {}
    }

# 导入原有的API端点（添加监控）
try:
    from main_agui import (
        run_agent_agui, get_system_status, chat_message,
        get_blood_analysis, get_system_logs, search_knowledge,
        get_security_status, get_agents_status
    )
    
    # 为原有端点添加监控装饰器
    app.post("/agui")(monitor_requests(run_agent_agui))
    app.get("/api/system/status")(monitor_requests(get_system_status))
    app.post("/api/chat/message")(monitor_requests(chat_message))
    app.post("/api/bloodline/analyze")(monitor_requests(get_blood_analysis))
    app.get("/api/system/logs")(monitor_requests(get_system_logs))
    app.post("/api/knowledge/search")(monitor_requests(search_knowledge))
    app.get("/api/security/status")(monitor_requests(get_security_status))
    app.get("/api/agents/status")(monitor_requests(get_agents_status))
    
    logger.info("原有API端点注册完成")
except ImportError as e:
    logger.error(f"原有API端点导入失败: {e}")

# =============================================================================
# 生产环境信号处理
# =============================================================================

def setup_signal_handlers():
    """设置信号处理器"""
    def signal_handler(signum, frame):
        logger.info(f"接收到信号 {signum}，开始优雅关闭...")
        asyncio.create_task(shutdown_handler())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

async def shutdown_handler():
    """关闭处理器"""
    logger.info("开始优雅关闭服务...")
    # 这里可以添加清理逻辑
    logger.info("服务已关闭")

# =============================================================================
# 生产环境启动
# =============================================================================

def main():
    """生产环境主函数"""
    # 设置信号处理器
    setup_signal_handlers()
    
    # 配置启动参数
    uvicorn_config = {
        "app": app,
        "host": config.HOST,
        "port": config.PORT,
        "log_level": config.LOG_LEVEL.lower(),
        "access_log": True,
        "loop": "uvloop",
        "workers": config.MAX_WORKERS,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "keepalive_timeout": config.KEEPALIVE_TIMEOUT,
        "max_requests": config.MAX_REQUESTS,
        "max_requests_jitter": config.MAX_REQUESTS_JITTER,
        "reload": False,
        "use_colors": False,
        "proxy_headers": True,
        "forwarded_allow_ips": "*"
    }
    
    logger.info("启动诺玛AI系统生产环境服务器...")
    logger.info(f"配置: {json.dumps({k: v for k, v in uvicorn_config.items() if k != 'app'}, indent=2)}")
    
    try:
        # 创建并运行服务器
        server = uvicorn.Server(uvicorn.Config(**uvicorn_config))
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()