"""
MCP服务器实现
提供RESTful API接口和WebSocket支持
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .models import ToolExecutionContext, ToolExecutionResult
from .tool_registry import global_registry
from .executor import ToolExecutor
from .result_handler import result_processor, ResultFormat
from ..skills import skill_manager, SkillExecutionContext
from ..security.permission_manager import permission_manager, User, UserRole, PermissionType
from ..utils.tool_runner import tool_runner, batch_processor
from ..config.config_manager import config_manager


logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="MCP工具系统",
    description="Model Context Protocol 工具系统API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class ToolExecutionRequest(BaseModel):
    tool_id: str = Field(..., description="工具ID")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    user_id: str = Field(default="user", description="用户ID")
    format_type: str = Field(default="json", description="结果格式")
    timeout: Optional[int] = Field(default=None, description="超时时间")

class BatchExecutionRequest(BaseModel):
    tasks: List[Dict[str, Any]] = Field(..., description="批量任务列表")
    user_id: str = Field(default="user", description="用户ID")

class SkillExecutionRequest(BaseModel):
    skill_id: str = Field(..., description="技能ID")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="技能参数")
    user_id: str = Field(default="user", description="用户ID")

class UserCreateRequest(BaseModel):
    id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    role: UserRole = Field(..., description="用户角色")
    permissions: List[PermissionType] = Field(default_factory=list, description="权限列表")

# 依赖注入
async def get_current_user(user_id: str = "user") -> User:
    """获取当前用户"""
    user = permission_manager.users.get(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user

# 健康检查接口
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# 工具管理接口
@app.get("/tools")
async def list_tools(category: Optional[str] = None, status: Optional[str] = None):
    """列出工具"""
    try:
        from ..core.models import ToolCategory, ToolStatus
        
        category_filter = ToolCategory(category) if category else None
        status_filter = ToolStatus(status) if status else None
        
        tools = global_registry.list_tools(
            category=category_filter,
            status=status_filter
        )
        
        return {
            "success": True,
            "tools": [tool.to_dict() for tool in tools],
            "total": len(tools)
        }
    except Exception as e:
        logger.error(f"列出工具失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/categories")
async def list_tool_categories():
    """按分类列出工具"""
    try:
        categories = global_registry.list_tools_by_category()
        
        result = {}
        for category, tools in categories.items():
            result[category.value] = {
                "tools": [tool.to_dict() for tool in tools],
                "count": len(tools)
            }
        
        return {
            "success": True,
            "categories": result
        }
    except Exception as e:
        logger.error(f"获取工具分类失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/{tool_id}")
async def get_tool(tool_id: str):
    """获取工具信息"""
    tool = global_registry.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="工具不存在")
    
    return {
        "success": True,
        "tool": tool.to_dict()
    }

@app.post("/tools/{tool_id}/execute")
async def execute_tool(
    tool_id: str,
    request: ToolExecutionRequest,
    background_tasks: BackgroundTasks
):
    """执行工具"""
    try:
        # 权限检查
        allowed = await permission_manager.check_permission(
            tool_id, request.parameters, request.user_id
        )
        if not allowed:
            raise HTTPException(status_code=403, detail="权限不足")
        
        # 验证工具
        valid, error_msg = global_registry.validate_tool_execution(tool_id, request.parameters)
        if not valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 执行工具
        result = await tool_runner.run_tool(
            tool_id=tool_id,
            parameters=request.parameters,
            user_id=request.user_id,
            format_type=request.format_type
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"工具执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/execute/batch")
async def execute_tools_batch(request: BatchExecutionRequest):
    """批量执行工具"""
    try:
        # 权限检查
        for task in request.tasks:
            tool_id = task.get("tool_id")
            parameters = task.get("parameters", {})
            
            allowed = await permission_manager.check_permission(
                tool_id, parameters, request.user_id
            )
            if not allowed:
                raise HTTPException(status_code=403, detail=f"用户 {request.user_id} 无权限执行工具 {tool_id}")
        
        # 批量执行
        results = await batch_processor.process_batch(request.tasks)
        
        return {
            "success": True,
            "results": results,
            "total_tasks": len(request.tasks),
            "successful_tasks": sum(1 for r in results.values() if r.get("success")),
            "failed_tasks": sum(1 for r in results.values() if not r.get("success"))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 技能管理接口
@app.get("/skills")
async def list_skills():
    """列出技能"""
    try:
        skills = skill_manager.list_skills()
        
        return {
            "success": True,
            "skills": [skill.to_dict() for skill in skills],
            "total": len(skills)
        }
    except Exception as e:
        logger.error(f"列出技能失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/skills/{skill_id}")
async def get_skill(skill_id: str):
    """获取技能信息"""
    skill_def = skill_manager.get_skill_definition(skill_id)
    if not skill_def:
        raise HTTPException(status_code=404, detail="技能不存在")
    
    return {
        "success": True,
        "skill": skill_def.to_dict()
    }

@app.post("/skills/{skill_id}/execute")
async def execute_skill(skill_id: str, request: SkillExecutionRequest):
    """执行技能"""
    try:
        # 权限检查（简化实现）
        # 实际应用中应该检查技能执行权限
        
        # 创建执行上下文
        context = SkillExecutionContext(
            skill_id=skill_id,
            parameters=request.parameters,
            user_id=request.user_id
        )
        
        # 执行技能
        result = await skill_manager.execute_skill(skill_id, request.parameters, context)
        
        return {
            "success": result.success,
            "skill_id": result.skill_id,
            "request_id": result.request_id,
            "steps_executed": result.steps_executed,
            "output": result.output,
            "error": result.error,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"技能执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 权限管理接口
@app.get("/permissions/check")
async def check_permission(tool_id: str, user_id: str = "user"):
    """检查权限"""
    try:
        # 简单的权限检查（不需要参数验证）
        allowed = await permission_manager.check_permission(tool_id, {}, user_id)
        
        return {
            "success": True,
            "allowed": allowed,
            "tool_id": tool_id,
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"权限检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/permissions/users")
async def create_user(request: UserCreateRequest):
    """创建用户"""
    try:
        user = User(
            id=request.id,
            username=request.username,
            role=request.role,
            permissions=set(request.permissions)
        )
        
        success = permission_manager.add_user(user)
        if not success:
            raise HTTPException(status_code=400, detail="用户创建失败")
        
        return {
            "success": True,
            "user": {
                "id": user.id,
                "username": user.username,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions]
            }
        }
        
    except Exception as e:
        logger.error(f"创建用户失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/permissions/users")
async def list_users():
    """列出用户"""
    try:
        users = []
        for user in permission_manager.users.values():
            users.append({
                "id": user.id,
                "username": user.username,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions],
                "is_active": user.is_active,
                "last_active": user.last_active.isoformat() if user.last_active else None
            })
        
        return {
            "success": True,
            "users": users,
            "total": len(users)
        }
    except Exception as e:
        logger.error(f"列出用户失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/permissions/logs")
async def get_access_logs(limit: int = 100, user_id: Optional[str] = None, tool_id: Optional[str] = None):
    """获取访问日志"""
    try:
        logs = permission_manager.get_access_logs(user_id, tool_id, limit)
        
        return {
            "success": True,
            "logs": logs,
            "total": len(logs)
        }
    except Exception as e:
        logger.error(f"获取访问日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 统计信息接口
@app.get("/statistics")
async def get_statistics():
    """获取系统统计信息"""
    try:
        # 工具统计
        tool_stats = global_registry.get_tool_statistics()
        
        # 技能统计
        skill_stats = skill_manager.get_execution_statistics()
        
        # 安全统计
        security_stats = permission_manager.get_security_statistics()
        
        # 执行统计
        executor = ToolExecutor()
        execution_stats = executor.get_execution_statistics()
        
        return {
            "success": True,
            "statistics": {
                "tools": tool_stats,
                "skills": skill_stats,
                "security": security_stats,
                "execution": execution_stats,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/statistics")
async def get_cache_statistics():
    """获取缓存统计"""
    try:
        cache_stats = result_processor.get_cache_statistics()
        
        return {
            "success": True,
            "cache_statistics": cache_stats
        }
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 配置管理接口
@app.get("/config")
async def get_config():
    """获取配置"""
    try:
        config = config_manager.get_config()
        
        return {
            "success": True,
            "config": config
        }
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/reload")
async def reload_config():
    """重新加载配置"""
    try:
        config_manager.reload_config()
        
        return {
            "success": True,
            "message": "配置重新加载成功"
        }
    except Exception as e:
        logger.error(f"重新加载配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务器错误",
            "detail": str(exc)
        }
    )

# 启动事件
@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("MCP服务器启动中...")
    
    # 加载配置
    config_manager.load_config()
    
    # 初始化工具注册
    await initialize_tools()
    
    # 初始化技能
    await initialize_skills()
    
    logger.info("MCP服务器启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    logger.info("MCP服务器关闭中...")
    
    # 清理资源
    executor = ToolExecutor()
    executor.cleanup_old_executions()
    
    result_processor.cleanup_cache()
    
    logger.info("MCP服务器已关闭")

async def initialize_tools():
    """初始化工具注册"""
    try:
        from ..tools.system_monitor import (
            CPUMonitorTool, MemoryMonitorTool, DiskMonitorTool,
            NetworkMonitorTool, ProcessMonitorTool
        )
        from ..tools.network_security import (
            PortScannerTool, VulnerabilityScannerTool, NetworkAnalyzerTool
        )
        from ..tools.bloodline_analysis import (
            BloodlinePurityAnalyzer, HeritageTracker
        )
        
        # 注册系统监控工具
        tools = [
            CPUMonitorTool(),
            MemoryMonitorTool(),
            DiskMonitorTool(),
            NetworkMonitorTool(),
            ProcessMonitorTool(),
            PortScannerTool(),
            VulnerabilityScannerTool(),
            NetworkAnalyzerTool(),
            BloodlinePurityAnalyzer(),
            HeritageTracker()
        ]
        
        for tool in tools:
            success = global_registry.register_tool(tool.__class__, tool.get_tool_definition())
            if success:
                logger.info(f"工具注册成功: {tool.get_tool_definition().name}")
            else:
                logger.warning(f"工具注册失败: {tool.get_tool_definition().name}")
        
        logger.info(f"工具初始化完成，共注册 {len(tools)} 个工具")
        
    except Exception as e:
        logger.error(f"工具初始化失败: {e}")

async def initialize_skills():
    """初始化技能"""
    try:
        from ..skills import AtomicSkill, CompositeSkill, SkillStep
        
        # 创建示例复合技能
        system_check = CompositeSkill(
            skill_id="system_check",
            steps=[
                SkillStep(
                    id="cpu_check",
                    name="CPU检查",
                    tool_id="cpu_monitor",
                    parameters={"duration": 10}
                ),
                SkillStep(
                    id="memory_check",
                    name="内存检查",
                    tool_id="memory_monitor",
                    parameters={"duration": 10}
                )
            ]
        )
        
        skill_manager.register_skill(system_check)
        
        # 创建原子技能
        cpu_skill = AtomicSkill("cpu_monitor")
        skill_manager.register_skill(cpu_skill)
        
        logger.info("技能初始化完成")
        
    except Exception as e:
        logger.error(f"技能初始化失败: {e}")

# 创建MCP服务器实例
mcp_server = app

def create_server():
    """创建服务器"""
    return mcp_server

if __name__ == "__main__":
    # 开发模式运行
    config = config_manager.get_config()
    
    uvicorn.run(
        "mcp.core.mcp_server:mcp_server",
        host=config.get("server", {}).get("host", "0.0.0.0"),
        port=config.get("server", {}).get("port", 8000),
        reload=config.get("server", {}).get("debug", False),
        log_level="info"
    )