"""
非阻塞等待系统API接口
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import logging

from .config import NonBlockingConfig, TaskPriority
from .task_manager import TaskManager, Task, TaskStatus
from .progress_pusher import ProgressPusher, ProgressEvent
from .async_result import AsyncResultManager, AsyncResult, ResultStatus
from .realtime_status import RealtimeStatusManager, StatusUpdate
from .timeout_handler import TimeoutHandler, TimeoutConfig, FailureType
from .user_experience import UserExperienceOptimizer, UXEvent
from .event_system import event_system


# Pydantic模型
class TaskCreateRequest(BaseModel):
    name: str = Field(..., description="任务名称")
    function_path: str = Field(..., description="函数路径")
    args: List[Any] = Field(default_factory=list, description="位置参数")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="关键字参数")
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[int] = Field(None, description="超时时间（秒）")
    max_retries: Optional[int] = Field(None, description="最大重试次数")
    correlation_id: Optional[str] = Field(None, description="关联ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class TaskResponse(BaseModel):
    task_id: str
    name: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    queue_position: Optional[int] = None
    estimated_time_remaining: Optional[float] = None


class ProgressResponse(BaseModel):
    task_id: str
    progress: float
    message: str
    timestamp: str
    estimated_time_remaining: Optional[float] = None
    speed: Optional[float] = None


class StatusUpdateResponse(BaseModel):
    task_id: str
    old_status: str
    new_status: str
    timestamp: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UXEventResponse(BaseModel):
    event_type: str
    message: str
    tone: str
    timestamp: str
    task_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class ResultResponse(BaseModel):
    result_id: str
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    age: float


class NonBlockingAPI:
    """非阻塞等待系统API"""
    
    def __init__(self, 
                 task_manager: TaskManager,
                 progress_pusher: ProgressPusher,
                 result_manager: AsyncResultManager,
                 status_manager: RealtimeStatusManager,
                 timeout_handler: TimeoutHandler,
                 ux_optimizer: UserExperienceOptimizer,
                 config: Optional[NonBlockingConfig] = None):
        
        self.task_manager = task_manager
        self.progress_pusher = progress_pusher
        self.result_manager = result_manager
        self.status_manager = status_manager
        self.timeout_handler = timeout_handler
        self.ux_optimizer = ux_optimizer
        self.config = config or NonBlockingConfig()
        
        self.app = FastAPI(title="非阻塞等待系统API", version="1.0.0")
        self._setup_routes()
        
        # WebSocket连接管理
        self.websocket_connections: Dict[str, WebSocket] = {}
    
    def _setup_routes(self) -> None:
        """设置路由"""
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "task_manager": "running",
                    "progress_pusher": "running" if self.progress_pusher.running else "stopped",
                    "result_manager": "running" if self.result_manager.running else "stopped",
                    "status_manager": "running" if self.status_manager.running else "stopped",
                    "timeout_handler": "running" if self.timeout_handler.running else "stopped",
                    "ux_optimizer": "running" if self.ux_optimizer.running else "stopped"
                }
            }
        
        @self.app.post("/tasks", response_model=Dict[str, str])
        async def create_task(request: TaskCreateRequest):
            """创建任务"""
            try:
                # 动态导入函数
                module_path, func_name = request.function_path.rsplit('.', 1)
                module = __import__(module_path)
                func = getattr(module, func_name)
                
                # 创建异步结果
                result_id = await self.result_manager.create_result(
                    task_id="pending",  # 将在任务创建后更新
                    expires_in=3600  # 1小时过期
                )
                
                # 提交任务
                task_id = await self.task_manager.submit_task(
                    name=request.name,
                    func=func,
                    *request.args,
                    priority=request.priority,
                    timeout=request.timeout,
                    max_retries=request.max_retries,
                    correlation_id=request.correlation_id,
                    metadata=request.metadata,
                    **request.kwargs
                )
                
                # 更新结果的任务ID
                result = self.result_manager.storage.get(result_id)
                if result:
                    result.task_id = task_id
                
                # 配置超时处理
                if request.timeout:
                    timeout_config = TimeoutConfig(hard_timeout=float(request.timeout))
                    self.timeout_handler.configure_task_timeout(task_id, timeout_config)
                    await self.timeout_handler.start_task_monitoring(task_id)
                
                # 发送队列信息
                if request.correlation_id:
                    await self.ux_optimizer.send_queue_info(task_id, request.correlation_id)
                
                return {
                    "task_id": task_id,
                    "result_id": result_id
                }
                
            except Exception as e:
                logging.error(f"Failed to create task: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/tasks/{task_id}", response_model=TaskResponse)
        async def get_task(task_id: str):
            """获取任务信息"""
            task = self.task_manager.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            queue_position = self.task_manager.get_queue_position(task_id)
            
            return TaskResponse(
                task_id=task.id,
                name=task.name,
                status=task.status.value,
                progress=task.progress,
                created_at=task.created_at.isoformat(),
                started_at=task.started_at.isoformat() if task.started_at else None,
                completed_at=task.completed_at.isoformat() if task.completed_at else None,
                result=task.result,
                error=task.error,
                metadata=task.metadata,
                queue_position=queue_position,
                estimated_time_remaining=self.progress_pusher.predictor.predict_remaining_time(task_id, task.progress)
            )
        
        @self.app.get("/tasks", response_model=List[TaskResponse])
        async def list_tasks(
            status: Optional[TaskStatus] = None,
            correlation_id: Optional[str] = None,
            limit: int = 50
        ):
            """获取任务列表"""
            if correlation_id:
                tasks = self.task_manager.get_tasks_by_correlation(correlation_id)
            elif status:
                tasks = self.task_manager.get_tasks_by_status(status)
            else:
                tasks = list(self.task_manager.tasks.values())
            
            # 限制返回数量
            tasks = tasks[-limit:]
            
            result = []
            for task in tasks:
                queue_position = self.task_manager.get_queue_position(task.id)
                result.append(TaskResponse(
                    task_id=task.id,
                    name=task.name,
                    status=task.status.value,
                    progress=task.progress,
                    created_at=task.created_at.isoformat(),
                    started_at=task.started_at.isoformat() if task.started_at else None,
                    completed_at=task.completed_at.isoformat() if task.completed_at else None,
                    result=task.result,
                    error=task.error,
                    metadata=task.metadata,
                    queue_position=queue_position,
                    estimated_time_remaining=self.progress_pusher.predictor.predict_remaining_time(task.id, task.progress)
                ))
            
            return result
        
        @self.app.delete("/tasks/{task_id}")
        async def cancel_task(task_id: str):
            """取消任务"""
            success = await self.task_manager.cancel_task(task_id)
            if not success:
                raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
            return {"message": "Task cancelled successfully"}
        
        @self.app.get("/tasks/{task_id}/progress", response_model=ProgressResponse)
        async def get_task_progress(task_id: str):
            """获取任务进度"""
            task = self.task_manager.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            
            prediction = self.progress_pusher.get_prediction(task_id)
            
            return ProgressResponse(
                task_id=task_id,
                progress=task.progress,
                message=self.progress_pusher._get_progress_message(task),
                timestamp=datetime.now(timezone.utc).isoformat(),
                estimated_time_remaining=prediction["estimated_time_remaining"] if prediction else None,
                speed=prediction["average_speed"] if prediction else None
            )
        
        @self.app.get("/results/{result_id}", response_model=ResultResponse)
        async def get_result(result_id: str):
            """获取结果"""
            result = self.result_manager.get_result_sync(result_id)
            if not result:
                raise HTTPException(status_code=404, detail="Result not found")
            
            return ResultResponse(
                result_id=result.id,
                task_id=result.task_id,
                status=result.status.value,
                result=result.result,
                error=result.error,
                created_at=result.created_at.isoformat(),
                age=result.age
            )
        
        @self.app.get("/results/{result_id}/wait")
        async def wait_for_result(result_id: str, timeout: Optional[float] = None):
            """等待结果"""
            try:
                result = await self.result_manager.get_result(result_id, timeout)
                return ResultResponse(
                    result_id=result.id,
                    task_id=result.task_id,
                    status=result.status.value,
                    result=result.result,
                    error=result.error,
                    created_at=result.created_at.isoformat(),
                    age=result.age
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Timeout waiting for result")
        
        @self.app.get("/stats")
        async def get_stats():
            """获取系统统计"""
            return {
                "task_manager": self.task_manager.get_stats(),
                "result_manager": self.result_manager.get_stats(),
                "status_manager": self.status_manager.get_connection_stats(),
                "timeout_handler": self.timeout_handler.get_stats(),
                "ux_optimizer": self.ux_optimizer.get_stats()
            }
        
        @self.app.websocket("/ws/{connection_id}")
        async def websocket_endpoint(websocket: WebSocket, connection_id: str):
            """WebSocket端点"""
            await websocket.accept()
            self.websocket_connections[connection_id] = websocket
            
            try:
                # 注册连接
                self.status_manager.add_connection(connection_id, websocket)
                
                # 保持连接
                while True:
                    # 接收消息
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    # 处理订阅请求
                    if message_data.get("action") == "subscribe":
                        await self._handle_subscription(connection_id, message_data)
                    elif message_data.get("action") == "unsubscribe":
                        await self._handle_unsubscription(connection_id, message_data)
                    
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
            finally:
                # 清理连接
                self.status_manager.remove_connection(connection_id)
                self.websocket_connections.pop(connection_id, None)
    
    async def _handle_subscription(self, connection_id: str, message_data: Dict[str, Any]):
        """处理订阅请求"""
        subscription_type = message_data.get("type")
        target_id = message_data.get("id")
        
        if subscription_type == "task":
            self.status_manager.subscribe_to_task(connection_id, target_id)
            self.progress_pusher.subscribe(target_id, lambda event: self._send_progress_to_connection(connection_id, event))
        elif subscription_type == "user":
            self.status_manager.subscribe_to_user(connection_id, target_id)
    
    async def _handle_unsubscription(self, connection_id: str, message_data: Dict[str, Any]):
        """处理取消订阅请求"""
        subscription_type = message_data.get("type")
        target_id = message_data.get("id")
        
        if subscription_type == "task":
            self.status_manager.unsubscribe_from_task(connection_id, target_id)
        elif subscription_type == "user":
            self.status_manager.unsubscribe_from_user(connection_id, target_id)
    
    async def _send_progress_to_connection(self, connection_id: str, progress_event: ProgressEvent):
        """发送进度到连接"""
        websocket = self.websocket_connections.get(connection_id)
        if not websocket:
            return
        
        try:
            response = ProgressResponse(
                task_id=progress_event.task_id,
                progress=progress_event.progress,
                message=progress_event.message,
                timestamp=progress_event.timestamp.isoformat(),
                estimated_time_remaining=progress_event.estimated_time_remaining,
                speed=progress_event.speed
            )
            
            await websocket.send_text(response.json())
        except Exception as e:
            logging.error(f"Failed to send progress to connection {connection_id}: {e}")
    
    def get_app(self) -> FastAPI:
        """获取FastAPI应用"""
        return self.app