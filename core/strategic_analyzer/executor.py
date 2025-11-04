"""
工具执行器
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

from .models import ToolExecutionContext, ToolExecutionResult
from .tool_registry import global_registry


logger = logging.getLogger(__name__)


class ToolExecutor:
    """工具执行器"""
    
    def __init__(self):
        self.execution_history: Dict[str, ToolExecutionResult] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0
        }
    
    async def execute_tool(self, context: ToolExecutionContext) -> ToolExecutionResult:
        """执行工具"""
        start_time = time.time()
        execution_id = context.request_id
        
        logger.info(f"开始执行工具: {context.tool_id}, 请求ID: {execution_id}")
        
        try:
            # 获取工具实例
            tool_instance = global_registry.get_tool_instance(context.tool_id)
            if not tool_instance:
                return ToolExecutionResult(
                    success=False,
                    error=f"工具实例不存在: {context.tool_id}",
                    tool_id=context.tool_id,
                    request_id=execution_id,
                    execution_time=time.time() - start_time
                )
            
            # 创建异步任务
            task = asyncio.create_task(
                self._execute_tool_async(tool_instance, context)
            )
            self.active_executions[execution_id] = task
            
            # 等待执行完成
            try:
                result = await asyncio.wait_for(
                    task, 
                    timeout=context.timeout
                )
                
                # 更新统计信息
                self._update_stats(True, time.time() - start_time)
                
                logger.info(f"工具执行成功: {context.tool_id}, 耗时: {result.execution_time:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                task.cancel()
                return ToolExecutionResult(
                    success=False,
                    error=f"工具执行超时: {context.timeout}s",
                    tool_id=context.tool_id,
                    request_id=execution_id,
                    execution_time=context.timeout
                )
            
        except Exception as e:
            error_msg = f"工具执行异常: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # 更新统计信息
            self._update_stats(False, time.time() - start_time)
            
            return ToolExecutionResult(
                success=False,
                error=error_msg,
                tool_id=context.tool_id,
                request_id=execution_id,
                execution_time=time.time() - start_time
            )
        
        finally:
            # 清理活跃执行记录
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_tool_async(self, tool_instance, context: ToolExecutionContext) -> ToolExecutionResult:
        """异步执行工具"""
        start_time = time.time()
        
        try:
            # 执行工具
            if asyncio.iscoroutinefunction(tool_instance.execute):
                result_data = await tool_instance.execute(**context.parameters)
            else:
                result_data = tool_instance.execute(**context.parameters)
            
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                success=True,
                data=result_data,
                tool_id=context.tool_id,
                request_id=context.request_id,
                execution_time=execution_time,
                metadata={
                    "parameters": context.parameters,
                    "user_id": context.user_id,
                    "session_id": context.session_id
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"工具执行失败: {str(e)}"
            
            return ToolExecutionResult(
                success=False,
                error=error_msg,
                tool_id=context.tool_id,
                request_id=context.request_id,
                execution_time=execution_time,
                metadata={
                    "parameters": context.parameters,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "traceback": traceback.format_exc()
                }
            )
    
    async def execute_tool_batch(self, contexts: list[ToolExecutionContext]) -> list[ToolExecutionResult]:
        """批量执行工具"""
        logger.info(f"批量执行工具: {len(contexts)}个任务")
        
        # 创建执行任务
        tasks = [
            self.execute_tool(context) 
            for context in contexts
        ]
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolExecutionResult(
                    success=False,
                    error=f"批量执行异常: {str(result)}",
                    tool_id=contexts[i].tool_id,
                    request_id=contexts[i].request_id
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        if execution_id in self.active_executions:
            task = self.active_executions[execution_id]
            task.cancel()
            del self.active_executions[execution_id]
            logger.info(f"取消执行任务: {execution_id}")
            return True
        return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态"""
        if execution_id in self.active_executions:
            task = self.active_executions[execution_id]
            return {
                "status": "running",
                "execution_id": execution_id,
                "done": task.done(),
                "cancelled": task.cancelled()
            }
        
        if execution_id in self.execution_history:
            result = self.execution_history[execution_id]
            return {
                "status": "completed",
                "execution_id": execution_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat()
            }
        
        return None
    
    def get_execution_history(self, limit: int = 100) -> list[Dict[str, Any]]:
        """获取执行历史"""
        history = list(self.execution_history.values())
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "tool_id": result.tool_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat(),
                "request_id": result.request_id,
                "error": result.error
            }
            for result in history[:limit]
        ]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            **self.execution_stats,
            "active_executions": len(self.active_executions),
            "success_rate": (
                self.execution_stats["successful_executions"] / 
                max(self.execution_stats["total_executions"], 1)
            ) * 100,
            "average_execution_time": (
                self.execution_stats["total_execution_time"] / 
                max(self.execution_stats["total_executions"], 1)
            )
        }
    
    def _update_stats(self, success: bool, execution_time: float):
        """更新统计信息"""
        self.execution_stats["total_executions"] += 1
        self.execution_stats["total_execution_time"] += execution_time
        
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
    
    def cleanup_old_executions(self, max_age_hours: int = 24):
        """清理旧的执行记录"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for execution_id, result in self.execution_history.items():
            if result.timestamp.timestamp() < cutoff_time:
                to_remove.append(execution_id)
        
        for execution_id in to_remove:
            del self.execution_history[execution_id]
        
        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 条旧的执行记录")