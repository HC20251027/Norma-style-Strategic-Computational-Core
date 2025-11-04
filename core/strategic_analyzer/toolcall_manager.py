"""
ToolcallManager - 工具调用管理器

这是Toolcall抽象层的核心组件，负责协调各个子组件：
1. Agent: 负责调度（调用工具、管理上下文）
2. LLM: 负责规划（工具选择、参数推断）
3. MCP: 负责执行（运行工具函数）

实现职责分离的完整工作流程
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from ..models import (
    ToolcallRequest, 
    ToolcallResponse, 
    ToolcallContext, 
    ToolcallStatus,
    ToolSelectionResult,
    ParameterInferenceResult,
    ExecutionPlan,
    OptimizationConfig
)

from .tool_selector import ToolSelector
from .parameter_inferencer import ParameterInferencer
from .context_manager import ContextManager
from .performance_optimizer import PerformanceOptimizer
from .error_handler import ErrorHandler
from ..llm_interface import LLMInterface
from ..registry import ToolcallRegistry


logger = logging.getLogger(__name__)


class ToolcallManager:
    """工具调用管理器
    
    负责协调整个工具调用流程，实现职责分离：
    - Agent: 调度和管理
    - LLM: 规划和推断  
    - MCP: 执行工具
    """
    
    def __init__(
        self,
        llm_interface: LLMInterface,
        tool_registry: ToolcallRegistry,
        config: Optional[OptimizationConfig] = None
    ):
        self.llm_interface = llm_interface
        self.tool_registry = tool_registry
        self.config = config or OptimizationConfig()
        
        # 初始化子组件
        self.tool_selector = ToolSelector(llm_interface)
        self.parameter_inferencer = ParameterInferencer(llm_interface)
        self.context_manager = ContextManager()
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.error_handler = ErrorHandler(
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        
        # 状态跟踪
        self.active_requests: Dict[str, ToolcallRequest] = {}
        self.request_locks: Dict[str, asyncio.Lock] = {}
        
        logger.info("ToolcallManager初始化完成")
    
    async def process_request(self, request: ToolcallRequest) -> ToolcallResponse:
        """处理工具调用请求
        
        完整的工作流程：
        1. 创建/获取上下文
        2. 工具选择 (LLM)
        3. 参数推断 (LLM)
        4. 执行计划生成 (LLM)
        5. 性能优化 (Agent)
        6. 工具执行 (MCP)
        7. 错误处理 (Agent)
        8. 结果返回 (Agent)
        """
        start_time = time.time()
        request_id = request.request_id
        
        logger.info(f"开始处理工具调用请求: {request_id}")
        
        try:
            # 记录活跃请求
            self.active_requests[request_id] = request
            lock = self._get_request_lock(request_id)
            
            async with lock:
                # 1. 创建/获取上下文
                context = await self.context_manager.get_or_create_context(request)
                await self.context_manager.update_context(context, status=ToolcallStatus.SELECTING_TOOLS)
                
                # 2. 获取可用工具列表
                available_tools = await self._get_available_tools()
                await self.context_manager.set_available_tools(context, available_tools)
                
                # 3. 工具选择 (LLM负责)
                selection_result = await self.tool_selector.select_tools(
                    query=request.user_query,
                    available_tools=available_tools,
                    context=context,
                    max_tools=request.max_tools
                )
                
                # 检查是否需要澄清
                if selection_result.requires_clarification:
                    return await self._create_clarification_response(
                        request, selection_result, time.time() - start_time
                    )
                
                # 4. 参数推断 (LLM负责)
                await self.context_manager.update_context(context, status=ToolcallStatus.INFERING_PARAMETERS)
                
                inference_result = await self.parameter_inferencer.infer_parameters(
                    query=request.user_query,
                    selected_tools=selection_result.selected_tools,
                    tool_definitions=available_tools,
                    context=context
                )
                
                # 检查是否有缺失参数
                if inference_result.missing_parameters:
                    return await self._create_parameter_clarification_response(
                        request, inference_result, time.time() - start_time
                    )
                
                # 5. 执行计划生成 (LLM负责)
                execution_plan = await self.llm_interface.generate_execution_plan(
                    query=request.user_query,
                    selected_tools=selection_result.selected_tools,
                    inferred_parameters=inference_result.inferred_parameters,
                    context=context
                )
                
                # 验证执行计划
                is_valid, validation_error = await self.llm_interface.validate_plan(execution_plan, context)
                if not is_valid:
                    logger.warning(f"执行计划验证失败: {validation_error}")
                    # 使用默认计划
                    execution_plan = self._create_default_plan(selection_result.selected_tools)
                
                # 6. 性能优化 (Agent负责)
                await self.context_manager.update_context(context, status=ToolcallStatus.EXECUTING)
                
                # 定义MCP执行函数
                mcp_executor = self._create_mcp_executor(context)
                
                # 执行工具调用
                tool_results = await self.performance_optimizer.optimize_execution(
                    plan=execution_plan,
                    context=context,
                    executor_func=mcp_executor
                )
                
                # 7. 错误处理和恢复 (Agent负责)
                failed_tools = [result for result in tool_results if not result.get("success", False)]
                
                if failed_tools:
                    logger.warning(f"部分工具执行失败: {failed_tools}")
                    
                    # 尝试恢复
                    recovery_success, recovery_plan = await self.error_handler.handle_plan_execution_error(
                        plan=execution_plan,
                        failed_tools=[r["tool_id"] for r in failed_tools],
                        context=context,
                        executor_func=mcp_executor
                    )
                    
                    if not recovery_success:
                        logger.error("恢复失败，返回部分结果")
                
                # 8. 生成最终响应 (Agent负责)
                execution_time = time.time() - start_time
                
                response = ToolcallResponse(
                    request_id=request_id,
                    status=ToolcallStatus.COMPLETED,
                    success=len(failed_tools) == 0,
                    data=self._aggregate_results(tool_results),
                    tool_results=tool_results,
                    execution_time=execution_time,
                    metadata={
                        "selection_result": selection_result.to_dict(),
                        "inference_result": inference_result.to_dict(),
                        "execution_plan": execution_plan.to_dict(),
                        "failed_tools": [r["tool_id"] for r in failed_tools]
                    }
                )
                
                # 更新上下文
                await self.context_manager.update_context(context, status=ToolcallStatus.COMPLETED)
                
                logger.info(f"工具调用请求处理完成: {request_id}, 耗时: {execution_time:.2f}s")
                return response
                
        except Exception as e:
            logger.error(f"处理工具调用请求失败: {e}")
            
            execution_time = time.time() - start_time
            return ToolcallResponse(
                request_id=request_id,
                status=ToolcallStatus.FAILED,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
        
        finally:
            # 清理活跃请求
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def batch_process_requests(self, requests: List[ToolcallRequest]) -> List[ToolcallResponse]:
        """批量处理工具调用请求"""
        logger.info(f"批量处理 {len(requests)} 个工具调用请求")
        
        # 并发处理请求
        tasks = [self.process_request(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                responses.append(ToolcallResponse(
                    request_id=requests[i].request_id,
                    status=ToolcallStatus.FAILED,
                    success=False,
                    error=str(result)
                ))
            else:
                responses.append(result)
        
        return responses
    
    async def cancel_request(self, request_id: str) -> bool:
        """取消请求"""
        if request_id not in self.active_requests:
            return False
        
        try:
            # 这里可以实现具体的取消逻辑
            # 比如取消正在执行的工具调用
            
            logger.info(f"取消请求: {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消请求失败: {e}")
            return False
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取请求状态"""
        if request_id not in self.active_requests:
            return None
        
        request = self.active_requests[request_id]
        
        return {
            "request_id": request_id,
            "status": "processing",
            "created_at": request.created_at.isoformat(),
            "user_query": request.user_query,
            "session_id": request.session_id
        }
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        # 收集各组件的统计信息
        context_stats = await self.context_manager.get_context_statistics()
        performance_stats = self.performance_optimizer.get_performance_statistics()
        error_stats = self.error_handler.get_error_statistics()
        selection_stats = self.tool_selector.get_selection_statistics()
        inference_stats = self.parameter_inferencer.get_inference_statistics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "active_requests": len(self.active_requests),
            "context_manager": context_stats,
            "performance_optimizer": performance_stats,
            "error_handler": error_stats,
            "tool_selector": selection_stats,
            "parameter_inferencer": inference_stats,
            "config": self.config.to_dict()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查各组件状态
            context_health = await self.context_manager.health_check()
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "context_manager": context_health,
                    "tool_selector": {"status": "healthy"},
                    "parameter_inferencer": {"status": "healthy"},
                    "performance_optimizer": {"status": "healthy"},
                    "error_handler": {"status": "healthy"},
                    "llm_interface": {"status": "healthy"},
                    "tool_registry": {"status": "healthy"}
                }
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def optimize_configuration(self) -> OptimizationConfig:
        """基于历史数据优化配置"""
        optimized_config = await self.performance_optimizer.optimize_configuration()
        self.config = optimized_config
        
        logger.info("配置优化完成")
        return optimized_config
    
    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        try:
            # 从工具注册中心获取工具定义
            tools = self.tool_registry.get_all_tool_definitions()
            return [tool.to_dict() for tool in tools]
        except Exception as e:
            logger.error(f"获取可用工具失败: {e}")
            return []
    
    def _create_mcp_executor(self, context: ToolcallContext) -> Callable:
        """创建MCP执行器函数"""
        async def mcp_execute_tool(tool_id: str, tool_context: ToolcallContext) -> Dict[str, Any]:
            """MCP工具执行函数"""
            try:
                # 从注册中心获取工具实例
                tool_instance = self.tool_registry.get_tool_instance(tool_id)
                if not tool_instance:
                    return {
                        "tool_id": tool_id,
                        "success": False,
                        "error": f"工具实例不存在: {tool_id}"
                    }
                
                # 获取推断的参数
                inferred_params = context.metadata.get("inferred_parameters", {})
                tool_params = inferred_params.get(tool_id, {})
                
                # 使用错误处理器执行工具
                async def execute_tool():
                    start_time = time.time()
                    
                    # 检查是否是异步函数
                    if hasattr(tool_instance.execute, '__call__'):
                        if asyncio.iscoroutinefunction(tool_instance.execute):
                            result = await tool_instance.execute(**tool_params)
                        else:
                            result = tool_instance.execute(**tool_params)
                    else:
                        result = tool_instance.execute(**tool_params)
                    
                    execution_time = time.time() - start_time
                    
                    return {
                        "tool_id": tool_id,
                        "success": True,
                        "data": result,
                        "execution_time": execution_time
                    }
                
                # 通过错误处理器执行
                success, result, error_info = await self.error_handler.handle_execution_error(
                    tool_id=tool_id,
                    error=Exception("模拟错误以触发重试机制"),  # 这里应该传入实际错误
                    context=tool_context,
                    retry_func=execute_tool
                )
                
                if success:
                    return result
                else:
                    return {
                        "tool_id": tool_id,
                        "success": False,
                        "error": error_info.get("error", "执行失败")
                    }
                    
            except Exception as e:
                logger.error(f"MCP执行工具失败: {tool_id}, {e}")
                return {
                    "tool_id": tool_id,
                    "success": False,
                    "error": str(e)
                }
        
        return mcp_execute_tool
    
    async def _create_clarification_response(
        self, 
        request: ToolcallRequest, 
        selection_result: ToolSelectionResult, 
        execution_time: float
    ) -> ToolcallResponse:
        """创建澄清响应"""
        return ToolcallResponse(
            request_id=request.request_id,
            status=ToolcallStatus.FAILED,
            success=False,
            error="需要澄清",
            execution_time=execution_time,
            metadata={
                "clarification_type": "tool_selection",
                "clarification_questions": selection_result.clarification_questions,
                "reasoning": selection_result.reasoning
            }
        )
    
    async def _create_parameter_clarification_response(
        self, 
        request: ToolcallRequest, 
        inference_result: ParameterInferenceResult, 
        execution_time: float
    ) -> ToolcallResponse:
        """创建参数澄清响应"""
        clarification_questions = []
        
        for tool_id, missing_params in inference_result.missing_parameters.items():
            for param in missing_params:
                clarification_questions.append(f"请为工具 {tool_id} 提供参数 {param}")
        
        return ToolcallResponse(
            request_id=request.request_id,
            status=ToolcallStatus.FAILED,
            success=False,
            error="需要参数澄清",
            execution_time=execution_time,
            metadata={
                "clarification_type": "parameter_inference",
                "clarification_questions": clarification_questions,
                "missing_parameters": inference_result.missing_parameters,
                "reasoning": inference_result.reasoning
            }
        )
    
    def _create_default_plan(self, selected_tools: List[str]) -> ExecutionPlan:
        """创建默认执行计划"""
        return ExecutionPlan(
            tool_sequence=selected_tools,
            dependencies={},
            parallel_groups=[[tool] for tool in selected_tools],
            estimated_time=len(selected_tools) * 5.0
        )
    
    def _aggregate_results(self, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合工具结果"""
        aggregated = {
            "summary": {
                "total_tools": len(tool_results),
                "successful_tools": sum(1 for r in tool_results if r.get("success", False)),
                "failed_tools": sum(1 for r in tool_results if not r.get("success", False))
            },
            "results": tool_results,
            "combined_data": {}
        }
        
        # 合并成功工具的数据
        successful_results = [r for r in tool_results if r.get("success", False)]
        if successful_results:
            combined_data = {}
            for result in successful_results:
                if isinstance(result.get("data"), dict):
                    combined_data.update(result["data"])
            
            aggregated["combined_data"] = combined_data
        
        return aggregated
    
    def _get_request_lock(self, request_id: str) -> asyncio.Lock:
        """获取请求锁"""
        if request_id not in self.request_locks:
            self.request_locks[request_id] = asyncio.Lock()
        return self.request_locks[request_id]
    
    async def shutdown(self):
        """关闭管理器"""
        logger.info("正在关闭ToolcallManager...")
        
        # 取消所有活跃请求
        for request_id in list(self.active_requests.keys()):
            await self.cancel_request(request_id)
        
        # 清理资源
        self.active_requests.clear()
        self.request_locks.clear()
        
        logger.info("ToolcallManager已关闭")