"""
智能化工具调用系统主集成
整合所有智能工具调用组件
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import json

from .tool_selector import (
    ContextualToolSelector, ToolCapability, ContextInfo, ToolMatch
)
from .parameter_inference import (
    ParameterInferenceEngine, ParameterSpec, ParameterType, InferredParameter, ValidationResult
)
from .concurrent_executor import (
    ConcurrentToolExecutor, ToolCall, ExecutionResult, ExecutionStrategy, ResultStatus
)
from .cache_manager import (
    CacheManager, CacheStrategy, PerformanceMonitor
)
from .chain_orchestrator import (
    IntelligentChainOrchestrator, ChainTemplate, ChainStep, StepType, ChainExecution
)

logger = logging.getLogger(__name__)


@dataclass
class IntelligentToolConfig:
    """智能工具系统配置"""
    # 工具选择配置
    tool_selection_enabled: bool = True
    max_tool_suggestions: int = 5
    min_relevance_threshold: float = 0.3
    
    # 参数推断配置
    parameter_inference_enabled: bool = True
    auto_validation: bool = True
    strict_mode: bool = False
    
    # 并发执行配置
    max_concurrent_tools: int = 10
    default_timeout: float = 30.0
    enable_retry: bool = True
    max_retries: int = 2
    
    # 缓存配置
    cache_enabled: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    
    # 链编排配置
    chain_orchestration_enabled: bool = True
    enable_parallel_chains: bool = True
    chain_timeout: float = 300.0
    
    # 性能优化配置
    performance_monitoring: bool = True
    auto_optimization: bool = True
    optimization_interval: int = 300  # 5分钟


class IntelligentToolSystem:
    """智能化工具调用系统"""
    
    def __init__(self, config: Optional[IntelligentToolConfig] = None):
        self.config = config or IntelligentToolConfig()
        
        # 初始化各个组件
        self.tool_selector = ContextualToolSelector()
        self.parameter_inference = ParameterInferenceEngine()
        self.concurrent_executor = ConcurrentToolExecutor(
            max_workers=self.config.max_concurrent_tools,
            default_timeout=self.config.default_timeout
        )
        
        if self.config.cache_enabled:
            cache_config = {
                'max_size': self.config.max_cache_size,
                'default_ttl': self.config.cache_ttl,
                'strategy': self.config.cache_strategy.value
            }
            self.cache_manager = CacheManager(cache_config)
        else:
            self.cache_manager = None
        
        self.chain_orchestrator = IntelligentChainOrchestrator()
        
        # 注册的内置工具
        self._register_builtin_tools()
        
        # 性能统计
        self.system_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'cache_hits': 0,
            'chains_executed': 0,
            'start_time': datetime.now()
        }
        
        logger.info("智能工具调用系统初始化完成")
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        # 这里可以注册一些示例工具
        
        # 示例：数据处理工具
        async def process_data(data: List[Any], operation: str = "sum") -> Any:
            """数据处理工具"""
            if operation == "sum":
                return sum(data) if data else 0
            elif operation == "avg":
                return sum(data) / len(data) if data else 0
            elif operation == "count":
                return len(data)
            else:
                return data
        
        # 示例：文本处理工具
        async def process_text(text: str, operation: str = "upper") -> str:
            """文本处理工具"""
            if operation == "upper":
                return text.upper()
            elif operation == "lower":
                return text.lower()
            elif operation == "title":
                return text.title()
            else:
                return text
        
        # 注册工具
        self.register_tool("process_data", process_data)
        self.register_tool("process_text", process_text)
        
        logger.info("内置工具注册完成")
    
    def register_tool(self, tool_name: str, tool_function: Callable):
        """注册工具"""
        # 注册到工具选择器
        tool_capability = self._create_tool_capability(tool_name, tool_function)
        self.tool_selector.register_tool(tool_capability)
        
        # 注册到并发执行器
        self.concurrent_executor.register_tool(tool_name, tool_function)
        
        # 注册到链编排器
        self.chain_orchestrator.register_tool(tool_name, tool_function)
        
        logger.info(f"工具注册成功: {tool_name}")
    
    def _create_tool_capability(self, tool_name: str, tool_function: Callable) -> ToolCapability:
        """创建工具能力描述"""
        # 简单的工具能力推断
        from .tool_selector import ToolCategory
        
        # 基于函数名推断类别
        if "data" in tool_name.lower():
            category = ToolCategory.DATA_PROCESSING
        elif "text" in tool_name.lower():
            category = ToolCategory.TEXT_PROCESSING
        elif "file" in tool_name.lower():
            category = ToolCategory.FILE_MANAGEMENT
        elif "web" in tool_name.lower():
            category = ToolCategory.WEB_SCRAPING
        else:
            category = ToolCategory.CODE_EXECUTION
        
        return ToolCapability(
            name=tool_name,
            category=category,
            description=f"工具: {tool_name}",
            keywords=[tool_name.replace("_", " ")],
            input_types=["any"],
            output_types=["any"],
            complexity_score=0.5,
            execution_time_estimate=1.0
        )
    
    async def execute_intelligent_tool_call(
        self,
        user_intent: str,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行智能化工具调用"""
        start_time = datetime.now()
        
        try:
            self.system_stats['total_calls'] += 1
            
            # 构建上下文信息
            context_info = ContextInfo(
                user_intent=user_intent,
                conversation_history=context.get('conversation_history', []) if context else [],
                current_task=context.get('current_task') if context else None,
                available_tools=list(self.tool_selector.tool_registry.values()),
                user_preferences=user_preferences or {},
                system_state=context or {}
            )
            
            # 1. 工具选择
            if self.config.tool_selection_enabled:
                tool_matches = self.tool_selector.select_tools(context_info, self.config.max_tool_suggestions)
                
                if not tool_matches or tool_matches[0].relevance_score < self.config.min_relevance_threshold:
                    return {
                        'status': 'no_suitable_tool',
                        'message': '未找到合适的工具',
                        'tool_suggestions': []
                    }
                
                selected_tool = tool_matches[0].tool
                logger.info(f"选择工具: {selected_tool.name} (相关性: {tool_matches[0].relevance_score:.2f})")
            else:
                # 默认选择第一个工具
                selected_tool = list(self.tool_selector.tool_registry.values())[0]
            
            # 2. 参数推断
            inferred_params = {}
            if self.config.parameter_inference_enabled:
                # 创建参数规范 (简化实现)
                param_specs = [
                    ParameterSpec("data", ParameterType.LIST, required=False),
                    ParameterSpec("text", ParameterType.STRING, required=False),
                    ParameterSpec("operation", ParameterType.STRING, required=False)
                ]
                
                inferred_params_list = self.parameter_inference.extract_parameters_from_text(
                    user_intent, param_specs
                )
                
                for param in inferred_params_list:
                    inferred_params[param.name] = param.value
                
                # 验证参数
                if self.config.auto_validation:
                    validation_results = self.parameter_inference.validate_parameters(
                        inferred_params, param_specs
                    )
                    
                    # 检查验证结果
                    invalid_params = [name for name, result in validation_results.items() if not result.is_valid]
                    if invalid_params and self.config.strict_mode:
                        return {
                            'status': 'validation_failed',
                            'message': f'参数验证失败: {invalid_params}',
                            'validation_errors': {name: result.errors for name, result in validation_results.items() if not result.is_valid}
                        }
            
            # 3. 缓存检查
            cache_result = None
            is_cached = False
            
            if self.cache_manager:
                cache_result, is_cached = self.cache_manager.get_cached_result(
                    selected_tool.name, inferred_params
                )
                
                if is_cached:
                    self.system_stats['cache_hits'] += 1
                    logger.info("使用缓存结果")
                    return {
                        'status': 'success',
                        'result': cache_result,
                        'from_cache': True,
                        'tool_used': selected_tool.name,
                        'execution_time': (datetime.now() - start_time).total_seconds()
                    }
            
            # 4. 执行工具调用
            tool_call = ToolCall(
                tool_id=f"call_{datetime.now().timestamp()}",
                tool_name=selected_tool.name,
                parameters=inferred_params,
                timeout=self.config.default_timeout,
                max_retries=self.config.max_retries
            )
            
            execution_result = await self.concurrent_executor.execute_single_tool(tool_call)
            
            if execution_result.status == ResultStatus.SUCCESS:
                self.system_stats['successful_calls'] += 1
                
                # 缓存结果
                if self.cache_manager:
                    self.cache_manager.cache_result(
                        selected_tool.name, inferred_params, execution_result.result
                    )
                
                return {
                    'status': 'success',
                    'result': execution_result.result,
                    'from_cache': False,
                    'tool_used': selected_tool.name,
                    'execution_time': execution_result.execution_time,
                    'confidence': execution_result.confidence_score
                }
            else:
                self.system_stats['failed_calls'] += 1
                return {
                    'status': 'failure',
                    'error': str(execution_result.error),
                    'tool_used': selected_tool.name,
                    'execution_time': execution_result.execution_time
                }
        
        except Exception as e:
            self.system_stats['failed_calls'] += 1
            logger.error(f"智能工具调用失败: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def execute_tool_chain(
        self,
        chain_template: ChainTemplate,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ChainExecution:
        """执行工具链"""
        try:
            self.system_stats['chains_executed'] += 1
            
            execution = await self.chain_orchestrator.execute_chain(
                chain_template.template_id, input_data, context
            )
            
            return execution
        
        except Exception as e:
            logger.error(f"工具链执行失败: {str(e)}")
            raise
    
    def get_tool_recommendations(
        self,
        user_intent: str,
        context: Optional[Dict[str, Any]] = None,
        count: int = 3
    ) -> List[ToolCapability]:
        """获取工具推荐"""
        context_info = ContextInfo(
            user_intent=user_intent,
            conversation_history=context.get('conversation_history', []) if context else [],
            current_task=context.get('current_task') if context else None,
            available_tools=list(self.tool_selector.tool_registry.values()),
            user_preferences=context.get('user_preferences', {}) if context else {},
            system_state=context or {}
        )
        
        return self.tool_selector.get_tool_recommendations(context_info, count)
    
    def suggest_parameters(
        self,
        tool_name: str,
        user_intent: str
    ) -> List[InferredParameter]:
        """建议参数"""
        if tool_name not in self.tool_selector.tool_registry:
            return []
        
        tool = self.tool_selector.tool_registry[tool_name]
        
        # 创建参数规范
        param_specs = []
        for input_type in tool.input_types:
            param_specs.append(ParameterSpec(
                name=input_type,
                type=ParameterType.STRING,  # 简化处理
                required=False
            ))
        
        return self.parameter_inference.extract_parameters_from_text(user_intent, param_specs)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'system_info': {
                'name': 'Intelligent Tool System',
                'version': '1.0.0',
                'start_time': self.system_stats['start_time'].isoformat(),
                'uptime': (datetime.now() - self.system_stats['start_time']).total_seconds()
            },
            'statistics': self.system_stats.copy(),
            'registered_tools': len(self.tool_selector.tool_registry),
            'active_executions': len(self.concurrent_executor.active_executions),
            'cache_enabled': self.cache_manager is not None,
            'chain_orchestration_enabled': self.config.chain_orchestration_enabled
        }
        
        # 添加缓存统计
        if self.cache_manager:
            status['cache_stats'] = self.cache_manager.get_performance_report()
        
        # 添加性能统计
        status['performance_stats'] = self.concurrent_executor.get_performance_metrics()
        
        # 添加链统计
        status['chain_stats'] = self.chain_orchestrator.get_execution_statistics()
        
        return status
    
    def optimize_system(self):
        """系统优化"""
        logger.info("开始系统优化")
        
        # 优化缓存
        if self.cache_manager:
            self.cache_manager.optimize_cache()
        
        # 清理过期的执行记录
        # (这里可以添加更多优化逻辑)
        
        logger.info("系统优化完成")
    
    def export_system_config(self) -> Dict[str, Any]:
        """导出系统配置"""
        return {
            'config': {
                'tool_selection_enabled': self.config.tool_selection_enabled,
                'max_tool_suggestions': self.config.max_tool_suggestions,
                'parameter_inference_enabled': self.config.parameter_inference_enabled,
                'cache_enabled': self.config.cache_enabled,
                'chain_orchestration_enabled': self.config.chain_orchestration_enabled
            },
            'tool_registry': self.tool_selector.export_tool_registry(),
            'system_stats': self.system_stats
        }
    
    def import_system_config(self, config_data: Dict[str, Any]):
        """导入系统配置"""
        if 'tool_registry' in config_data:
            self.tool_selector.import_tool_registry(config_data['tool_registry'])
        
        if 'system_stats' in config_data:
            self.system_stats.update(config_data['system_stats'])
        
        logger.info("系统配置导入完成")
    
    async def shutdown(self):
        """关闭系统"""
        logger.info("正在关闭智能工具系统...")
        
        # 取消所有活跃的执行
        for execution_id in list(self.concurrent_executor.active_executions.keys()):
            self.concurrent_executor.cancel_execution(execution_id)
        
        # 等待活跃执行完成
        while self.concurrent_executor.active_executions:
            await asyncio.sleep(0.1)
        
        # 导出最终统计
        final_stats = self.get_system_status()
        
        logger.info(f"智能工具系统已关闭. 最终统计: {final_stats['statistics']}")


# 全局系统实例
_intelligent_tool_system: Optional[IntelligentToolSystem] = None


def get_intelligent_tool_system(config: Optional[IntelligentToolConfig] = None) -> IntelligentToolSystem:
    """获取全局智能工具系统实例"""
    global _intelligent_tool_system
    
    if _intelligent_tool_system is None:
        _intelligent_tool_system = IntelligentToolSystem(config)
    
    return _intelligent_tool_system


async def execute_smart_tool_call(
    user_intent: str,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[IntelligentToolConfig] = None
) -> Dict[str, Any]:
    """便捷的智能工具调用接口"""
    system = get_intelligent_tool_system(config)
    return await system.execute_intelligent_tool_call(user_intent, context)


# 示例使用
async def main():
    """主函数示例"""
    # 创建系统实例
    system = get_intelligent_tool_system()
    
    # 执行智能工具调用
    result = await execute_smart_tool_call(
        user_intent="计算列表 [1, 2, 3, 4, 5] 的总和",
        context={
            'conversation_history': [],
            'user_preferences': {}
        }
    )
    
    print("执行结果:", json.dumps(result, indent=2, ensure_ascii=False))
    
    # 获取系统状态
    status = system.get_system_status()
    print("系统状态:", json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())