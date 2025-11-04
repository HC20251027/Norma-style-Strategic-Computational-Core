"""
工具调用链的智能编排系统
实现复杂工作流的智能规划和执行
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import networkx as nx
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class ChainStatus(Enum):
    """链状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepType(Enum):
    """步骤类型枚举"""
    TOOL_CALL = "tool_call"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    MERGE = "merge"
    TRANSFORM = "transform"
    VALIDATION = "validation"


class ConditionType(Enum):
    """条件类型枚举"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    EXISTS = "exists"
    CUSTOM = "custom"


@dataclass
class ChainStep:
    """链步骤定义"""
    step_id: str
    name: str
    step_type: StepType
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 2
    on_failure: str = "continue"  # continue, stop, retry
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 并行执行相关
    parallel_group: Optional[str] = None
    merge_strategy: str = "first_success"  # first_success, all_success, majority
    
    # 循环相关
    loop_count: Optional[int] = None
    loop_condition: Optional[str] = None
    loop_variable: Optional[str] = None
    
    # 条件相关
    condition_expression: Optional[str] = None
    true_steps: List[str] = field(default_factory=list)
    false_steps: List[str] = field(default_factory=list)


@dataclass
class ChainExecution:
    """链执行状态"""
    chain_id: str
    status: ChainStatus
    current_step: Optional[str]
    completed_steps: Set[str]
    failed_steps: Set[str]
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_errors: Dict[str, Exception] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainTemplate:
    """链模板"""
    template_id: str
    name: str
    description: str
    steps: List[ChainStep]
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = "system"


class IntelligentChainOrchestrator:
    """智能链编排器"""
    
    def __init__(self):
        self.chains: Dict[str, ChainTemplate] = {}
        self.active_executions: Dict[str, ChainExecution] = {}
        self.step_registry: Dict[str, Callable] = {}
        self.execution_graphs: Dict[str, nx.DiGraph] = {}
        
        # 性能统计
        self.execution_stats = defaultdict(int)
        self.step_performance = defaultdict(list)
        
        # 初始化默认步骤类型
        self._register_default_steps()
        
        # 链模板库
        self.template_library = {}
    
    def _register_default_steps(self):
        """注册默认步骤类型"""
        # 这些是系统内置的步骤类型，实际的工具调用会通过外部注册
        pass
    
    def register_tool(self, tool_name: str, tool_function: Callable):
        """注册工具函数"""
        self.step_registry[tool_name] = tool_function
        logger.info(f"注册工具到编排器: {tool_name}")
    
    def create_chain_template(self, template: ChainTemplate) -> str:
        """创建链模板"""
        # 验证模板
        validation_result = self._validate_chain_template(template)
        if not validation_result['valid']:
            raise ValueError(f"链模板验证失败: {validation_result['errors']}")
        
        self.chains[template.template_id] = template
        self.template_library[template.template_id] = template
        
        # 构建执行图
        self._build_execution_graph(template.template_id)
        
        logger.info(f"创建链模板: {template.name} ({template.template_id})")
        return template.template_id
    
    def _validate_chain_template(self, template: ChainTemplate) -> Dict[str, Any]:
        """验证链模板"""
        errors = []
        warnings = []
        
        # 检查步骤ID唯一性
        step_ids = [step.step_id for step in template.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("步骤ID不唯一")
        
        # 检查依赖关系
        for step in template.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"步骤 {step.step_id} 依赖不存在的步骤: {dep}")
        
        # 检查循环依赖
        if self._has_circular_dependency(template.steps):
            errors.append("存在循环依赖")
        
        # 检查工具存在性
        for step in template.steps:
            if step.step_type == StepType.TOOL_CALL and step.tool_name:
                if step.tool_name not in self.step_registry:
                    warnings.append(f"步骤 {step.step_id} 使用的工具未注册: {step.tool_name}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _has_circular_dependency(self, steps: List[ChainStep]) -> bool:
        """检查循环依赖"""
        graph = nx.DiGraph()
        
        # 构建依赖图
        for step in steps:
            graph.add_node(step.step_id)
            for dep in step.dependencies:
                graph.add_edge(step.step_id, dep)
        
        try:
            nx.find_cycle(graph)
            return True
        except nx.NetworkXNoCycle:
            return False
    
    def _build_execution_graph(self, chain_id: str):
        """构建执行图"""
        template = self.chains[chain_id]
        graph = nx.DiGraph()
        
        # 添加节点
        for step in template.steps:
            graph.add_node(step.step_id, **step.__dict__)
        
        # 添加边 (依赖关系)
        for step in template.steps:
            for dep in step.dependencies:
                graph.add_edge(step.step_id, dep)
        
        self.execution_graphs[chain_id] = graph
    
    async def execute_chain(self, chain_id: str, input_data: Dict[str, Any], 
                          context: Optional[Dict[str, Any]] = None) -> ChainExecution:
        """执行链"""
        if chain_id not in self.chains:
            raise ValueError(f"链模板不存在: {chain_id}")
        
        # 创建执行实例
        execution_id = str(uuid.uuid4())
        execution = ChainExecution(
            chain_id=execution_id,
            status=ChainStatus.RUNNING,
            current_step=None,
            completed_steps=set(),
            failed_steps=set(),
            context=context or {}
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            logger.info(f"开始执行链: {chain_id}, 执行ID: {execution_id}")
            
            # 执行链
            result = await self._execute_chain_steps(chain_id, execution, input_data)
            
            execution.status = ChainStatus.COMPLETED
            execution.end_time = datetime.now()
            
            logger.info(f"链执行完成: {chain_id}, 耗时: {execution.end_time - execution.start_time}")
            
            return execution
            
        except Exception as e:
            execution.status = ChainStatus.FAILED
            execution.end_time = datetime.now()
            logger.error(f"链执行失败: {chain_id}, 错误: {str(e)}")
            raise
        
        finally:
            # 清理活跃执行
            self.active_executions.pop(execution_id, None)
    
    async def _execute_chain_steps(self, chain_id: str, execution: ChainExecution, input_data: Dict[str, Any]):
        """执行链步骤"""
        template = self.chains[chain_id]
        graph = self.execution_graphs[chain_id]
        
        # 获取拓扑排序
        try:
            execution_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # 如果有环，使用其他方法
            execution_order = self._get_fallback_execution_order(template.steps)
        
        # 执行每个步骤
        for step_id in execution_order:
            step = next(s for s in template.steps if s.step_id == step_id)
            
            # 检查依赖是否完成
            if not all(dep in execution.completed_steps for dep in step.dependencies):
                logger.warning(f"步骤 {step_id} 的依赖未完成，跳过")
                continue
            
            execution.current_step = step_id
            
            try:
                # 执行步骤
                step_result = await self._execute_step(step, execution, input_data)
                
                # 存储结果
                execution.step_results[step_id] = step_result
                execution.completed_steps.add(step_id)
                
                # 更新上下文
                execution.context[f"step_{step_id}_result"] = step_result
                
                logger.debug(f"步骤完成: {step_id}")
                
            except Exception as e:
                execution.step_errors[step_id] = e
                execution.failed_steps.add(step_id)
                
                # 根据失败策略处理
                if step.on_failure == "stop":
                    raise e
                elif step.on_failure == "retry" and step.retry_count < step.max_retries:
                    step.retry_count += 1
                    logger.warning(f"重试步骤: {step_id}")
                    # 重新执行此步骤
                    execution.current_step = step_id
                    step_result = await self._execute_step(step, execution, input_data)
                    execution.step_results[step_id] = step_result
                    execution.completed_steps.add(step_id)
                else:
                    logger.error(f"步骤失败: {step_id}, 错误: {str(e)}")
                    if step.on_failure == "stop":
                        raise e
    
    async def _execute_step(self, step: ChainStep, execution: ChainExecution, input_data: Dict[str, Any]) -> Any:
        """执行单个步骤"""
        start_time = datetime.now()
        
        try:
            if step.step_type == StepType.TOOL_CALL:
                return await self._execute_tool_call(step, execution, input_data)
            elif step.step_type == StepType.CONDITION:
                return await self._execute_condition(step, execution, input_data)
            elif step.step_type == StepType.LOOP:
                return await self._execute_loop(step, execution, input_data)
            elif step.step_type == StepType.PARALLEL:
                return await self._execute_parallel(step, execution, input_data)
            elif step.step_type == StepType.TRANSFORM:
                return await self._execute_transform(step, execution, input_data)
            elif step.step_type == StepType.VALIDATION:
                return await self._execute_validation(step, execution, input_data)
            else:
                raise ValueError(f"不支持的步骤类型: {step.step_type}")
        
        finally:
            # 记录性能统计
            execution_time = (datetime.now() - start_time).total_seconds()
            self.step_performance[step.step_id].append(execution_time)
    
    async def _execute_tool_call(self, step: ChainStep, execution: ChainExecution, input_data: Dict[str, Any]) -> Any:
        """执行工具调用步骤"""
        if step.tool_name not in self.step_registry:
            raise ValueError(f"工具未注册: {step.tool_name}")
        
        tool_function = self.step_registry[step.tool_name]
        
        # 准备参数
        parameters = self._prepare_parameters(step.parameters, execution, input_data)
        
        # 执行工具
        if asyncio.iscoroutinefunction(tool_function):
            result = await tool_function(**parameters)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: tool_function(**parameters))
        
        return result
    
    async def _execute_condition(self, step: ChainStep, execution: ChainExecution, input_data: Dict[str, Any]) -> bool:
        """执行条件步骤"""
        # 简单的条件评估
        if step.condition_expression:
            # 这里应该实现表达式求值引擎
            # 暂时返回True
            return True
        
        # 评估条件列表
        for condition in step.conditions:
            if not self._evaluate_condition(condition, execution, input_data):
                return False
        
        return True
    
    async def _execute_loop(self, step: ChainStep, execution: ChainExecution, input_data: Dict[str, Any]) -> List[Any]:
        """执行循环步骤"""
        results = []
        
        if step.loop_count:
            # 固定次数循环
            for i in range(step.loop_count):
                loop_context = execution.context.copy()
                if step.loop_variable:
                    loop_context[step.loop_variable] = i
                
                # 执行循环体 (这里需要递归执行子步骤)
                # 简化实现
                results.append(f"loop_iteration_{i}")
        
        elif step.loop_condition:
            # 条件循环
            iteration = 0
            while iteration < 100:  # 防止无限循环
                if not self._evaluate_loop_condition(step.loop_condition, execution, input_data):
                    break
                
                loop_context = execution.context.copy()
                if step.loop_variable:
                    loop_context[step.loop_variable] = iteration
                
                results.append(f"loop_iteration_{iteration}")
                iteration += 1
        
        return results
    
    async def _execute_parallel(self, step: ChainStep, execution: ChainExecution, input_data: Dict[str, Any]) -> List[Any]:
        """执行并行步骤"""
        # 这里应该执行并行组的步骤
        # 简化实现
        return ["parallel_result_1", "parallel_result_2"]
    
    async def _execute_transform(self, step: ChainStep, execution: ChainExecution, input_data: Dict[str, Any]) -> Any:
        """执行转换步骤"""
        # 数据转换逻辑
        source_data = execution.context.get(step.parameters.get('source', 'input'))
        transform_type = step.parameters.get('type', 'identity')
        
        if transform_type == 'identity':
            return source_data
        elif transform_type == 'filter':
            filter_condition = step.parameters.get('condition', '')
            # 简化过滤逻辑
            return source_data
        elif transform_type == 'map':
            map_function = step.parameters.get('function', '')
            # 简化映射逻辑
            return source_data
        else:
            return source_data
    
    async def _execute_validation(self, step: ChainStep, execution: ChainExecution, input_data: Dict[str, Any]) -> bool:
        """执行验证步骤"""
        # 验证逻辑
        validation_rules = step.parameters.get('rules', [])
        
        for rule in validation_rules:
            if not self._apply_validation_rule(rule, execution, input_data):
                raise ValueError(f"验证失败: {rule}")
        
        return True
    
    def _prepare_parameters(self, parameters: Dict[str, Any], execution: ChainExecution, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备执行参数"""
        prepared = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # 参数引用
                var_name = value[2:-1]
                if var_name in execution.context:
                    prepared[key] = execution.context[var_name]
                elif var_name in input_data:
                    prepared[key] = input_data[var_name]
                else:
                    prepared[key] = value  # 保持原值
            else:
                prepared[key] = value
        
        return prepared
    
    def _evaluate_condition(self, condition: Dict[str, Any], execution: ChainExecution, input_data: Dict[str, Any]) -> bool:
        """评估条件"""
        condition_type = condition.get('type', ConditionType.EQUALS.value)
        left_value = condition.get('left')
        right_value = condition.get('right')
        
        # 解析变量引用
        if isinstance(left_value, str) and left_value.startswith('${'):
            var_name = left_value[2:-1]
            left_value = execution.context.get(var_name, input_data.get(var_name, left_value))
        
        if isinstance(right_value, str) and right_value.startswith('${'):
            var_name = right_value[2:-1]
            right_value = execution.context.get(var_name, input_data.get(var_name, right_value))
        
        # 评估条件
        if condition_type == ConditionType.EQUALS.value:
            return left_value == right_value
        elif condition_type == ConditionType.NOT_EQUALS.value:
            return left_value != right_value
        elif condition_type == ConditionType.GREATER_THAN.value:
            return left_value > right_value
        elif condition_type == ConditionType.LESS_THAN.value:
            return left_value < right_value
        elif condition_type == ConditionType.CONTAINS.value:
            return str(left_value) in str(right_value)
        elif condition_type == ConditionType.EXISTS.value:
            return left_value is not None
        
        return False
    
    def _evaluate_loop_condition(self, condition: str, execution: ChainExecution, input_data: Dict[str, Any]) -> bool:
        """评估循环条件"""
        # 简化实现
        return True
    
    def _apply_validation_rule(self, rule: Dict[str, Any], execution: ChainExecution, input_data: Dict[str, Any]) -> bool:
        """应用验证规则"""
        # 简化实现
        return True
    
    def _get_fallback_execution_order(self, steps: List[ChainStep]) -> List[str]:
        """获取后备执行顺序"""
        # 按依赖数量排序
        return sorted([step.step_id for step in steps], 
                     key=lambda step_id: len([s for s in steps if step_id in s.dependencies]))
    
    def get_chain_status(self, execution_id: str) -> Optional[ChainExecution]:
        """获取链执行状态"""
        return self.active_executions.get(execution_id)
    
    def cancel_chain_execution(self, execution_id: str) -> bool:
        """取消链执行"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = ChainStatus.CANCELLED
            execution.end_time = datetime.now()
            logger.info(f"取消链执行: {execution_id}")
            return True
        return False
    
    def pause_chain_execution(self, execution_id: str) -> bool:
        """暂停链执行"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = ChainStatus.PAUSED
            logger.info(f"暂停链执行: {execution_id}")
            return True
        return False
    
    def resume_chain_execution(self, execution_id: str) -> bool:
        """恢复链执行"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            if execution.status == ChainStatus.PAUSED:
                execution.status = ChainStatus.RUNNING
                logger.info(f"恢复链执行: {execution_id}")
                return True
        return False
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        total_executions = sum(self.execution_stats.values())
        
        return {
            'total_executions': total_executions,
            'successful_executions': self.execution_stats.get('successful', 0),
            'failed_executions': self.execution_stats.get('failed', 0),
            'cancelled_executions': self.execution_stats.get('cancelled', 0),
            'active_executions': len(self.active_executions),
            'registered_chains': len(self.chains),
            'registered_tools': len(self.step_registry)
        }
    
    def export_chain_template(self, chain_id: str) -> Dict[str, Any]:
        """导出链模板"""
        if chain_id not in self.chains:
            raise ValueError(f"链模板不存在: {chain_id}")
        
        template = self.chains[chain_id]
        
        return {
            'template_id': template.template_id,
            'name': template.name,
            'description': template.description,
            'steps': [step.__dict__ for step in template.steps],
            'input_schema': template.input_schema,
            'output_schema': template.output_schema,
            'tags': template.tags,
            'version': template.version
        }
    
    def import_chain_template(self, template_data: Dict[str, Any]) -> str:
        """导入链模板"""
        # 重建步骤对象
        steps = []
        for step_data in template_data['steps']:
            step = ChainStep(
                step_id=step_data['step_id'],
                name=step_data['name'],
                step_type=StepType(step_data['step_type']),
                tool_name=step_data.get('tool_name'),
                parameters=step_data.get('parameters', {}),
                conditions=step_data.get('conditions', []),
                dependencies=step_data.get('dependencies', []),
                timeout=step_data.get('timeout'),
                retry_count=step_data.get('retry_count', 0),
                max_retries=step_data.get('max_retries', 2),
                on_failure=step_data.get('on_failure', 'continue'),
                metadata=step_data.get('metadata', {}),
                parallel_group=step_data.get('parallel_group'),
                merge_strategy=step_data.get('merge_strategy', 'first_success'),
                loop_count=step_data.get('loop_count'),
                loop_condition=step_data.get('loop_condition'),
                loop_variable=step_data.get('loop_variable'),
                condition_expression=step_data.get('condition_expression'),
                true_steps=step_data.get('true_steps', []),
                false_steps=step_data.get('false_steps', [])
            )
            steps.append(step)
        
        template = ChainTemplate(
            template_id=template_data['template_id'],
            name=template_data['name'],
            description=template_data['description'],
            steps=steps,
            input_schema=template_data.get('input_schema', {}),
            output_schema=template_data.get('output_schema', {}),
            tags=template_data.get('tags', []),
            version=template_data.get('version', '1.0')
        )
        
        return self.create_chain_template(template)