"""
任务分解器

使用LLM将复杂任务智能分解为可执行的子任务
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio

from .models import Task, TaskPriority, TaskDependency, TaskExecutionPlan
from ..llm.interfaces.llm_interface import LLMInterface


class TaskDecomposer:
    """LLM驱动的任务分解器"""
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.decomposition_templates = {
            "research": self._get_research_template(),
            "development": self._get_development_template(),
            "analysis": self._get_analysis_template(),
            "creative": self._get_creative_template(),
            "generic": self._get_generic_template()
        }
    
    async def decompose_task(
        self,
        task: Task,
        max_depth: int = 3,
        min_task_duration: int = 5,  # 最小任务时长（分钟）
        max_subtasks: int = 10
    ) -> TaskExecutionPlan:
        """
        分解复杂任务
        
        Args:
            task: 原始任务
            max_depth: 最大分解深度
            min_task_duration: 最小任务时长（分钟）
            max_subtasks: 最大子任务数量
            
        Returns:
            任务执行计划
        """
        # 创建执行计划
        plan = TaskExecutionPlan(
            name=f"{task.name} - 执行计划",
            description=f"将任务 '{task.name}' 分解为可执行的子任务",
            tasks=[task]
        )
        
        # 如果任务足够简单，不需要分解
        if self._is_simple_task(task):
            task.status = "ready"
            return plan
        
        # 递归分解任务
        await self._decompose_recursive(task, plan, 0, max_depth, min_task_duration, max_subtasks)
        
        # 分析依赖关系
        await self._analyze_dependencies(plan)
        
        return plan
    
    async def _decompose_recursive(
        self,
        task: Task,
        plan: TaskExecutionPlan,
        current_depth: int,
        max_depth: int,
        min_duration: int,
        max_subtasks: int
    ):
        """递归分解任务"""
        if current_depth >= max_depth:
            task.status = "ready"
            return
        
        # 获取分解提示
        prompt = self._build_decomposition_prompt(task, current_depth, min_duration, max_subtasks)
        
        try:
            # 调用LLM进行任务分解
            response = await self.llm.generate_response(prompt)
            subtasks_data = self._parse_decomposition_response(response)
            
            if not subtasks_data:
                task.status = "ready"
                return
            
            # 创建子任务
            subtasks = []
            for i, subtask_data in enumerate(subtasks_data):
                subtask = self._create_subtask(task, subtask_data, i)
                subtasks.append(subtask)
                plan.add_task(subtask)
            
            # 更新原任务状态
            task.status = "planning"
            task.progress = 0.0
            
            # 递归分解子任务
            for subtask in subtasks:
                await self._decompose_recursive(
                    subtask, plan, current_depth + 1, 
                    max_depth, min_duration, max_subtasks
                )
            
        except Exception as e:
            print(f"任务分解失败: {e}")
            task.status = "ready"  # 分解失败时保持原任务可执行
    
    def _build_decomposition_prompt(
        self,
        task: Task,
        depth: int,
        min_duration: int,
        max_subtasks: int
    ) -> str:
        """构建任务分解提示"""
        template = self.decomposition_templates.get(task.task_type, self.decomposition_templates["generic"])
        
        prompt = f"""
{template}

任务信息:
- 名称: {task.name}
- 描述: {task.description}
- 类型: {task.task_type}
- 当前深度: {depth}
- 任务参数: {json.dumps(task.parameters, ensure_ascii=False, indent=2)}

分解要求:
1. 将任务分解为 {max_subtasks} 个以内的子任务
2. 每个子任务预计时长至少 {min_duration} 分钟
3. 子任务应该具体、可执行、有明确的输出
4. 考虑任务之间的依赖关系

请以JSON格式返回分解结果，格式如下:
{{
    "subtasks": [
        {{
            "name": "子任务名称",
            "description": "详细描述",
            "priority": "high/normal/low",
            "estimated_duration": 预计分钟数,
            "task_type": "子任务类型",
            "parameters": {{"参数": "值"}},
            "dependencies": ["依赖的子任务名称列表"]
        }}
    ],
    "reasoning": "分解思路说明"
}}
"""
        return prompt
    
    def _parse_decomposition_response(self, response: str) -> List[Dict[str, Any]]:
        """解析LLM返回的分解结果"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return []
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            subtasks = data.get("subtasks", [])
            return subtasks
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析分解结果失败: {e}")
            return []
    
    def _create_subtask(self, parent_task: Task, subtask_data: Dict[str, Any], index: int) -> Task:
        """创建子任务"""
        subtask = Task(
            name=subtask_data.get("name", f"{parent_task.name} - 子任务 {index + 1}"),
            description=subtask_data.get("description", ""),
            task_type=subtask_data.get("task_type", parent_task.task_type),
            priority=TaskPriority(subtask_data.get("priority", "normal")),
            estimated_duration=subtask_data.get("estimated_duration"),
            parameters=subtask_data.get("parameters", {}),
            inputs={"parent_task_id": parent_task.id},
            metadata={
                "parent_task_id": parent_task.id,
                "decomposition_depth": parent_task.metadata.get("decomposition_depth", 0) + 1
            }
        )
        
        # 添加标签
        subtask.tags = parent_task.tags.copy()
        subtask.tags.append(f"subtask_{index + 1}")
        
        return subtask
    
    def _is_simple_task(self, task: Task) -> bool:
        """判断任务是否足够简单，不需要分解"""
        # 基于任务描述和参数判断复杂度
        complexity_indicators = [
            len(task.description) < 100,  # 描述较短
            len(task.parameters) <= 2,    # 参数较少
            task.estimated_duration and task.estimated_duration <= 30,  # 预计时间短
            "simple" in task.name.lower() or "basic" in task.name.lower()
        ]
        
        return sum(complexity_indicators) >= 3
    
    async def _analyze_dependencies(self, plan: TaskExecutionPlan):
        """分析任务依赖关系"""
        # 基于任务描述和类型推断依赖关系
        for task in plan.tasks:
            if task.metadata.get("parent_task_id"):
                parent_id = task.metadata["parent_task_id"]
                dependency = TaskDependency(
                    source_task_id=parent_id,
                    target_task_id=task.id,
                    dependency_type="finish_to_start"
                )
                plan.add_dependency(dependency)
    
    def _get_research_template(self) -> str:
        """研究任务分解模板"""
        return """
你是一个专业的任务分解专家。请将研究类任务分解为以下阶段:

1. 信息收集阶段
   - 确定研究范围和目标
   - 收集相关资料和数据源
   - 整理已有信息

2. 分析阶段
   - 深度分析收集的信息
   - 识别关键问题和趋势
   - 进行对比和评估

3. 总结阶段
   - 整理分析结果
   - 形成结论和建议
   - 准备输出报告

每个子任务应该:
- 有明确的研究目标
- 指定具体的信息来源或分析方法
- 有可衡量的输出结果
"""
    
    def _get_development_template(self) -> str:
        """开发任务分解模板"""
        return """
你是一个专业的软件开发任务分解专家。请将开发任务分解为以下阶段:

1. 需求分析阶段
   - 理解业务需求
   - 设计技术方案
   - 规划开发步骤

2. 实现阶段
   - 编写代码实现
   - 单元测试
   - 集成测试

3. 部署阶段
   - 代码审查
   - 部署准备
   - 上线验证

每个子任务应该:
- 有明确的输入和输出
- 指定具体的技术要求
- 有可验证的完成标准
"""
    
    def _get_analysis_template(self) -> str:
        """分析任务分解模板"""
        return """
你是一个专业的数据分析任务分解专家。请将分析任务分解为以下阶段:

1. 数据准备阶段
   - 收集和清洗数据
   - 数据质量检查
   - 数据预处理

2. 分析阶段
   - 探索性数据分析
   - 统计分析和建模
   - 结果验证

3. 报告阶段
   - 可视化结果
   - 撰写分析报告
   - 结论和建议

每个子任务应该:
- 指定具体的数据源或分析方法
- 有明确的分析目标
- 产生可量化的结果
"""
    
    def _get_creative_template(self) -> str:
        """创意任务分解模板"""
        return """
你是一个专业的创意任务分解专家。请将创意任务分解为以下阶段:

1. 构思阶段
   - 头脑风暴和创意收集
   - 概念设计和框架搭建
   - 创意评估和筛选

2. 制作阶段
   - 内容创作和实现
   - 质量检查和优化
   - 细节完善

3. 完善阶段
   - 最终审查
   - 格式化和整理
   - 交付准备

每个子任务应该:
- 有明确的创意目标
- 指定具体的创作方法
- 有可评估的创意质量
"""
    
    def _get_generic_template(self) -> str:
        """通用任务分解模板"""
        return """
你是一个专业的任务分解专家。请将任何复杂任务分解为以下原则:

分解原则:
1. 每个子任务应该具体、可执行
2. 考虑任务的逻辑顺序和依赖关系
3. 确保子任务的粒度适中（既不过于粗糙，也不过于细致）
4. 为每个子任务提供清晰的输入和输出定义

分解步骤:
1. 理解任务的最终目标和期望输出
2. 识别关键的工作流程和里程碑
3. 将大任务分解为3-7个子任务
4. 确定子任务之间的依赖关系
5. 为每个子任务设定合理的预计时长

每个子任务应该包含:
- 清晰的任务名称和描述
- 具体的执行步骤
- 明确的完成标准
- 预期的输出结果
"""