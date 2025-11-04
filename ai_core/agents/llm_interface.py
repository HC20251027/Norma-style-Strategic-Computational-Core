"""
LLM接口抽象层

负责与LLM进行交互，实现：
1. 工具描述转换为LLM可理解的格式
2. 智能工具选择
3. 参数推断
4. 执行计划生成
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import asyncio
import time

from .models import (
    ToolSelectionResult, 
    ParameterInferenceResult, 
    ExecutionPlan,
    ToolcallContext,
    ToolcallRequest
)


logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """LLM接口抽象类"""
    
    @abstractmethod
    async def select_tools(
        self, 
        query: str, 
        available_tools: List[Dict[str, Any]], 
        context: ToolcallContext,
        max_tools: Optional[int] = None
    ) -> ToolSelectionResult:
        """选择合适的工具"""
        pass
    
    @abstractmethod
    async def infer_parameters(
        self, 
        query: str, 
        selected_tools: List[str], 
        tool_definitions: List[Dict[str, Any]], 
        context: ToolcallContext
    ) -> ParameterInferenceResult:
        """推断工具参数"""
        pass
    
    @abstractmethod
    async def generate_execution_plan(
        self, 
        query: str, 
        selected_tools: List[str], 
        inferred_parameters: Dict[str, Any], 
        context: ToolcallContext
    ) -> ExecutionPlan:
        """生成执行计划"""
        pass
    
    @abstractmethod
    async def validate_plan(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> Tuple[bool, Optional[str]]:
        """验证执行计划"""
        pass


class OpenAIInterface(LLMInterface):
    """OpenAI接口实现"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.client = None
    
    async def select_tools(
        self, 
        query: str, 
        available_tools: List[Dict[str, Any]], 
        context: ToolcallContext,
        max_tools: Optional[int] = None
    ) -> ToolSelectionResult:
        """选择合适的工具"""
        try:
            # 构建工具描述
            tools_description = self._format_tools_for_llm(available_tools)
            
            # 构建提示词
            prompt = self._build_tool_selection_prompt(
                query, tools_description, max_tools
            )
            
            # 调用LLM
            response = await self._call_llm(prompt)
            
            # 解析响应
            result = self._parse_tool_selection_response(response, available_tools)
            
            logger.info(f"工具选择完成: {result.selected_tools}")
            return result
            
        except Exception as e:
            logger.error(f"工具选择失败: {e}")
            return ToolSelectionResult(
                selected_tools=[],
                confidence_scores={},
                reasoning=f"工具选择失败: {str(e)}",
                requires_clarification=True,
                clarification_questions=["无法理解您的需求，请提供更详细的信息"]
            )
    
    async def infer_parameters(
        self, 
        query: str, 
        selected_tools: List[str], 
        tool_definitions: List[Dict[str, Any]], 
        context: ToolcallContext
    ) -> ParameterInferenceResult:
        """推断工具参数"""
        try:
            # 过滤出选中工具的定义
            selected_tool_defs = [
                tool for tool in tool_definitions 
                if tool["id"] in selected_tools
            ]
            
            # 构建参数推断提示词
            prompt = self._build_parameter_inference_prompt(
                query, selected_tool_defs, context
            )
            
            # 调用LLM
            response = await self._call_llm(prompt)
            
            # 解析响应
            result = self._parse_parameter_inference_response(response, selected_tools)
            
            logger.info(f"参数推断完成: {result.inferred_parameters}")
            return result
            
        except Exception as e:
            logger.error(f"参数推断失败: {e}")
            return ParameterInferenceResult(
                inferred_parameters={},
                confidence_scores={},
                reasoning=f"参数推断失败: {str(e)}",
                missing_parameters={tool_id: ["参数推断失败"] for tool_id in selected_tools}
            )
    
    async def generate_execution_plan(
        self, 
        query: str, 
        selected_tools: List[str], 
        inferred_parameters: Dict[str, Any], 
        context: ToolcallContext
    ) -> ExecutionPlan:
        """生成执行计划"""
        try:
            # 构建执行计划提示词
            prompt = self._build_execution_plan_prompt(
                query, selected_tools, inferred_parameters
            )
            
            # 调用LLM
            response = await self._call_llm(prompt)
            
            # 解析响应
            result = self._parse_execution_plan_response(response, selected_tools)
            
            logger.info(f"执行计划生成完成: {result.tool_sequence}")
            return result
            
        except Exception as e:
            logger.error(f"执行计划生成失败: {e}")
            return ExecutionPlan(
                tool_sequence=selected_tools,  # 默认顺序执行
                dependencies={},
                parallel_groups=[[tool] for tool in selected_tools],
                estimated_time=len(selected_tools) * 5.0  # 默认每个工具5秒
            )
    
    async def validate_plan(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> Tuple[bool, Optional[str]]:
        """验证执行计划"""
        try:
            # 基本验证
            if not plan.tool_sequence:
                return False, "执行计划为空"
            
            # 检查依赖关系是否合理
            for tool_id, dependencies in plan.dependencies.items():
                for dep in dependencies:
                    if dep not in plan.tool_sequence:
                        return False, f"工具 {tool_id} 的依赖 {dep} 不在执行序列中"
            
            # 构建验证提示词
            prompt = self._build_plan_validation_prompt(plan, context)
            
            # 调用LLM进行高级验证
            response = await self._call_llm(prompt)
            
            # 解析验证结果
            is_valid, error_msg = self._parse_validation_response(response)
            
            return is_valid, error_msg
            
        except Exception as e:
            logger.error(f"执行计划验证失败: {e}")
            return False, f"验证过程出错: {str(e)}"
    
    def _format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> str:
        """将工具列表格式化为LLM可理解的格式"""
        formatted = []
        for tool in tools:
            tool_info = {
                "id": tool["id"],
                "name": tool["name"],
                "description": tool["description"],
                "category": tool.get("category", "unknown"),
                "parameters": [
                    {
                        "name": param["name"],
                        "type": param["type"],
                        "description": param["description"],
                        "required": param.get("required", True)
                    }
                    for param in tool.get("parameters", [])
                ]
            }
            formatted.append(tool_info)
        
        return json.dumps(formatted, ensure_ascii=False, indent=2)
    
    def _build_tool_selection_prompt(
        self, 
        query: str, 
        tools_description: str, 
        max_tools: Optional[int]
    ) -> str:
        """构建工具选择提示词"""
        max_tools_text = f" (最多选择 {max_tools} 个工具)" if max_tools else ""
        
        prompt = f"""
你是一个专业的工具选择助手。用户提出了以下需求：

用户需求：{query}

可用的工具列表：
{tools_description}

请根据用户需求，从可用工具中选择最合适的工具{max_tools_text}。

选择原则：
1. 优先选择能够直接解决用户问题的工具
2. 考虑工具的功能匹配度和准确性
3. 避免选择功能重复的工具
4. 如果需要多个工具，考虑它们之间的协作关系

请以JSON格式返回结果，包含以下字段：
{{
    "selected_tools": ["工具ID1", "工具ID2"],
    "confidence_scores": {{"工具ID1": 0.9, "工具ID2": 0.8}},
    "reasoning": "选择理由说明",
    "alternatives": ["备选工具ID1", "备选工具ID2"],
    "requires_clarification": false,
    "clarification_questions": []
}}

如果用户需求不够明确，需要澄清，请将 requires_clarification 设为 true，并提供具体的问题。
"""
        return prompt
    
    def _build_parameter_inference_prompt(
        self, 
        query: str, 
        tool_definitions: List[Dict[str, Any]], 
        context: ToolcallContext
    ) -> str:
        """构建参数推断提示词"""
        tools_info = json.dumps(tool_definitions, ensure_ascii=False, indent=2)
        
        prompt = f"""
你是一个专业的参数推断助手。根据用户需求和工具定义，推断每个工具需要的最优参数。

用户需求：{query}

选中的工具定义：
{tools_info}

上下文信息：
{json.dumps(context.to_dict(), ensure_ascii=False, indent=2)}

请为每个工具推断合适的参数值。推断原则：
1. 基于用户需求和工具功能进行合理推断
2. 优先使用默认值和常见值
3. 对于不确定的参数，标记为缺失
4. 考虑参数之间的依赖关系

请以JSON格式返回结果：
{{
    "inferred_parameters": {{
        "工具ID1": {{"参数名": "参数值"}},
        "工具ID2": {{"参数名": "参数值"}}
    }},
    "confidence_scores": {{"工具ID1": 0.9, "工具ID2": 0.8}},
    "reasoning": "推断理由说明",
    "missing_parameters": {{"工具ID1": ["缺失参数列表"]}},
    "ambiguous_parameters": {{"工具ID1": ["模糊参数列表"]}}
}}
"""
        return prompt
    
    def _build_execution_plan_prompt(
        self, 
        query: str, 
        selected_tools: List[str], 
        inferred_parameters: Dict[str, Any]
    ) -> str:
        """构建执行计划提示词"""
        tools_info = json.dumps({
            "selected_tools": selected_tools,
            "inferred_parameters": inferred_parameters
        }, ensure_ascii=False, indent=2)
        
        prompt = f"""
你是一个专业的执行计划制定助手。根据选中的工具和推断的参数，制定最优的执行计划。

用户需求：{query}

工具和参数信息：
{tools_info}

请制定执行计划，考虑以下因素：
1. 工具之间的依赖关系
2. 并行执行的可能性
3. 执行效率和资源利用
4. 错误处理和备用方案

请以JSON格式返回结果：
{{
    "tool_sequence": ["工具ID1", "工具ID2", "工具ID3"],
    "dependencies": {{"工具ID2": ["工具ID1"], "工具ID3": ["工具ID2"]}},
    "parallel_groups": [["工具ID1", "工具ID4"], ["工具ID2"], ["工具ID3"]],
    "estimated_time": 15.5,
    "resource_requirements": {{"memory": "2GB", "cpu": "50%"}},
    "fallback_strategies": [
        {{"trigger": "工具ID2失败", "action": "使用工具ID5替代"}}
    ]
}}
"""
        return prompt
    
    def _build_plan_validation_prompt(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> str:
        """构建计划验证提示词"""
        plan_info = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
        
        prompt = f"""
你是一个专业的执行计划验证助手。请验证以下执行计划的合理性：

执行计划：
{plan_info}

上下文信息：
{json.dumps(context.to_dict(), ensure_ascii=False, indent=2)}

请检查：
1. 依赖关系是否合理
2. 并行分组是否高效
3. 资源需求是否合理
4. 备用策略是否完善
5. 是否存在死锁或循环依赖

请以JSON格式返回结果：
{{
    "is_valid": true,
    "error_message": null,
    "warnings": ["警告信息列表"],
    "suggestions": ["改进建议列表"]
}}
"""
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        try:
            import httpx
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的AI助手，擅长工具选择、参数推断和执行计划制定。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"API调用失败: {response.status_code}")
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            raise
    
    def _parse_tool_selection_response(
        self, 
        response: str, 
        available_tools: List[Dict[str, Any]]
    ) -> ToolSelectionResult:
        """解析工具选择响应"""
        try:
            # 尝试解析JSON
            data = json.loads(response)
            
            selected_tools = data.get("selected_tools", [])
            confidence_scores = data.get("confidence_scores", {})
            reasoning = data.get("reasoning", "")
            alternatives = data.get("alternatives", [])
            requires_clarification = data.get("requires_clarification", False)
            clarification_questions = data.get("clarification_questions", [])
            
            # 验证工具ID是否有效
            valid_tools = []
            tool_ids = {tool["id"] for tool in available_tools}
            
            for tool_id in selected_tools:
                if tool_id in tool_ids:
                    valid_tools.append(tool_id)
                else:
                    logger.warning(f"无效的工具ID: {tool_id}")
            
            return ToolSelectionResult(
                selected_tools=valid_tools,
                confidence_scores=confidence_scores,
                reasoning=reasoning,
                alternatives=alternatives,
                requires_clarification=requires_clarification,
                clarification_questions=clarification_questions
            )
            
        except json.JSONDecodeError:
            logger.error(f"无法解析LLM响应: {response}")
            return ToolSelectionResult(
                selected_tools=[],
                confidence_scores={},
                reasoning="响应解析失败",
                requires_clarification=True,
                clarification_questions=["无法理解系统响应，请重试"]
            )
    
    def _parse_parameter_inference_response(
        self, 
        response: str, 
        selected_tools: List[str]
    ) -> ParameterInferenceResult:
        """解析参数推断响应"""
        try:
            data = json.loads(response)
            
            inferred_parameters = data.get("inferred_parameters", {})
            confidence_scores = data.get("confidence_scores", {})
            reasoning = data.get("reasoning", "")
            missing_parameters = data.get("missing_parameters", {})
            ambiguous_parameters = data.get("ambiguous_parameters", {})
            
            return ParameterInferenceResult(
                inferred_parameters=inferred_parameters,
                confidence_scores=confidence_scores,
                reasoning=reasoning,
                missing_parameters=missing_parameters,
                ambiguous_parameters=ambiguous_parameters
            )
            
        except json.JSONDecodeError:
            logger.error(f"无法解析参数推断响应: {response}")
            return ParameterInferenceResult(
                inferred_parameters={},
                confidence_scores={},
                reasoning="响应解析失败",
                missing_parameters={tool_id: ["解析失败"] for tool_id in selected_tools}
            )
    
    def _parse_execution_plan_response(
        self, 
        response: str, 
        selected_tools: List[str]
    ) -> ExecutionPlan:
        """解析执行计划响应"""
        try:
            data = json.loads(response)
            
            tool_sequence = data.get("tool_sequence", selected_tools)
            dependencies = data.get("dependencies", {})
            parallel_groups = data.get("parallel_groups", [])
            estimated_time = data.get("estimated_time", 0.0)
            resource_requirements = data.get("resource_requirements", {})
            fallback_strategies = data.get("fallback_strategies", [])
            
            return ExecutionPlan(
                tool_sequence=tool_sequence,
                dependencies=dependencies,
                parallel_groups=parallel_groups,
                estimated_time=estimated_time,
                resource_requirements=resource_requirements,
                fallback_strategies=fallback_strategies
            )
            
        except json.JSONDecodeError:
            logger.error(f"无法解析执行计划响应: {response}")
            return ExecutionPlan(
                tool_sequence=selected_tools,
                dependencies={},
                parallel_groups=[[tool] for tool in selected_tools]
            )
    
    def _parse_validation_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """解析验证响应"""
        try:
            data = json.loads(response)
            
            is_valid = data.get("is_valid", False)
            error_message = data.get("error_message")
            
            return is_valid, error_message
            
        except json.JSONDecodeError:
            logger.error(f"无法解析验证响应: {response}")
            return False, "验证响应解析失败"


class MockLLMInterface(LLMInterface):
    """模拟LLM接口，用于测试"""
    
    def __init__(self):
        self.call_count = 0
    
    async def select_tools(
        self, 
        query: str, 
        available_tools: List[Dict[str, Any]], 
        context: ToolcallContext,
        max_tools: Optional[int] = None
    ) -> ToolSelectionResult:
        """模拟工具选择"""
        await asyncio.sleep(0.1)  # 模拟延迟
        
        # 简单的模拟逻辑
        selected = []
        scores = {}
        
        for tool in available_tools[:max_tools or len(available_tools)]:
            # 基于关键词匹配进行简单选择
            if any(keyword in query.lower() for keyword in tool["name"].lower().split()):
                selected.append(tool["id"])
                scores[tool["id"]] = 0.8
        
        return ToolSelectionResult(
            selected_tools=selected,
            confidence_scores=scores,
            reasoning=f"基于关键词匹配选择工具",
            alternatives=[],
            requires_clarification=len(selected) == 0
        )
    
    async def infer_parameters(
        self, 
        query: str, 
        selected_tools: List[str], 
        tool_definitions: List[Dict[str, Any]], 
        context: ToolcallContext
    ) -> ParameterInferenceResult:
        """模拟参数推断"""
        await asyncio.sleep(0.1)
        
        inferred = {}
        scores = {}
        
        for tool_id in selected_tools:
            tool_def = next((t for t in tool_definitions if t["id"] == tool_id), None)
            if tool_def:
                # 简单的参数推断
                params = {}
                for param in tool_def.get("parameters", []):
                    if param.get("required", True):
                        # 设置默认值
                        if param["type"] == "str":
                            params[param["name"]] = f"default_{param['name']}"
                        elif param["type"] == "int":
                            params[param["name"]] = 1
                        elif param["type"] == "bool":
                            params[param["name"]] = False
                
                inferred[tool_id] = params
                scores[tool_id] = 0.7
        
        return ParameterInferenceResult(
            inferred_parameters=inferred,
            confidence_scores=scores,
            reasoning="使用默认参数值",
            missing_parameters={},
            ambiguous_parameters={}
        )
    
    async def generate_execution_plan(
        self, 
        query: str, 
        selected_tools: List[str], 
        inferred_parameters: Dict[str, Any], 
        context: ToolcallContext
    ) -> ExecutionPlan:
        """模拟执行计划生成"""
        await asyncio.sleep(0.1)
        
        return ExecutionPlan(
            tool_sequence=selected_tools,
            dependencies={},
            parallel_groups=[[tool] for tool in selected_tools],
            estimated_time=len(selected_tools) * 2.0
        )
    
    async def validate_plan(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> Tuple[bool, Optional[str]]:
        """模拟计划验证"""
        await asyncio.sleep(0.05)
        
        return True, None