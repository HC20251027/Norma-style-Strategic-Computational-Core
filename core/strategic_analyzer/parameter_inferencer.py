"""
参数推断器

负责根据用户查询和工具定义智能推断工具参数
"""

import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import asyncio

from .models import ParameterInferenceResult, ToolcallContext
from .llm_interface import LLMInterface


logger = logging.getLogger(__name__)


class ParameterInferencer:
    """参数推断器"""
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        self.inference_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.parameter_patterns: Dict[str, Dict[str, Any]] = {}
        self.default_values: Dict[str, Dict[str, Any]] = {}
        
    async def infer_parameters(
        self,
        query: str,
        selected_tools: List[str],
        tool_definitions: List[Dict[str, Any]],
        context: ToolcallContext,
        use_cache: bool = True
    ) -> ParameterInferenceResult:
        """推断工具参数"""
        start_time = time.time()
        
        try:
            # 检查缓存
            if use_cache:
                cached_result = self._get_cached_inference(query, selected_tools, tool_definitions)
                if cached_result:
                    logger.info("使用缓存的参数推断结果")
                    return cached_result
            
            # 规则基础的初步推断
            rule_based_params = self._infer_from_rules(query, selected_tools, tool_definitions, context)
            
            # 基于历史模式的推断
            history_based_params = self._infer_from_history(query, selected_tools, tool_definitions)
            
            # 合并规则和历史推断结果
            combined_params = self._merge_inference_results(rule_based_params, history_based_params)
            
            # 使用LLM进行智能推断
            llm_result = await self.llm_interface.infer_parameters(
                query=query,
                selected_tools=selected_tools,
                tool_definitions=tool_definitions,
                context=context
            )
            
            # 合并LLM推断结果
            final_result = self._merge_with_llm_result(combined_params, llm_result, selected_tools)
            
            # 后处理和验证
            final_result = self._postprocess_inference(final_result, tool_definitions)
            
            # 记录推断历史
            self._record_inference(query, final_result, time.time() - start_time)
            
            # 缓存结果
            if use_cache:
                self._cache_inference(query, selected_tools, tool_definitions, final_result)
            
            logger.info(f"参数推断完成: {final_result.inferred_parameters}")
            return final_result
            
        except Exception as e:
            logger.error(f"参数推断失败: {e}")
            return ParameterInferenceResult(
                inferred_parameters={},
                confidence_scores={},
                reasoning=f"参数推断过程出错: {str(e)}",
                missing_parameters={tool_id: ["推断失败"] for tool_id in selected_tools}
            )
    
    def _infer_from_rules(
        self, 
        query: str, 
        selected_tools: List[str], 
        tool_definitions: List[Dict[str, Any]], 
        context: ToolcallContext
    ) -> Dict[str, Dict[str, Any]]:
        """基于规则的参数推断"""
        inferred_params = {}
        query_lower = query.lower()
        
        for tool_id in selected_tools:
            tool_def = next((t for t in tool_definitions if t["id"] == tool_id), None)
            if not tool_def:
                continue
            
            params = {}
            
            # 解析查询中的数值参数
            for param in tool_def.get("parameters", []):
                param_name = param["name"]
                param_type = param.get("type", "str")
                
                # 数值类型推断
                if param_type in ["int", "float"]:
                    numbers = re.findall(r'\d+(?:\.\d+)?', query)
                    if numbers:
                        try:
                            if param_type == "int":
                                params[param_name] = int(numbers[0])
                            else:
                                params[param_name] = float(numbers[0])
                        except ValueError:
                            pass
                
                # 布尔类型推断
                elif param_type == "bool":
                    if any(word in query_lower for word in ["是", "true", "启用", "开启"]):
                        params[param_name] = True
                    elif any(word in query_lower for word in ["否", "false", "禁用", "关闭"]):
                        params[param_name] = False
                
                # 字符串类型推断
                elif param_type == "str":
                    # 从上下文中提取相关字符串
                    if param_name in context.metadata:
                        params[param_name] = context.metadata[param_name]
            
            inferred_params[tool_id] = params
        
        return inferred_params
    
    def _infer_from_history(
        self, 
        query: str, 
        selected_tools: List[str], 
        tool_definitions: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """基于历史模式的参数推断"""
        inferred_params = {}
        
        # 查找相似的历史查询
        similar_queries = self._find_similar_queries(query)
        
        for tool_id in selected_tools:
            tool_def = next((t for t in tool_definitions if t["id"] == tool_id), None)
            if not tool_def:
                continue
            
            params = {}
            
            # 从相似查询中学习参数模式
            for similar_query in similar_queries:
                history_params = self._extract_parameters_from_history(tool_id, similar_query)
                if history_params:
                    # 合并参数值，优先选择最频繁的值
                    for param_name, value in history_params.items():
                        if param_name not in params:
                            params[param_name] = value
            
            inferred_params[tool_id] = params
        
        return inferred_params
    
    def _merge_inference_results(
        self, 
        rule_params: Dict[str, Dict[str, Any]], 
        history_params: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """合并推断结果"""
        merged = {}
        
        all_tools = set(rule_params.keys()) | set(history_params.keys())
        
        for tool_id in all_tools:
            tool_params = {}
            
            # 合并规则推断结果
            if tool_id in rule_params:
                tool_params.update(rule_params[tool_id])
            
            # 合并历史推断结果（仅补充缺失的参数）
            if tool_id in history_params:
                for param_name, value in history_params[tool_id].items():
                    if param_name not in tool_params:
                        tool_params[param_name] = value
            
            merged[tool_id] = tool_params
        
        return merged
    
    def _merge_with_llm_result(
        self, 
        combined_params: Dict[str, Dict[str, Any]], 
        llm_result: ParameterInferenceResult, 
        selected_tools: List[str]
    ) -> ParameterInferenceResult:
        """合并LLM推断结果"""
        # 合并参数推断结果
        final_params = {}
        confidence_scores = {}
        
        for tool_id in selected_tools:
            tool_params = combined_params.get(tool_id, {})
            llm_params = llm_result.inferred_parameters.get(tool_id, {})
            
            # LLM结果优先，规则结果补充
            final_tool_params = llm_params.copy()
            final_tool_params.update(tool_params)
            
            final_params[tool_id] = final_tool_params
            confidence_scores[tool_id] = llm_result.confidence_scores.get(tool_id, 0.5)
        
        return ParameterInferenceResult(
            inferred_parameters=final_params,
            confidence_scores=confidence_scores,
            reasoning=llm_result.reasoning,
            missing_parameters=llm_result.missing_parameters,
            ambiguous_parameters=llm_result.ambiguous_parameters
        )
    
    def _postprocess_inference(
        self, 
        result: ParameterInferenceResult, 
        tool_definitions: List[Dict[str, Any]]
    ) -> ParameterInferenceResult:
        """后处理和验证推断结果"""
        # 验证参数类型和范围
        for tool_id, params in result.inferred_parameters.items():
            tool_def = next((t for t in tool_definitions if t["id"] == tool_id), None)
            if not tool_def:
                continue
            
            # 验证每个参数
            validated_params = {}
            for param_name, value in params.items():
                param_def = next((p for p in tool_def.get("parameters", []) if p["name"] == param_name), None)
                if param_def:
                    if self._validate_parameter_value(value, param_def):
                        validated_params[param_name] = value
                    else:
                        logger.warning(f"参数值验证失败: {tool_id}.{param_name} = {value}")
                else:
                    # 未定义的参数，直接保留
                    validated_params[param_name] = value
            
            result.inferred_parameters[tool_id] = validated_params
        
        return result
    
    def _validate_parameter_value(self, value: Any, param_def: Dict[str, Any]) -> bool:
        """验证参数值"""
        param_type = param_def.get("type", "str")
        
        try:
            if param_type == "int":
                return isinstance(value, int)
            elif param_type == "float":
                return isinstance(value, (int, float))
            elif param_type == "bool":
                return isinstance(value, bool)
            elif param_type == "str":
                return isinstance(value, str)
            elif param_type == "list":
                return isinstance(value, list)
            elif param_type == "dict":
                return isinstance(value, dict)
            else:
                # 未知类型，默认通过
                return True
        except Exception:
            return False
    
    def _get_cached_inference(
        self, 
        query: str, 
        selected_tools: List[str], 
        tool_definitions: List[Dict[str, Any]]
    ) -> Optional[ParameterInferenceResult]:
        """获取缓存的推断结果"""
        cache_key = self._generate_inference_cache_key(query, selected_tools, tool_definitions)
        
        # 这里应该实现实际的缓存逻辑
        return None
    
    def _cache_inference(
        self, 
        query: str, 
        selected_tools: List[str], 
        tool_definitions: List[Dict[str, Any]], 
        result: ParameterInferenceResult
    ):
        """缓存推断结果"""
        cache_key = self._generate_inference_cache_key(query, selected_tools, tool_definitions)
        
        # 这里应该实现实际的缓存逻辑
        logger.debug(f"缓存参数推断结果: {cache_key}")
    
    def _generate_inference_cache_key(
        self, 
        query: str, 
        selected_tools: List[str], 
        tool_definitions: List[Dict[str, Any]]
    ) -> str:
        """生成推断缓存键"""
        import hashlib
        
        tool_ids = sorted(selected_tools)
        content = f"{query}|{','.join(tool_ids)}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _record_inference(self, query: str, result: ParameterInferenceResult, duration: float):
        """记录推断历史"""
        session_id = "default"  # 应该从上下文获取
        
        record = {
            "query": query,
            "inferred_parameters": result.inferred_parameters,
            "confidence_scores": result.confidence_scores,
            "duration": duration,
            "timestamp": time.time()
        }
        
        self.inference_history[session_id].append(record)
        
        # 保持历史记录在合理范围内
        if len(self.inference_history[session_id]) > 1000:
            self.inference_history[session_id] = self.inference_history[session_id][-500:]
    
    def _find_similar_queries(self, query: str, threshold: float = 0.3) -> List[str]:
        """查找相似的历史查询"""
        similar_queries = []
        query_words = set(query.lower().split())
        
        for session_history in self.inference_history.values():
            for record in session_history:
                history_query = record["query"]
                history_words = set(history_query.lower().split())
                
                if not query_words or not history_words:
                    continue
                
                # 计算Jaccard相似度
                intersection = query_words.intersection(history_words)
                union = query_words.union(history_words)
                similarity = len(intersection) / len(union)
                
                if similarity > threshold:
                    similar_queries.append(history_query)
        
        return similar_queries[:5]  # 限制返回数量
    
    def _extract_parameters_from_history(self, tool_id: str, query: str) -> Optional[Dict[str, Any]]:
        """从历史记录中提取参数"""
        for session_history in self.inference_history.values():
            for record in session_history:
                if record["query"] == query and tool_id in record["inferred_parameters"]:
                    return record["inferred_parameters"][tool_id]
        return None
    
    def get_inference_statistics(self, session_id: str = "default") -> Dict[str, Any]:
        """获取推断统计信息"""
        history = self.inference_history.get(session_id, [])
        
        if not history:
            return {
                "total_inferences": 0,
                "average_duration": 0.0,
                "parameter_accuracy": {},
                "most_inferred_parameters": {}
            }
        
        # 统计推断的参数
        param_counts = defaultdict(int)
        total_duration = 0.0
        
        for record in history:
            total_duration += record["duration"]
            for tool_id, params in record["inferred_parameters"].items():
                for param_name in params.keys():
                    param_counts[f"{tool_id}.{param_name}"] += 1
        
        most_inferred = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_inferences": len(history),
            "average_duration": total_duration / len(history),
            "parameter_accuracy": {},  # 需要额外的验证逻辑
            "most_inferred_parameters": most_inferred
        }
    
    def learn_parameter_pattern(self, tool_id: str, param_name: str, pattern: str, value: Any):
        """学习参数模式"""
        if tool_id not in self.parameter_patterns:
            self.parameter_patterns[tool_id] = {}
        
        if param_name not in self.parameter_patterns[tool_id]:
            self.parameter_patterns[tool_id][param_name] = []
        
        self.parameter_patterns[tool_id][param_name].append({
            "pattern": pattern,
            "value": value,
            "timestamp": time.time()
        })
        
        # 保持模式数量在合理范围内
        if len(self.parameter_patterns[tool_id][param_name]) > 100:
            self.parameter_patterns[tool_id][param_name] = self.parameter_patterns[tool_id][param_name][-50:]
    
    def suggest_parameter_values(self, tool_id: str, param_name: str, context: str) -> List[Any]:
        """基于上下文建议参数值"""
        if tool_id not in self.parameter_patterns or param_name not in self.parameter_patterns[tool_id]:
            return []
        
        patterns = self.parameter_patterns[tool_id][param_name]
        suggestions = []
        
        # 基于模式匹配建议值
        for pattern_info in patterns:
            if self._match_pattern(context, pattern_info["pattern"]):
                suggestions.append(pattern_info["value"])
        
        # 去重并限制数量
        unique_suggestions = list(set(suggestions))[:5]
        
        return unique_suggestions
    
    def _match_pattern(self, context: str, pattern: str) -> bool:
        """匹配模式"""
        # 简单的模式匹配逻辑
        context_lower = context.lower()
        pattern_lower = pattern.lower()
        
        # 关键词匹配
        pattern_words = pattern_lower.split()
        matches = sum(1 for word in pattern_words if word in context_lower)
        
        return matches / len(pattern_words) > 0.5 if pattern_words else False