"""
智能工具选择器

负责根据用户查询和上下文智能选择合适的工具
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import asyncio

from .models import ToolSelectionResult, ToolcallContext
from .llm_interface import LLMInterface


logger = logging.getLogger(__name__)


class ToolSelector:
    """智能工具选择器"""
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        self.selection_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
    async def select_tools(
        self,
        query: str,
        available_tools: List[Dict[str, Any]],
        context: ToolcallContext,
        max_tools: Optional[int] = None,
        use_cache: bool = True
    ) -> ToolSelectionResult:
        """选择合适的工具"""
        start_time = time.time()
        
        try:
            # 检查缓存
            if use_cache:
                cached_result = self._get_cached_selection(query, available_tools)
                if cached_result:
                    logger.info("使用缓存的工具选择结果")
                    return cached_result
            
            # 基于规则的预筛选
            prefiltered_tools = self._prefilter_tools(query, available_tools)
            
            # 如果预筛选后没有工具，返回空结果
            if not prefiltered_tools:
                return ToolSelectionResult(
                    selected_tools=[],
                    confidence_scores={},
                    reasoning="没有找到匹配的工具",
                    requires_clarification=True,
                    clarification_questions=["请提供更具体的需求描述"]
                )
            
            # 使用LLM进行智能选择
            llm_result = await self.llm_interface.select_tools(
                query=query,
                available_tools=prefiltered_tools,
                context=context,
                max_tools=max_tools
            )
            
            # 后处理和验证
            final_result = self._postprocess_selection(
                llm_result, available_tools, query
            )
            
            # 记录选择历史
            self._record_selection(query, final_result, time.time() - start_time)
            
            # 缓存结果
            if use_cache:
                self._cache_selection(query, available_tools, final_result)
            
            logger.info(f"工具选择完成: {final_result.selected_tools}")
            return final_result
            
        except Exception as e:
            logger.error(f"工具选择失败: {e}")
            return ToolSelectionResult(
                selected_tools=[],
                confidence_scores={},
                reasoning=f"工具选择过程出错: {str(e)}",
                requires_clarification=True,
                clarification_questions=["系统暂时无法处理您的请求，请稍后重试"]
            )
    
    def _prefilter_tools(self, query: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于规则的预筛选"""
        query_lower = query.lower()
        filtered = []
        
        # 关键词匹配
        for tool in tools:
            # 检查工具名称
            if any(keyword in tool["name"].lower() for keyword in query_lower.split()):
                filtered.append(tool)
                continue
            
            # 检查工具描述
            if any(keyword in tool.get("description", "").lower() for keyword in query_lower.split()):
                filtered.append(tool)
                continue
            
            # 检查标签
            if any(keyword in tag.lower() for tag in tool.get("tags", []) for keyword in query_lower.split()):
                filtered.append(tool)
                continue
        
        # 如果预筛选结果太少，回退到所有工具
        if len(filtered) < 2 and len(tools) > 2:
            logger.info("预筛选结果太少，回退到所有工具")
            return tools
        
        return filtered
    
    def _postprocess_selection(
        self, 
        result: ToolSelectionResult, 
        available_tools: List[Dict[str, Any]], 
        query: str
    ) -> ToolSelectionResult:
        """后处理和验证选择结果"""
        # 验证工具ID有效性
        valid_tools = []
        valid_scores = {}
        tool_ids = {tool["id"] for tool in available_tools}
        
        for tool_id in result.selected_tools:
            if tool_id in tool_ids:
                valid_tools.append(tool_id)
                valid_scores[tool_id] = result.confidence_scores.get(tool_id, 0.0)
            else:
                logger.warning(f"无效的工具ID: {tool_id}")
        
        # 调整置信度分数
        if valid_tools:
            max_score = max(valid_scores.values())
            if max_score > 0:
                # 归一化分数
                for tool_id in valid_scores:
                    valid_scores[tool_id] = valid_scores[tool_id] / max_score
        
        # 检查是否需要澄清
        requires_clarification = (
            result.requires_clarification or 
            len(valid_tools) == 0 or
            all(score < 0.3 for score in valid_scores.values())
        )
        
        # 生成澄清问题
        clarification_questions = result.clarification_questions.copy()
        if len(valid_tools) == 0:
            clarification_questions.append("没有找到匹配的工具，请尝试使用不同的关键词")
        elif all(score < 0.3 for score in valid_scores.values()):
            clarification_questions.append("工具匹配度较低，请提供更具体的需求")
        
        return ToolSelectionResult(
            selected_tools=valid_tools,
            confidence_scores=valid_scores,
            reasoning=result.reasoning,
            alternatives=result.alternatives,
            requires_clarification=requires_clarification,
            clarification_questions=clarification_questions
        )
    
    def _get_cached_selection(self, query: str, tools: List[Dict[str, Any]]) -> Optional[ToolSelectionResult]:
        """获取缓存的选择结果"""
        # 简化的缓存逻辑：基于查询文本和工具列表的哈希
        cache_key = self._generate_cache_key(query, tools)
        
        # 这里应该使用Redis或其他缓存系统
        # 目前返回None表示不使用缓存
        return None
    
    def _cache_selection(self, query: str, tools: List[Dict[str, Any]], result: ToolSelectionResult):
        """缓存选择结果"""
        cache_key = self._generate_cache_key(query, tools)
        
        # 这里应该实现实际的缓存逻辑
        # 目前只是记录到内存中用于演示
        logger.debug(f"缓存工具选择结果: {cache_key}")
    
    def _generate_cache_key(self, query: str, tools: List[Dict[str, Any]]) -> str:
        """生成缓存键"""
        import hashlib
        
        tool_ids = sorted([tool["id"] for tool in tools])
        content = f"{query}|{','.join(tool_ids)}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _record_selection(self, query: str, result: ToolSelectionResult, duration: float):
        """记录选择历史"""
        session_id = "default"  # 应该从上下文获取
        
        record = {
            "query": query,
            "selected_tools": result.selected_tools,
            "confidence_scores": result.confidence_scores,
            "duration": duration,
            "timestamp": time.time()
        }
        
        self.selection_history[session_id].append(record)
        
        # 保持历史记录在合理范围内
        if len(self.selection_history[session_id]) > 1000:
            self.selection_history[session_id] = self.selection_history[session_id][-500:]
        
        # 更新性能指标
        self._update_performance_metrics(result, duration)
    
    def _update_performance_metrics(self, result: ToolSelectionResult, duration: float):
        """更新性能指标"""
        for tool_id in result.selected_tools:
            if tool_id not in self.performance_metrics:
                self.performance_metrics[tool_id] = {
                    "selection_count": 0,
                    "total_duration": 0.0,
                    "average_confidence": 0.0
                }
            
            metrics = self.performance_metrics[tool_id]
            metrics["selection_count"] += 1
            metrics["total_duration"] += duration
            
            # 更新平均置信度
            current_avg = metrics["average_confidence"]
            count = metrics["selection_count"]
            new_confidence = result.confidence_scores.get(tool_id, 0.0)
            metrics["average_confidence"] = (current_avg * (count - 1) + new_confidence) / count
    
    def get_selection_statistics(self, session_id: str = "default") -> Dict[str, Any]:
        """获取选择统计信息"""
        history = self.selection_history.get(session_id, [])
        
        if not history:
            return {
                "total_selections": 0,
                "average_duration": 0.0,
                "most_selected_tools": [],
                "performance_metrics": {}
            }
        
        # 统计最常选择的工具
        tool_counts = defaultdict(int)
        total_duration = 0.0
        
        for record in history:
            total_duration += record["duration"]
            for tool_id in record["selected_tools"]:
                tool_counts[tool_id] += 1
        
        most_selected = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_selections": len(history),
            "average_duration": total_duration / len(history),
            "most_selected_tools": most_selected,
            "performance_metrics": dict(self.performance_metrics)
        }
    
    def get_tool_popularity(self) -> Dict[str, float]:
        """获取工具流行度"""
        tool_popularity = defaultdict(float)
        total_selections = 0
        
        for session_history in self.selection_history.values():
            for record in session_history:
                total_selections += len(record["selected_tools"])
                for tool_id in record["selected_tools"]:
                    tool_popularity[tool_id] += 1
        
        # 计算流行度分数
        if total_selections > 0:
            for tool_id in tool_popularity:
                tool_popularity[tool_id] = tool_popularity[tool_id] / total_selections
        
        return dict(tool_popularity)
    
    def suggest_alternative_tools(self, query: str, current_tools: List[str]) -> List[str]:
        """建议替代工具"""
        # 基于历史选择模式推荐相似工具
        suggestions = []
        
        # 查找相似查询的工具选择模式
        for session_history in self.selection_history.values():
            for record in session_history:
                if self._is_similar_query(query, record["query"]):
                    # 推荐该查询中选择了但当前未选择的工具
                    for tool_id in record["selected_tools"]:
                        if tool_id not in current_tools:
                            suggestions.append(tool_id)
        
        # 去重并限制数量
        unique_suggestions = list(set(suggestions))[:5]
        
        return unique_suggestions
    
    def _is_similar_query(self, query1: str, query2: str) -> bool:
        """判断两个查询是否相似"""
        # 简单的相似度判断：基于共同关键词
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Jaccard相似度
        similarity = len(intersection) / len(union)
        
        return similarity > 0.3  # 阈值可调