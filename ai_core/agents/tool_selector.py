"""
上下文感知的工具选择算法
基于用户意图、历史上下文和工具特性进行智能工具选择
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """工具类别枚举"""
    DATA_PROCESSING = "data_processing"
    FILE_MANAGEMENT = "file_management"
    WEB_SCRAPING = "web_scraping"
    API_INTEGRATION = "api_integration"
    CODE_EXECUTION = "code_execution"
    TEXT_PROCESSING = "text_processing"
    MULTIMEDIA = "multimedia"
    SYSTEM_OPERATIONS = "system_operations"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"


@dataclass
class ToolCapability:
    """工具能力描述"""
    name: str
    category: ToolCategory
    description: str
    keywords: List[str]
    input_types: List[str]
    output_types: List[str]
    complexity_score: float = 0.5
    execution_time_estimate: float = 1.0
    reliability_score: float = 0.9
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = "system"


@dataclass
class ContextInfo:
    """上下文信息"""
    user_intent: str
    conversation_history: List[Dict[str, Any]]
    current_task: Optional[str]
    available_tools: List[ToolCapability]
    user_preferences: Dict[str, Any]
    system_state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolMatch:
    """工具匹配结果"""
    tool: ToolCapability
    relevance_score: float
    confidence: float
    reasoning: str
    alternative_tools: List[str] = field(default_factory=list)


class ContextualToolSelector:
    """上下文感知的工具选择器"""
    
    def __init__(self):
        self.tool_registry: Dict[str, ToolCapability] = {}
        self.usage_history: Dict[str, List[datetime]] = defaultdict(list)
        self.success_rates: Dict[str, float] = {}
        self.context_weights = {
            'intent_match': 0.3,
            'keyword_relevance': 0.2,
            'historical_success': 0.2,
            'tool_reliability': 0.15,
            'complexity_appropriateness': 0.1,
            'user_preference': 0.05
        }
        self._initialize_default_tools()
    
    def _initialize_default_tools(self):
        """初始化默认工具"""
        default_tools = [
            ToolCapability(
                name="file_reader",
                category=ToolCategory.FILE_MANAGEMENT,
                description="读取文件内容",
                keywords=["读取", "文件", "打开", "load", "read"],
                input_types=["file_path"],
                output_types=["text", "content"],
                complexity_score=0.3,
                execution_time_estimate=0.5
            ),
            ToolCapability(
                name="web_scraper",
                category=ToolCategory.WEB_SCRAPING,
                description="从网页提取数据",
                keywords=["网页", "爬取", "抓取", "scrape", "web"],
                input_types=["url"],
                output_types=["html", "data"],
                complexity_score=0.7,
                execution_time_estimate=2.0
            ),
            ToolCapability(
                name="data_analyzer",
                category=ToolCategory.DATA_PROCESSING,
                description="数据分析处理",
                keywords=["分析", "统计", "数据", "analyze", "data"],
                input_types=["dataset", "data"],
                output_types=["results", "charts"],
                complexity_score=0.8,
                execution_time_estimate=3.0
            ),
            ToolCapability(
                name="api_caller",
                category=ToolCategory.API_INTEGRATION,
                description="调用外部API",
                keywords=["API", "接口", "调用", "request", "fetch"],
                input_types=["endpoint", "parameters"],
                output_types=["json", "response"],
                complexity_score=0.6,
                execution_time_estimate=1.5
            ),
            ToolCapability(
                name="code_executor",
                category=ToolCategory.CODE_EXECUTION,
                description="执行代码",
                keywords=["代码", "执行", "运行", "code", "execute"],
                input_types=["code", "language"],
                output_types=["result", "output"],
                complexity_score=0.9,
                execution_time_estimate=2.5
            )
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: ToolCapability):
        """注册工具"""
        self.tool_registry[tool.name] = tool
        if tool.name not in self.success_rates:
            self.success_rates[tool.name] = 0.9
        logger.info(f"注册工具: {tool.name}")
    
    def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """分析用户意图"""
        intent_patterns = {
            'file_operation': r'(读取|打开|保存|删除|文件|load|read|save|delete)',
            'web_operation': r'(网页|爬取|抓取|网站|web|scrape)',
            'data_analysis': r'(分析|统计|计算|data|analyze|calculate)',
            'api_call': r'(API|接口|请求|request|api)',
            'code_execution': r'(代码|执行|运行|code|execute|run)',
            'text_processing': r'(文本|处理|转换|text|process|transform)'
        }
        
        intent_scores = {}
        for intent, pattern in intent_patterns.items():
            matches = len(re.findall(pattern, user_input, re.IGNORECASE))
            intent_scores[intent] = matches / max(len(user_input.split()), 1)
        
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'primary_intent': primary_intent,
            'intent_scores': intent_scores,
            'keywords': self._extract_keywords(user_input),
            'complexity_estimate': self._estimate_complexity(user_input)
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取，实际应用中可以使用更复杂的NLP技术
        words = re.findall(r'\b\w+\b', text.lower())
        # 过滤常见停用词
        stop_words = {'的', '了', '在', '是', '我', '你', '他', '她', '它', '这', '那', '一', '个'}
        return [word for word in words if word not in stop_words and len(word) > 1]
    
    def _estimate_complexity(self, text: str) -> float:
        """估计任务复杂度"""
        complexity_indicators = {
            'multiple_operations': len(re.findall(r'(和|与|以及|and)', text)),
            'conditional_logic': len(re.findall(r'(如果|那么|否则|if|else)', text)),
            'data_processing': len(re.findall(r'(分析|计算|处理|analyze|calculate)', text)),
            'integration': len(re.findall(r'(集成|连接|combine|integrate)', text))
        }
        
        complexity_score = sum(complexity_indicators.values()) * 0.2
        return min(complexity_score, 1.0)
    
    def calculate_tool_relevance(self, tool: ToolCapability, intent_analysis: Dict[str, Any], context: ContextInfo) -> float:
        """计算工具相关性分数"""
        score = 0.0
        
        # 意图匹配分数
        if intent_analysis['primary_intent'] in tool.category.value:
            score += self.context_weights['intent_match']
        
        # 关键词相关性
        keyword_matches = sum(1 for keyword in intent_analysis['keywords'] 
                            if keyword in [k.lower() for k in tool.keywords])
        if tool.keywords:
            keyword_score = keyword_matches / len(tool.keywords)
            score += keyword_score * self.context_weights['keyword_relevance']
        
        # 历史成功率
        if tool.name in self.success_rates:
            score += self.success_rates[tool.name] * self.context_weights['historical_success']
        
        # 工具可靠性
        score += tool.reliability_score * self.context_weights['tool_reliability']
        
        # 复杂度适配性
        complexity_diff = abs(intent_analysis['complexity_estimate'] - tool.complexity_score)
        complexity_score = 1.0 - complexity_diff
        score += complexity_score * self.context_weights['complexity_appropriateness']
        
        # 用户偏好
        if tool.name in context.user_preferences.get('preferred_tools', []):
            score += self.context_weights['user_preference']
        
        return min(score, 1.0)
    
    def select_tools(self, context: ContextInfo, max_tools: int = 5) -> List[ToolMatch]:
        """选择最适合的工具"""
        intent_analysis = self.analyze_intent(context.user_intent)
        
        tool_matches = []
        for tool_name, tool in self.tool_registry.items():
            relevance_score = self.calculate_tool_relevance(tool, intent_analysis, context)
            
            if relevance_score > 0.3:  # 最低阈值
                confidence = self._calculate_confidence(tool, relevance_score, context)
                reasoning = self._generate_reasoning(tool, intent_analysis, relevance_score)
                
                tool_match = ToolMatch(
                    tool=tool,
                    relevance_score=relevance_score,
                    confidence=confidence,
                    reasoning=reasoning
                )
                tool_matches.append(tool_match)
        
        # 按相关性分数排序
        tool_matches.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 返回前N个工具
        return tool_matches[:max_tools]
    
    def _calculate_confidence(self, tool: ToolCapability, relevance_score: float, context: ContextInfo) -> float:
        """计算选择置信度"""
        confidence = relevance_score
        
        # 基于历史使用频率调整置信度
        recent_usage = sum(1 for usage_time in self.usage_history[tool.name] 
                          if usage_time > datetime.now() - timedelta(days=7))
        if recent_usage > 0:
            confidence *= min(1.0 + recent_usage * 0.1, 1.5)
        
        # 基于工具可靠性调整
        confidence *= tool.reliability_score
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self, tool: ToolCapability, intent_analysis: Dict[str, Any], relevance_score: float) -> str:
        """生成选择理由"""
        reasons = []
        
        if intent_analysis['primary_intent'] in tool.category.value:
            reasons.append(f"工具类别匹配用户意图({intent_analysis['primary_intent']})")
        
        keyword_matches = sum(1 for keyword in intent_analysis['keywords'] 
                            if keyword in [k.lower() for k in tool.keywords])
        if keyword_matches > 0:
            reasons.append(f"关键词匹配度: {keyword_matches}/{len(tool.keywords)}")
        
        if tool.reliability_score > 0.9:
            reasons.append("高可靠性工具")
        
        if tool.complexity_score < 0.5:
            reasons.append("适合简单任务")
        elif tool.complexity_score > 0.8:
            reasons.append("适合复杂任务")
        
        return "; ".join(reasons)
    
    def update_tool_performance(self, tool_name: str, success: bool, execution_time: float):
        """更新工具性能统计"""
        now = datetime.now()
        self.usage_history[tool_name].append(now)
        
        # 更新成功率 (使用指数移动平均)
        if tool_name in self.success_rates:
            alpha = 0.1  # 平滑因子
            self.success_rates[tool_name] = (alpha * (1.0 if success else 0.0) + 
                                           (1 - alpha) * self.success_rates[tool_name])
        else:
            self.success_rates[tool_name] = 1.0 if success else 0.0
        
        # 清理过旧的使用记录 (保留最近30天)
        cutoff_date = now - timedelta(days=30)
        self.usage_history[tool_name] = [
            usage_time for usage_time in self.usage_history[tool_name] 
            if usage_time > cutoff_date
        ]
        
        logger.info(f"更新工具性能: {tool_name}, 成功率: {self.success_rates[tool_name]:.2f}")
    
    def get_tool_recommendations(self, context: ContextInfo, count: int = 3) -> List[ToolCapability]:
        """获取工具推荐"""
        matches = self.select_tools(context, max_tools=count)
        return [match.tool for match in matches]
    
    def export_tool_registry(self) -> Dict[str, Any]:
        """导出工具注册表"""
        return {
            'tools': {name: {
                'name': tool.name,
                'category': tool.category.value,
                'description': tool.description,
                'keywords': tool.keywords,
                'input_types': tool.input_types,
                'output_types': tool.output_types,
                'complexity_score': tool.complexity_score,
                'reliability_score': tool.reliability_score,
                'version': tool.version
            } for name, tool in self.tool_registry.items()},
            'success_rates': self.success_rates,
            'usage_statistics': {
                tool_name: len(usage_times) 
                for tool_name, usage_times in self.usage_history.items()
            }
        }
    
    def import_tool_registry(self, data: Dict[str, Any]):
        """导入工具注册表"""
        if 'tools' in data:
            for tool_data in data['tools'].values():
                tool = ToolCapability(
                    name=tool_data['name'],
                    category=ToolCategory(tool_data['category']),
                    description=tool_data['description'],
                    keywords=tool_data['keywords'],
                    input_types=tool_data['input_types'],
                    output_types=tool_data['output_types'],
                    complexity_score=tool_data.get('complexity_score', 0.5),
                    reliability_score=tool_data.get('reliability_score', 0.9),
                    version=tool_data.get('version', '1.0')
                )
                self.register_tool(tool)
        
        if 'success_rates' in data:
            self.success_rates.update(data['success_rates'])