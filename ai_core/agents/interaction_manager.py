#!/usr/bin/env python3
"""
诺玛交互管理器
品牌化交互体验管理

功能特性:
- 智能交互流程管理
- 品牌化交互模式
- 上下文感知交互
- 交互质量监控
- 个性化交互优化

作者: 皇
创建时间: 2025-10-31
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import random

class InteractionType(Enum):
    """交互类型枚举"""
    GREETING = "greeting"             # 问候交互
    CONVERSATION = "conversation"     # 对话交互
    QUESTION = "question"             # 问答交互
    COMMAND = "command"               # 命令交互
    FEEDBACK = "feedback"             # 反馈交互
    ERROR = "error"                   # 错误处理交互
    GOODBYE = "goodbye"              # 告别交互

class InteractionState(Enum):
    """交互状态枚举"""
    INITIALIZING = "initializing"     # 初始化
    ACTIVE = "active"                 # 活跃
    PAUSED = "paused"                 # 暂停
    COMPLETING = "completing"         # 完成中
    COMPLETED = "completed"           # 已完成
    ERROR = "error"                   # 错误

class InteractionQuality(Enum):
    """交互质量等级"""
    EXCELLENT = "excellent"           # 优秀
    GOOD = "good"                     # 良好
    AVERAGE = "average"               # 一般
    POOR = "poor"                     # 较差
    FAILED = "failed"                 # 失败

@dataclass
class InteractionSession:
    """交互会话数据类"""
    session_id: str
    user_id: str
    start_time: str
    end_time: Optional[str]
    interaction_type: InteractionType
    state: InteractionState
    message_count: int
    quality_score: float
    context_data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class InteractionMetrics:
    """交互指标数据类"""
    response_time: float              # 响应时间
    message_length: int              # 消息长度
    user_satisfaction: Optional[float] # 用户满意度
    context_retention: float         # 上下文保持度
    brand_consistency: float         # 品牌一致性
    technical_accuracy: float        # 技术准确性

class NormaInteractionManager:
    """诺玛交互管理器"""
    
    def __init__(self):
        self.active_sessions = {}
        self.interaction_history = []
        self.quality_metrics = {}
        self.brand_guidelines = self._initialize_brand_guidelines()
        self.interaction_patterns = self._initialize_interaction_patterns()
        self.context_managers = {}
        self.quality_analyzers = {}
        
        # 交互统计
        self.stats = {
            "total_sessions": 0,
            "total_messages": 0,
            "average_quality": 0.0,
            "popular_interactions": {},
            "user_satisfaction": 0.0
        }
    
    def _initialize_brand_guidelines(self) -> Dict[str, Any]:
        """初始化品牌指导原则"""
        return {
            "communication_style": {
                "tone": "professional_friendly",
                "formality_level": 0.7,
                "technical_depth": 0.8,
                "empathy_level": 0.6,
                "humor_level": 0.3
            },
            "response_structure": {
                "greeting": True,
                "acknowledgment": True,
                "main_content": True,
                "action_items": True,
                "closing": True
            },
            "brand_elements": {
                "identity_mentions": ["诺玛·劳恩斯", "卡塞尔学院", "主控系统"],
                "personality_traits": ["理性", "逻辑", "专业", "负责"],
                "avoid_phrases": ["绝对", "肯定", "永远", "一定"],
                "preferred_terms": ["建议", "推荐", "需要", "重要"]
            },
            "quality_standards": {
                "response_time_max": 5.0,      # 最大响应时间(秒)
                "message_length_min": 10,      # 最小消息长度
                "message_length_max": 500,     # 最大消息长度
                "context_retention_min": 0.8,  # 最小上下文保持度
                "brand_consistency_min": 0.9   # 最小品牌一致性
            }
        }
    
    def _initialize_interaction_patterns(self) -> Dict[str, Dict[str, Any]]:
        """初始化交互模式"""
        return {
            "greeting": {
                "duration_range": (2, 5),      # 持续时间范围(秒)
                "message_count_range": (1, 3), # 消息数量范围
                "quality_weight": 0.2,          # 质量权重
                "required_elements": ["品牌标识", "问候", "服务承诺"]
            },
            "conversation": {
                "duration_range": (30, 300),   # 持续时间范围(秒)
                "message_count_range": (3, 20), # 消息数量范围
                "quality_weight": 0.4,          # 质量权重
                "required_elements": ["上下文保持", "逻辑连贯", "个性化"]
            },
            "question": {
                "duration_range": (5, 30),     # 持续时间范围(秒)
                "message_count_range": (1, 5), # 消息数量范围
                "quality_weight": 0.3,          # 质量权重
                "required_elements": ["准确回答", "详细解释", "相关建议"]
            },
            "command": {
                "duration_range": (3, 15),     # 持续时间范围(秒)
                "message_count_range": (1, 3), # 消息数量范围
                "quality_weight": 0.25,         # 质量权重
                "required_elements": ["确认执行", "进度反馈", "结果报告"]
            },
            "feedback": {
                "duration_range": (2, 10),     # 持续时间范围(秒)
                "message_count_range": (1, 2), # 消息数量范围
                "quality_weight": 0.15,         # 质量权重
                "required_elements": ["感谢", "确认", "改进承诺"]
            },
            "error": {
                "duration_range": (3, 20),     # 持续时间范围(秒)
                "message_count_range": (1, 4), # 消息数量范围
                "quality_weight": 0.1,          # 质量权重
                "required_elements": ["道歉", "解释", "解决方案"]
            },
            "goodbye": {
                "duration_range": (2, 8),      # 持续时间范围(秒)
                "message_count_range": (1, 2), # 消息数量范围
                "quality_weight": 0.1,          # 质量权重
                "required_elements": ["告别", "服务总结", "后续支持"]
            }
        }
    
    def start_interaction_session(self, user_id: str, interaction_type: InteractionType,
                                context_data: Dict[str, Any] = None) -> str:
        """开始交互会话"""
        try:
            session_id = f"session_{int(datetime.now().timestamp())}_{user_id}"
            
            session = InteractionSession(
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.now().isoformat(),
                end_time=None,
                interaction_type=interaction_type,
                state=InteractionState.INITIALIZING,
                message_count=0,
                quality_score=0.0,
                context_data=context_data or {},
                metadata={}
            )
            
            self.active_sessions[session_id] = session
            self.stats["total_sessions"] += 1
            
            # 初始化上下文管理器
            self.context_managers[session_id] = self._create_context_manager(session_id)
            
            return session_id
            
        except Exception as e:
            print(f"交互会话启动失败: {e}")
            return ""
    
    def _create_context_manager(self, session_id: str) -> Dict[str, Any]:
        """创建上下文管理器"""
        return {
            "conversation_history": [],
            "user_preferences": {},
            "topic_stack": [],
            "emotion_state": "neutral",
            "technical_context": {},
            "brand_elements_used": [],
            "interaction_goals": []
        }
    
    async def process_interaction(self, session_id: str, user_input: str,
                                response_generator: Callable) -> AsyncGenerator[str, None]:
        """处理交互"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"会话 {session_id} 不存在")
            
            session = self.active_sessions[session_id]
            session.state = InteractionState.ACTIVE
            
            # 分析用户输入
            input_analysis = self._analyze_user_input(user_input, session)
            
            # 更新上下文
            self._update_context(session_id, user_input, input_analysis)
            
            # 生成响应
            response_start_time = datetime.now()
            
            async for response_chunk in response_generator(user_input, session):
                # 记录响应片段
                session.message_count += 1
                self.stats["total_messages"] += 1
                
                # 分析响应质量
                quality_metrics = self._analyze_response_quality(
                    response_chunk, session, input_analysis
                )
                
                # 发送响应片段
                yield response_chunk
                
                # 更新会话质量
                self._update_session_quality(session_id, quality_metrics)
            
            # 完成交互处理
            response_time = (datetime.now() - response_start_time).total_seconds()
            self._complete_interaction_step(session_id, user_input, response_time)
            
        except Exception as e:
            print(f"交互处理失败: {e}")
            session.state = InteractionState.ERROR
            yield f"抱歉，处理您的请求时遇到了问题：{str(e)}"
    
    def _analyze_user_input(self, user_input: str, session: InteractionSession) -> Dict[str, Any]:
        """分析用户输入"""
        return {
            "sentiment": self._analyze_sentiment(user_input),
            "complexity": self._analyze_complexity(user_input),
            "intent": self._analyze_intent(user_input),
            "urgency": self._assess_urgency(user_input),
            "technical_level": self._assess_technical_level(user_input),
            "emotional_tone": self._detect_emotional_tone(user_input),
            "context_relevance": self._assess_context_relevance(user_input, session)
        }
    
    def _analyze_sentiment(self, user_input: str) -> str:
        """分析情感倾向"""
        positive_words = ["好", "棒", "优秀", "满意", "感谢", "喜欢", "太棒了"]
        negative_words = ["坏", "差", "不满", "讨厌", "问题", "错误", "糟糕"]
        neutral_words = ["一般", "普通", "正常", "标准"]
        
        input_lower = user_input.lower()
        
        pos_count = sum(1 for word in positive_words if word in input_lower)
        neg_count = sum(1 for word in negative_words if word in input_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_complexity(self, user_input: str) -> str:
        """分析复杂度"""
        technical_terms = ["系统", "数据库", "API", "算法", "架构", "协议", "代码"]
        question_marks = user_input.count('?') + user_input.count('？')
        word_count = len(user_input.split())
        
        tech_count = sum(1 for term in technical_terms if term in user_input)
        
        if tech_count >= 3 or question_marks >= 2 or word_count > 50:
            return "high"
        elif tech_count >= 1 or question_marks >= 1 or word_count > 20:
            return "medium"
        else:
            return "low"
    
    def _analyze_intent(self, user_input: str) -> str:
        """分析意图"""
        intent_patterns = {
            "greeting": ["你好", "hello", "hi", "早上好", "下午好", "晚上好"],
            "question": ["什么", "为什么", "如何", "怎么", "?", "？"],
            "request": ["请", "需要", "要求", "帮助", "协助"],
            "command": ["执行", "运行", "开始", "停止", "命令"],
            "complaint": ["不满", "问题", "错误", "故障", "抱怨"],
            "praise": ["好", "棒", "优秀", "满意", "感谢"],
            "goodbye": ["再见", "bye", "拜拜", "结束", "退出"]
        }
        
        input_lower = user_input.lower()
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in input_lower for pattern in patterns):
                return intent
        
        return "general"
    
    def _assess_urgency(self, user_input: str) -> str:
        """评估紧急程度"""
        urgent_words = ["紧急", "立即", "马上", "快", "急", "重要"]
        warning_words = ["警告", "注意", "小心", "危险", "风险"]
        
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in urgent_words):
            return "high"
        elif any(word in input_lower for word in warning_words):
            return "medium"
        else:
            return "low"
    
    def _assess_technical_level(self, user_input: str) -> str:
        """评估技术水平"""
        technical_indicators = [
            "系统", "数据库", "API", "算法", "架构", "协议",
            "代码", "编程", "开发", "技术", "专业"
        ]
        
        tech_count = sum(1 for term in technical_indicators if term in user_input)
        
        if tech_count >= 3:
            return "high"
        elif tech_count >= 1:
            return "medium"
        else:
            return "low"
    
    def _detect_emotional_tone(self, user_input: str) -> str:
        """检测情感语调"""
        emotional_indicators = {
            "urgent": ["紧急", "立即", "马上", "快", "急"],
            "frustrated": ["烦", "恼", "生气", "不满", "糟糕"],
            "excited": ["兴奋", "激动", "太好了", "太棒了"],
            "confused": ["困惑", "不明白", "不懂", "迷茫"],
            "calm": ["平静", "冷静", "正常", "一般"]
        }
        
        input_lower = user_input.lower()
        
        for tone, indicators in emotional_indicators.items():
            if any(indicator in input_lower for indicator in indicators):
                return tone
        
        return "neutral"
    
    def _assess_context_relevance(self, user_input: str, session: InteractionSession) -> float:
        """评估上下文相关性"""
        if not session.context_data:
            return 0.5
        
        # 简单的关键词匹配
        context_keywords = []
        for value in session.context_data.values():
            if isinstance(value, str):
                context_keywords.extend(value.split())
        
        input_words = user_input.lower().split()
        matches = sum(1 for word in input_words if word in context_keywords)
        
        return min(1.0, matches / max(1, len(input_words)))
    
    def _update_context(self, session_id: str, user_input: str, analysis: Dict[str, Any]):
        """更新上下文"""
        if session_id in self.context_managers:
            context_manager = self.context_managers[session_id]
            
            # 添加到对话历史
            context_manager["conversation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "analysis": analysis
            })
            
            # 更新情感状态
            context_manager["emotion_state"] = analysis["emotional_tone"]
            
            # 保持历史记录在合理范围内
            if len(context_manager["conversation_history"]) > 20:
                context_manager["conversation_history"] = context_manager["conversation_history"][-20:]
    
    def _analyze_response_quality(self, response: str, session: InteractionSession,
                                input_analysis: Dict[str, Any]) -> InteractionMetrics:
        """分析响应质量"""
        response_time = 0.0  # 这里应该计算实际响应时间
        
        # 消息长度分析
        message_length = len(response)
        
        # 品牌一致性检查
        brand_consistency = self._check_brand_consistency(response)
        
        # 技术准确性评估
        technical_accuracy = self._assess_technical_accuracy(response, input_analysis)
        
        # 上下文保持度
        context_retention = self._calculate_context_retention(response, session)
        
        return InteractionMetrics(
            response_time=response_time,
            message_length=message_length,
            user_satisfaction=None,  # 暂时无法获取
            context_retention=context_retention,
            brand_consistency=brand_consistency,
            technical_accuracy=technical_accuracy
        )
    
    def _check_brand_consistency(self, response: str) -> float:
        """检查品牌一致性"""
        brand_elements = self.brand_guidelines["brand_elements"]
        
        # 检查身份标识
        identity_score = sum(1 for identity in brand_elements["identity_mentions"] 
                           if identity in response) / len(brand_elements["identity_mentions"])
        
        # 检查个性特征
        personality_score = sum(1 for trait in brand_elements["personality_traits"]
                              if trait in response) / len(brand_elements["personality_traits"])
        
        # 检查避免短语
        avoid_score = 1.0
        for avoid_phrase in brand_elements["avoid_phrases"]:
            if avoid_phrase in response:
                avoid_score -= 0.2
        
        # 检查偏好术语
        preferred_score = sum(1 for term in brand_elements["preferred_terms"]
                            if term in response) / len(brand_elements["preferred_terms"])
        
        # 综合评分
        consistency_score = (identity_score + personality_score + avoid_score + preferred_score) / 4
        return max(0.0, min(1.0, consistency_score))
    
    def _assess_technical_accuracy(self, response: str, input_analysis: Dict[str, Any]) -> float:
        """评估技术准确性"""
        # 简单的技术准确性评估
        if input_analysis["technical_level"] == "high":
            # 高技术级别需要包含更多技术术语
            technical_terms = ["系统", "数据", "分析", "处理", "优化"]
            tech_score = sum(1 for term in technical_terms if term in response) / len(technical_terms)
        else:
            tech_score = 0.8  # 默认分数
        
        return min(1.0, tech_score)
    
    def _calculate_context_retention(self, response: str, session: InteractionSession) -> float:
        """计算上下文保持度"""
        if not session.context_data:
            return 0.5
        
        # 检查响应是否与上下文相关
        context_keywords = []
        for value in session.context_data.values():
            if isinstance(value, str):
                context_keywords.extend(value.split())
        
        response_words = response.lower().split()
        matches = sum(1 for word in response_words if word in context_keywords)
        
        return min(1.0, matches / max(1, len(response_words)))
    
    def _update_session_quality(self, session_id: str, metrics: InteractionMetrics):
        """更新会话质量"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # 计算综合质量分数
            quality_score = (
                metrics.brand_consistency * 0.3 +
                metrics.technical_accuracy * 0.25 +
                metrics.context_retention * 0.25 +
                (1.0 if 10 <= metrics.message_length <= 500 else 0.5) * 0.2
            )
            
            session.quality_score = (session.quality_score + quality_score) / 2
    
    def _complete_interaction_step(self, session_id: str, user_input: str, response_time: float):
        """完成交互步骤"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # 更新统计
            if session.interaction_type.value not in self.stats["popular_interactions"]:
                self.stats["popular_interactions"][session.interaction_type.value] = 0
            self.stats["popular_interactions"][session.interaction_type.value] += 1
    
    def end_interaction_session(self, session_id: str) -> Dict[str, Any]:
        """结束交互会话"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "会话不存在"}
            
            session = self.active_sessions[session_id]
            session.state = InteractionState.COMPLETED
            session.end_time = datetime.now().isoformat()
            
            # 计算会话持续时间
            start_time = datetime.fromisoformat(session.start_time)
            end_time = datetime.fromisoformat(session.end_time)
            duration = (end_time - start_time).total_seconds()
            
            # 生成会话报告
            session_report = {
                "session_id": session_id,
                "user_id": session.user_id,
                "duration": duration,
                "message_count": session.message_count,
                "quality_score": session.quality_score,
                "interaction_type": session.interaction_type.value,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "context_summary": self._summarize_context(session_id)
            }
            
            # 移动到历史记录
            self.interaction_history.append(session_report)
            del self.active_sessions[session_id]
            
            # 清理上下文管理器
            if session_id in self.context_managers:
                del self.context_managers[session_id]
            
            return session_report
            
        except Exception as e:
            print(f"会话结束失败: {e}")
            return {"error": str(e)}
    
    def _summarize_context(self, session_id: str) -> Dict[str, Any]:
        """总结上下文"""
        if session_id in self.context_managers:
            context_manager = self.context_managers[session_id]
            
            return {
                "conversation_turns": len(context_manager["conversation_history"]),
                "emotion_state": context_manager["emotion_state"],
                "topics_discussed": len(context_manager["topic_stack"]),
                "brand_elements_used": len(context_manager["brand_elements_used"])
            }
        
        return {}
    
    def get_interaction_analytics(self) -> Dict[str, Any]:
        """获取交互分析"""
        return {
            "session_stats": self.stats,
            "quality_distribution": self._calculate_quality_distribution(),
            "popular_patterns": self._analyze_popular_patterns(),
            "user_satisfaction_trends": self._analyze_satisfaction_trends(),
            "brand_consistency_score": self._calculate_brand_consistency_score(),
            "recommendations": self._generate_improvement_recommendations()
        }
    
    def _calculate_quality_distribution(self) -> Dict[str, int]:
        """计算质量分布"""
        distribution = {"excellent": 0, "good": 0, "average": 0, "poor": 0, "failed": 0}
        
        for session in self.active_sessions.values():
            score = session.quality_score
            if score >= 0.9:
                distribution["excellent"] += 1
            elif score >= 0.7:
                distribution["good"] += 1
            elif score >= 0.5:
                distribution["average"] += 1
            elif score >= 0.3:
                distribution["poor"] += 1
            else:
                distribution["failed"] += 1
        
        return distribution
    
    def _analyze_popular_patterns(self) -> Dict[str, Any]:
        """分析流行模式"""
        return {
            "most_common_intent": max(self.stats["popular_interactions"], 
                                     key=self.stats["popular_interactions"].get) 
                                 if self.stats["popular_interactions"] else "unknown",
            "average_session_length": sum(s.message_count for s in self.active_sessions.values()) / 
                                    max(1, len(self.active_sessions)),
            "peak_activity_hours": self._calculate_peak_hours()
        }
    
    def _calculate_peak_hours(self) -> List[int]:
        """计算高峰时段"""
        hour_counts = {}
        
        for session in self.interaction_history:
            start_time = datetime.fromisoformat(session["start_time"])
            hour = start_time.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        return sorted(hour_counts.keys(), key=hour_counts.get, reverse=True)[:3]
    
    def _analyze_satisfaction_trends(self) -> Dict[str, Any]:
        """分析满意度趋势"""
        # 这里应该从实际数据计算
        return {
            "current_score": 0.85,
            "trend": "improving",
            "factors": ["响应速度", "准确性", "个性化"]
        }
    
    def _calculate_brand_consistency_score(self) -> float:
        """计算品牌一致性分数"""
        if not self.active_sessions:
            return 0.9  # 默认分数
        
        total_consistency = 0
        for session in self.active_sessions:
            # 这里应该计算实际品牌一致性
            total_consistency += 0.9  # 模拟分数
        
        return total_consistency / len(self.active_sessions)
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于质量分数的建议
        avg_quality = sum(s.quality_score for s in self.active_sessions.values()) / max(1, len(self.active_sessions))
        
        if avg_quality < 0.7:
            recommendations.append("建议提高响应质量和准确性")
        
        # 基于品牌一致性的建议
        brand_score = self._calculate_brand_consistency_score()
        if brand_score < 0.8:
            recommendations.append("建议加强品牌元素的一致性使用")
        
        # 基于用户满意度的建议
        if not recommendations:
            recommendations.append("当前交互质量良好，继续保持")
        
        return recommendations
    
    def optimize_interaction_for_user(self, user_id: str, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """为用户优化交互"""
        try:
            optimization = {
                "recommended_style": "professional",
                "interaction_pace": "normal",
                "technical_depth": "medium",
                "personalization_level": "high",
                "suggested_topics": [],
                "avoid_topics": []
            }
            
            # 根据用户偏好调整
            if user_preferences.get("greeting_style") == "casual":
                optimization["recommended_style"] = "friendly"
            elif user_preferences.get("greeting_style") == "formal":
                optimization["recommended_style"] = "formal"
            
            # 根据技术水平调整
            tech_level = user_preferences.get("technical_level", "medium")
            optimization["technical_depth"] = tech_level
            
            # 根据兴趣推荐话题
            interests = user_preferences.get("interests", [])
            optimization["suggested_topics"] = interests[:3]  # 前3个兴趣
            
            return optimization
            
        except Exception as e:
            print(f"交互优化失败: {e}")
            return {}