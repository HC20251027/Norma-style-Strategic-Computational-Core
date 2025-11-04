#!/usr/bin/env python3
"""
对话视图组件
提供对话历史的可视化和管理功能

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import NormaLogger

class ViewMode(Enum):
    """视图模式枚举"""
    CHAT = "chat"
    TIMELINE = "timeline"
    TREE = "tree"
    ANALYTICS = "analytics"

class FilterType(Enum):
    """过滤器类型枚举"""
    DATE_RANGE = "date_range"
    SENDER = "sender"
    MESSAGE_TYPE = "message_type"
    KEYWORD = "keyword"
    SESSION = "session"

@dataclass
class ConversationAnalytics:
    """对话分析数据"""
    session_id: str
    total_messages: int
    user_messages: int
    assistant_messages: int
    avg_response_time: float
    message_frequency: Dict[str, int]  # 按小时统计
    common_topics: List[str]
    sentiment_analysis: Dict[str, float]
    activity_patterns: Dict[str, Any]
    engagement_metrics: Dict[str, float]

class ConversationView:
    """对话视图类"""
    
    def __init__(self, chat_interface):
        """初始化对话视图
        
        Args:
            chat_interface: 聊天界面实例
        """
        self.chat_interface = chat_interface
        self.logger = NormaLogger("conversation_view")
        
        # 视图配置
        self.view_config = {
            "default_mode": ViewMode.CHAT,
            "messages_per_page": 50,
            "enable_analytics": True,
            "enable_filtering": True,
            "enable_search": True,
            "auto_refresh": True,
            "refresh_interval": 30  # 秒
        }
        
        # 当前视图状态
        self.current_view_mode = ViewMode.CHAT
        self.current_filters: Dict[FilterType, Any] = {}
        self.search_query: str = ""
        self.current_page: int = 1
        
        # 缓存
        self.analytics_cache: Dict[str, ConversationAnalytics] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        
        # 视图组件
        self.view_components = {
            "message_list": None,
            "filter_panel": None,
            "search_bar": None,
            "pagination": None,
            "analytics_panel": None
        }
    
    def set_view_mode(self, mode: ViewMode) -> None:
        """设置视图模式"""
        self.current_view_mode = mode
        self.logger.info(f"切换视图模式: {mode.value}")
    
    def set_filters(self, filters: Dict[FilterType, Any]) -> None:
        """设置过滤器"""
        self.current_filters.update(filters)
        self.current_page = 1  # 重置页码
        self.logger.info(f"设置过滤器: {filters}")
    
    def clear_filters(self) -> None:
        """清空过滤器"""
        self.current_filters.clear()
        self.current_page = 1
        self.logger.info("清空过滤器")
    
    def set_search_query(self, query: str) -> None:
        """设置搜索查询"""
        self.search_query = query
        self.current_page = 1
        self.logger.info(f"设置搜索查询: {query}")
    
    def get_filtered_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """获取过滤后的消息"""
        
        messages = self.chat_interface.get_session_messages(session_id, limit)
        
        # 应用过滤器
        for filter_type, filter_value in self.current_filters.items():
            messages = self._apply_filter(messages, filter_type, filter_value)
        
        # 应用搜索查询
        if self.search_query:
            messages = self._apply_search(messages, self.search_query)
        
        # 分页
        page_size = self.view_config["messages_per_page"]
        start_idx = (self.current_page - 1) * page_size
        end_idx = start_idx + page_size
        
        return messages[start_idx:end_idx]
    
    def _apply_filter(
        self,
        messages: List[Dict[str, Any]],
        filter_type: FilterType,
        filter_value: Any
    ) -> List[Dict[str, Any]]:
        """应用过滤器"""
        
        if filter_type == FilterType.DATE_RANGE:
            start_date, end_date = filter_value
            filtered = []
            for msg in messages:
                msg_time = datetime.fromisoformat(msg["timestamp"])
                if start_date <= msg_time <= end_date:
                    filtered.append(msg)
            return filtered
        
        elif filter_type == FilterType.SENDER:
            return [msg for msg in messages if msg["sender"] == filter_value]
        
        elif filter_type == FilterType.MESSAGE_TYPE:
            return [msg for msg in messages if msg["message_type"] == filter_value]
        
        elif filter_type == FilterType.KEYWORD:
            keyword = filter_value.lower()
            return [
                msg for msg in messages
                if keyword in msg["content"].lower()
            ]
        
        elif filter_type == FilterType.SESSION:
            return [msg for msg in messages if msg.get("metadata", {}).get("session_id") == filter_value]
        
        return messages
    
    def _apply_search(self, messages: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """应用搜索"""
        
        query_lower = query.lower()
        results = []
        
        for msg in messages:
            # 搜索内容
            if query_lower in msg["content"].lower():
                results.append(msg)
                continue
            
            # 搜索元数据
            metadata = msg.get("metadata", {})
            for key, value in metadata.items():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(msg)
                    break
        
        return results
    
    def render_chat_view(self, session_id: str) -> Dict[str, Any]:
        """渲染聊天视图"""
        
        messages = self.get_filtered_messages(session_id)
        
        return {
            "view_mode": "chat",
            "session_id": session_id,
            "messages": messages,
            "pagination": self._render_pagination(len(messages)),
            "filters_applied": len(self.current_filters) > 0,
            "search_query": self.search_query,
            "message_stats": self._calculate_message_stats(messages)
        }
    
    def render_timeline_view(self, session_id: str) -> Dict[str, Any]:
        """渲染时间线视图"""
        
        messages = self.get_filtered_messages(session_id)
        
        # 按时间分组
        timeline_groups = self._group_messages_by_time(messages)
        
        return {
            "view_mode": "timeline",
            "session_id": session_id,
            "timeline_groups": timeline_groups,
            "total_groups": len(timeline_groups),
            "date_range": self._get_date_range(messages),
            "filters_applied": len(self.current_filters) > 0
        }
    
    def render_tree_view(self, session_id: str) -> Dict[str, Any]:
        """渲染树状视图"""
        
        messages = self.get_filtered_messages(session_id)
        
        # 构建对话树
        conversation_tree = self._build_conversation_tree(messages)
        
        return {
            "view_mode": "tree",
            "session_id": session_id,
            "conversation_tree": conversation_tree,
            "total_nodes": self._count_tree_nodes(conversation_tree),
            "depth_levels": self._get_tree_depth(conversation_tree)
        }
    
    def render_analytics_view(self, session_id: str) -> Dict[str, Any]:
        """渲染分析视图"""
        
        # 获取或计算分析数据
        analytics = self._get_analytics(session_id)
        
        return {
            "view_mode": "analytics",
            "session_id": session_id,
            "analytics": analytics.to_dict() if analytics else {},
            "charts": self._generate_charts_data(analytics),
            "insights": self._generate_insights(analytics),
            "recommendations": self._generate_recommendations(analytics)
        }
    
    def _group_messages_by_time(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按时间分组消息"""
        
        groups = []
        current_group = None
        last_date = None
        
        for msg in messages:
            msg_time = datetime.fromisoformat(msg["timestamp"])
            msg_date = msg_time.date()
            
            # 检查是否需要新组
            if last_date != msg_date or current_group is None:
                if current_group:
                    groups.append(current_group)
                
                current_group = {
                    "date": msg_date.isoformat(),
                    "date_display": msg_date.strftime("%Y年%m月%d日"),
                    "messages": [msg],
                    "message_count": 1,
                    "time_range": {
                        "start": msg_time.time().isoformat(),
                        "end": msg_time.time().isoformat()
                    }
                }
                last_date = msg_date
            else:
                current_group["messages"].append(msg)
                current_group["message_count"] += 1
                
                # 更新时间范围
                if msg_time.time() < datetime.fromisoformat(current_group["time_range"]["start"]).time():
                    current_group["time_range"]["start"] = msg_time.time().isoformat()
                if msg_time.time() > datetime.fromisoformat(current_group["time_range"]["end"]).time():
                    current_group["time_range"]["end"] = msg_time.time().isoformat()
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _build_conversation_tree(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建对话树"""
        
        # 简化的树结构：按主题分组
        topics = {}
        
        for msg in messages:
            # 简单的主题提取（实际应用中可以使用更复杂的NLP）
            content = msg["content"]
            if len(content) > 20:
                # 提取关键词作为主题
                words = content.split()[:3]  # 取前3个词作为主题
                topic = "_".join(words).lower()
            else:
                topic = "short_message"
            
            if topic not in topics:
                topics[topic] = {
                    "topic": topic,
                    "messages": [],
                    "message_count": 0,
                    "participants": set(),
                    "start_time": None,
                    "end_time": None
                }
            
            topics[topic]["messages"].append(msg)
            topics[topic]["message_count"] += 1
            topics[topic]["participants"].add(msg["sender"])
            
            msg_time = datetime.fromisoformat(msg["timestamp"])
            if topics[topic]["start_time"] is None or msg_time < topics[topic]["start_time"]:
                topics[topic]["start_time"] = msg_time
            if topics[topic]["end_time"] is None or msg_time > topics[topic]["end_time"]:
                topics[topic]["end_time"] = msg_time
        
        # 转换集合为列表
        for topic_data in topics.values():
            topic_data["participants"] = list(topic_data["participants"])
            topic_data["start_time"] = topic_data["start_time"].isoformat() if topic_data["start_time"] else None
            topic_data["end_time"] = topic_data["end_time"].isoformat() if topic_data["end_time"] else None
        
        return {
            "topics": list(topics.values()),
            "total_topics": len(topics),
            "total_messages": len(messages)
        }
    
    def _count_tree_nodes(self, tree: Dict[str, Any]) -> int:
        """计算树节点数量"""
        return tree["total_topics"]
    
    def _get_tree_depth(self, tree: Dict[str, Any]) -> int:
        """获取树深度"""
        return 2  # 简化处理：主题 -> 消息
    
    def _get_analytics(self, session_id: str) -> Optional[ConversationAnalytics]:
        """获取分析数据"""
        
        # 检查缓存
        if session_id in self.analytics_cache:
            expiry_time = self.cache_expiry.get(session_id)
            if expiry_time and datetime.now() < expiry_time:
                return self.analytics_cache[session_id]
        
        # 计算分析数据
        messages = self.chat_interface.get_session_messages(session_id)
        
        if not messages:
            return None
        
        analytics = self._calculate_analytics(session_id, messages)
        
        # 缓存结果（30分钟过期）
        self.analytics_cache[session_id] = analytics
        self.cache_expiry[session_id] = datetime.now() + timedelta(minutes=30)
        
        return analytics
    
    def _calculate_analytics(self, session_id: str, messages: List[Dict[str, Any]]) -> ConversationAnalytics:
        """计算分析数据"""
        
        total_messages = len(messages)
        user_messages = [msg for msg in messages if msg["sender"] == "user"]
        assistant_messages = [msg for msg in messages if msg["sender"] == "assistant"]
        
        # 计算响应时间（简化处理）
        response_times = []
        for i in range(1, len(messages), 2):
            if i < len(messages):
                # 假设用户消息后紧跟助手消息
                user_time = datetime.fromisoformat(messages[i-1]["timestamp"])
                assistant_time = datetime.fromisoformat(messages[i]["timestamp"])
                response_time = (assistant_time - user_time).total_seconds()
                response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # 消息频率统计
        message_frequency = {}
        for msg in messages:
            msg_time = datetime.fromisoformat(msg["timestamp"])
            hour_key = msg_time.strftime("%H:00")
            message_frequency[hour_key] = message_frequency.get(hour_key, 0) + 1
        
        # 常见主题（简化处理）
        common_topics = self._extract_common_topics(messages)
        
        # 情感分析（简化处理）
        sentiment_analysis = self._analyze_sentiment(messages)
        
        # 活动模式
        activity_patterns = {
            "peak_hours": self._find_peak_hours(message_frequency),
            "session_duration": self._calculate_session_duration(messages),
            "message_velocity": self._calculate_message_velocity(messages)
        }
        
        # 参与度指标
        engagement_metrics = {
            "user_engagement": len(user_messages) / total_messages if total_messages > 0 else 0,
            "response_rate": len(assistant_messages) / len(user_messages) if user_messages else 0,
            "conversation_depth": len(set(msg["sender"] for msg in messages))
        }
        
        return ConversationAnalytics(
            session_id=session_id,
            total_messages=total_messages,
            user_messages=len(user_messages),
            assistant_messages=len(assistant_messages),
            avg_response_time=avg_response_time,
            message_frequency=message_frequency,
            common_topics=common_topics,
            sentiment_analysis=sentiment_analysis,
            activity_patterns=activity_patterns,
            engagement_metrics=engagement_metrics
        )
    
    def _extract_common_topics(self, messages: List[Dict[str, Any]]) -> List[str]:
        """提取常见主题"""
        
        # 简化处理：提取关键词
        all_words = []
        for msg in messages:
            words = msg["content"].split()
            all_words.extend([word for word in words if len(word) > 3])
        
        # 统计词频
        word_count = {}
        for word in all_words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 返回最常见的词
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5]]
    
    def _analyze_sentiment(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """情感分析（简化处理）"""
        
        # 简化情感分析：基于关键词
        positive_words = ["好", "棒", "优秀", "喜欢", "感谢", "good", "great", "excellent", "thanks"]
        negative_words = ["坏", "差", "糟糕", "讨厌", "问题", "bad", "terrible", "hate", "problem"]
        
        positive_count = 0
        negative_count = 0
        
        for msg in messages:
            content = msg["content"].lower()
            for word in positive_words:
                if word in content:
                    positive_count += 1
            for word in negative_words:
                if word in content:
                    negative_count += 1
        
        total = positive_count + negative_count
        if total == 0:
            return {"positive": 0.5, "negative": 0.5, "neutral": 1.0}
        
        return {
            "positive": positive_count / total,
            "negative": negative_count / total,
            "neutral": 1.0 - (positive_count + negative_count) / total
        }
    
    def _find_peak_hours(self, message_frequency: Dict[str, int]) -> List[str]:
        """找到高峰时段"""
        
        if not message_frequency:
            return []
        
        sorted_hours = sorted(message_frequency.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]
    
    def _calculate_session_duration(self, messages: List[Dict[str, Any]]) -> float:
        """计算会话持续时间"""
        
        if len(messages) < 2:
            return 0
        
        start_time = datetime.fromisoformat(messages[0]["timestamp"])
        end_time = datetime.fromisoformat(messages[-1]["timestamp"])
        
        return (end_time - start_time).total_seconds()
    
    def _calculate_message_velocity(self, messages: List[Dict[str, Any]]) -> float:
        """计算消息速度（消息/分钟）"""
        
        duration = self._calculate_session_duration(messages)
        if duration == 0:
            return 0
        
        return len(messages) / (duration / 60)
    
    def _calculate_message_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算消息统计"""
        
        if not messages:
            return {}
        
        user_count = len([msg for msg in messages if msg["sender"] == "user"])
        assistant_count = len([msg for msg in messages if msg["sender"] == "assistant"])
        
        # 计算平均消息长度
        total_length = sum(len(msg["content"]) for msg in messages)
        avg_length = total_length / len(messages)
        
        return {
            "total": len(messages),
            "user": user_count,
            "assistant": assistant_count,
            "avg_length": round(avg_length, 2),
            "date_range": {
                "start": messages[0]["timestamp"],
                "end": messages[-1]["timestamp"]
            }
        }
    
    def _render_pagination(self, total_items: int) -> Dict[str, Any]:
        """渲染分页信息"""
        
        page_size = self.view_config["messages_per_page"]
        total_pages = (total_items + page_size - 1) // page_size
        
        return {
            "current_page": self.current_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "page_size": page_size,
            "has_previous": self.current_page > 1,
            "has_next": self.current_page < total_pages
        }
    
    def _get_date_range(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """获取日期范围"""
        
        if not messages:
            return None
        
        start_date = datetime.fromisoformat(messages[0]["timestamp"]).date().isoformat()
        end_date = datetime.fromisoformat(messages[-1]["timestamp"]).date().isoformat()
        
        return {
            "start": start_date,
            "end": end_date
        }
    
    def _generate_charts_data(self, analytics: Optional[ConversationAnalytics]) -> Dict[str, Any]:
        """生成图表数据"""
        
        if not analytics:
            return {}
        
        return {
            "message_frequency": {
                "type": "line",
                "data": analytics.message_frequency,
                "title": "消息频率分布"
            },
            "sentiment_analysis": {
                "type": "pie",
                "data": analytics.sentiment_analysis,
                "title": "情感分析"
            },
            "engagement_metrics": {
                "type": "bar",
                "data": analytics.engagement_metrics,
                "title": "参与度指标"
            }
        }
    
    def _generate_insights(self, analytics: Optional[ConversationAnalytics]) -> List[str]:
        """生成洞察"""
        
        if not analytics:
            return []
        
        insights = []
        
        # 响应时间洞察
        if analytics.avg_response_time < 2:
            insights.append("诺玛的响应速度很快，用户体验良好")
        elif analytics.avg_response_time > 10:
            insights.append("响应时间较长，可能需要优化处理速度")
        
        # 参与度洞察
        if analytics.engagement_metrics["user_engagement"] > 0.6:
            insights.append("用户参与度很高，对话质量优秀")
        
        # 活动模式洞察
        peak_hours = analytics.activity_patterns["peak_hours"]
        if peak_hours:
            insights.append(f"最活跃时段: {', '.join(peak_hours)}")
        
        return insights
    
    def _generate_recommendations(self, analytics: Optional[ConversationAnalytics]) -> List[str]:
        """生成建议"""
        
        if not analytics:
            return []
        
        recommendations = []
        
        # 基于分析结果生成建议
        if analytics.avg_response_time > 5:
            recommendations.append("建议优化系统响应速度")
        
        if analytics.engagement_metrics["user_engagement"] < 0.4:
            recommendations.append("建议改进对话内容以提高用户参与度")
        
        if len(analytics.common_topics) < 3:
            recommendations.append("建议丰富对话主题，提供更多样化的交互")
        
        return recommendations
    
    def export_view_data(self, session_id: str, format: str = "json") -> Optional[str]:
        """导出视图数据"""
        
        view_data = {}
        
        if self.current_view_mode == ViewMode.CHAT:
            view_data = self.render_chat_view(session_id)
        elif self.current_view_mode == ViewMode.TIMELINE:
            view_data = self.render_timeline_view(session_id)
        elif self.current_view_mode == ViewMode.TREE:
            view_data = self.render_tree_view(session_id)
        elif self.current_view_mode == ViewMode.ANALYTICS:
            view_data = self.render_analytics_view(session_id)
        
        if format == "json":
            return json.dumps(view_data, ensure_ascii=False, indent=2, default=str)
        
        return None
    
    def get_view_summary(self, session_id: str) -> Dict[str, Any]:
        """获取视图摘要"""
        
        messages = self.chat_interface.get_session_messages(session_id)
        
        return {
            "session_id": session_id,
            "view_mode": self.current_view_mode.value,
            "total_messages": len(messages),
            "filtered_messages": len(self.get_filtered_messages(session_id)),
            "filters_active": len(self.current_filters),
            "search_active": bool(self.search_query),
            "current_page": self.current_page,
            "total_pages": (len(messages) + self.view_config["messages_per_page"] - 1) // self.view_config["messages_per_page"]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        return {
            "component": "conversation_view",
            "view_mode": self.current_view_mode.value,
            "filters_active": len(self.current_filters),
            "search_query": self.search_query,
            "current_page": self.current_page,
            "analytics_cached": len(self.analytics_cache),
            "view_config": self.view_config
        }