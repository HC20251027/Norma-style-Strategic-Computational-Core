"""
个性化学习系统
基于用户交互历史的个性化学习和适应能力
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import statistics


class LearningType(Enum):
    """学习类型"""
    PREFERENCE_LEARNING = "preference_learning"    # 偏好学习
    BEHAVIOR_LEARNING = "behavior_learning"        # 行为学习
    ADAPTIVE_RESPONSE = "adaptive_response"        # 适应性响应
    PATTERN_DISCOVERY = "pattern_discovery"        # 模式发现
    KNOWLEDGE_GRAPH_UPDATE = "knowledge_graph_update"  # 知识图谱更新


class InteractionType(Enum):
    """交互类型"""
    QUERY = "query"                    # 查询
    FEEDBACK = "feedback"              # 反馈
    CORRECTION = "correction"          # 纠正
    PREFERENCE = "preference"          # 偏好表达
    CLARIFICATION = "clarification"    # 澄清
    FOLLOW_UP = "follow_up"            # 后续问题


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    preferences: Dict[str, float]      # 偏好分数 0-1
    interaction_patterns: Dict[str, int]  # 交互模式计数
    knowledge_gaps: List[str]          # 知识空白
    learning_rate: float               # 学习速度
    adaptation_history: List[Dict[str, Any]]  # 适应历史
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.fromisoformat(self.created_at)
        if not isinstance(self.updated_at, datetime):
            self.updated_at = datetime.fromisoformat(self.updated_at)


@dataclass
class LearningRecord:
    """学习记录"""
    id: str
    user_id: str
    interaction_type: InteractionType
    content: str
    context: Dict[str, Any]
    feedback_score: Optional[float]    # 反馈分数 -1到1
    learning_outcome: Dict[str, Any]
    timestamp: datetime
    
    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(self.timestamp)


class PersonalizedLearningSystem:
    """个性化学习系统"""
    
    def __init__(self, memory_manager, config: Dict[str, Any]):
        self.memory_manager = memory_manager
        self.config = config
        
        # 学习配置
        self.learning_rate = config.get('learning_rate', 0.1)
        self.adaptation_threshold = config.get('adaptation_threshold', 0.7)
        self.pattern_window = config.get('pattern_window', 100)  # 模式分析窗口
        
        # 用户画像存储
        self.user_profiles: Dict[str, UserProfile] = {}
        self.learning_records: List[LearningRecord] = []
        
        # 模式识别
        self.interaction_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.preference_evolution: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # 学习统计
        self.learning_stats = {
            'total_interactions': 0,
            'successful_adaptations': 0,
            'preference_changes': 0,
            'pattern_discoveries': 0
        }
    
    async def initialize(self):
        """初始化个性化学习系统"""
        await self._load_user_profiles()
        print("个性化学习系统初始化完成")
    
    async def _load_user_profiles(self):
        """加载用户画像"""
        # 从记忆系统中加载用户画像
        # 这里简化实现，实际应该从数据库加载
        pass
    
    async def learn_from_interaction(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """从交互中学习"""
        
        # 创建学习记录
        learning_record = LearningRecord(
            id=f"learning_{datetime.now().timestamp()}",
            user_id=user_id,
            interaction_type=InteractionType.QUERY,
            content=user_input,
            context=context or {},
            feedback_score=None,
            learning_outcome={},
            timestamp=datetime.now()
        )
        
        # 更新用户画像
        await self._update_user_profile(user_id, user_input, context)
        
        # 学习偏好
        preference_updates = await self._learn_preferences(user_id, user_input, context)
        
        # 学习行为模式
        behavior_updates = await self._learn_behavior_patterns(user_id, user_input, context)
        
        # 适应响应策略
        adaptation_updates = await self._adapt_response_strategy(user_id, user_input, context)
        
        # 发现模式
        pattern_discoveries = await self._discover_patterns(user_id)
        
        # 更新知识图谱
        kg_updates = await self._update_knowledge_from_interaction(user_input, context)
        
        learning_outcome = {
            'preference_updates': preference_updates,
            'behavior_updates': behavior_updates,
            'adaptation_updates': adaptation_updates,
            'pattern_discoveries': pattern_discoveries,
            'knowledge_graph_updates': kg_updates
        }
        
        learning_record.learning_outcome = learning_outcome
        self.learning_records.append(learning_record)
        
        # 更新统计
        self.learning_stats['total_interactions'] += 1
        
        return learning_outcome
    
    async def _update_user_profile(self, user_id: str, content: str, context: Optional[Dict[str, Any]]):
        """更新用户画像"""
        
        if user_id not in self.user_profiles:
            # 创建新用户画像
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences={},
                interaction_patterns={},
                knowledge_gaps=[],
                learning_rate=self.learning_rate,
                adaptation_history=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        profile = self.user_profiles[user_id]
        
        # 更新交互模式
        interaction_type = self._classify_interaction_type(content)
        profile.interaction_patterns[interaction_type] = profile.interaction_patterns.get(interaction_type, 0) + 1
        
        # 检测知识空白
        knowledge_gaps = self._detect_knowledge_gaps(content)
        for gap in knowledge_gaps:
            if gap not in profile.knowledge_gaps:
                profile.knowledge_gaps.append(gap)
        
        profile.updated_at = datetime.now()
    
    def _classify_interaction_type(self, content: str) -> str:
        """分类交互类型"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['什么', '什么', 'how', 'what', 'why', 'when', 'where']):
            return 'question'
        elif any(word in content_lower for word in ['喜欢', '偏好', 'prefer', 'like', 'love']):
            return 'preference'
        elif any(word in content_lower for word in ['不对', '错误', 'wrong', 'incorrect', 'error']):
            return 'correction'
        elif any(word in content_lower for word in ['更', '最好', 'better', 'best', 'prefer']):
            return 'comparison'
        else:
            return 'general'
    
    def _detect_knowledge_gaps(self, content: str) -> List[str]:
        """检测知识空白"""
        gaps = []
        
        # 简单的知识空白检测
        question_words = ['什么', '如何', '怎么', '为什么', 'who', 'what', 'how', 'why', 'when', 'where']
        
        for word in question_words:
            if word in content.lower():
                gaps.append(f"question_about_{word}")
        
        # 检测未知的概念
        unknown_indicators = ['不知道', '不清楚', '不太明白', "don't know", 'unclear', 'confused']
        
        for indicator in unknown_indicators:
            if indicator in content.lower():
                gaps.append('concept_clarification_needed')
        
        return gaps
    
    async def _learn_preferences(
        self,
        user_id: str,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """学习用户偏好"""
        
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {}
        
        preference_updates = {}
        
        # 学习内容偏好
        content_preferences = self._extract_content_preferences(content)
        for topic, score in content_preferences.items():
            current_score = profile.preferences.get(topic, 0.5)
            new_score = current_score + self.learning_rate * (score - current_score)
            profile.preferences[topic] = max(0.0, min(1.0, new_score))
            preference_updates[topic] = {
                'old_score': current_score,
                'new_score': new_score,
                'change': new_score - current_score
            }
        
        # 学习响应风格偏好
        response_preferences = self._extract_response_preferences(content, context)
        for style, score in response_preferences.items():
            current_score = profile.preferences.get(f"response_{style}", 0.5)
            new_score = current_score + self.learning_rate * (score - current_score)
            profile.preferences[f"response_{style}"] = max(0.0, min(1.0, new_score))
            preference_updates[f"response_{style}"] = {
                'old_score': current_score,
                'new_score': new_score,
                'change': new_score - current_score
            }
        
        # 记录偏好变化
        timestamp = datetime.now()
        for topic, score in profile.preferences.items():
            self.preference_evolution[user_id].append((timestamp, score))
        
        return preference_updates
    
    def _extract_content_preferences(self, content: str) -> Dict[str, float]:
        """提取内容偏好"""
        preferences = {}
        
        # 主题偏好检测
        topics = {
            'technology': ['技术', '科技', '编程', '代码', 'computer', 'tech', 'programming'],
            'science': ['科学', '研究', '实验', 'science', 'research', 'study'],
            'art': ['艺术', '绘画', '音乐', 'art', 'music', 'creative'],
            'business': ['商业', '经济', '管理', 'business', 'economy', 'management'],
            'education': ['教育', '学习', '课程', 'education', 'learning', 'course'],
            'entertainment': ['娱乐', '游戏', '电影', 'entertainment', 'game', 'movie']
        }
        
        content_lower = content.lower()
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                preferences[topic] = min(1.0, score / len(keywords))
        
        return preferences
    
    def _extract_response_preferences(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """提取响应风格偏好"""
        preferences = {}
        
        content_lower = content.lower()
        
        # 详细程度偏好
        if len(content) > 100 or any(word in content_lower for word in ['详细', '详细说明', '详细解释', 'detailed', 'explain']):
            preferences['detailed'] = 0.8
        elif len(content) < 50 or any(word in content_lower for word in ['简单', '简要', 'brief', 'simple', 'summary']):
            preferences['brief'] = 0.8
        
        # 风格偏好
        if any(word in content_lower for word in ['正式', '专业', 'formal', 'professional']):
            preferences['formal'] = 0.7
        
        if any(word in content_lower for word in ['友好', '亲切', 'friendly', 'casual']):
            preferences['casual'] = 0.7
        
        # 互动性偏好
        if any(word in content_lower for word in ['问题', '建议', 'question', 'suggestion']):
            preferences['interactive'] = 0.6
        
        return preferences
    
    async def _learn_behavior_patterns(
        self,
        user_id: str,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """学习行为模式"""
        
        behavior_updates = {}
        
        # 记录交互模式
        interaction_record = {
            'timestamp': datetime.now(),
            'content_length': len(content),
            'content_type': self._classify_interaction_type(content),
            'context_keys': list(context.keys()) if context else []
        }
        
        self.interaction_patterns[user_id].append(interaction_record)
        
        # 保持窗口大小
        if len(self.interaction_patterns[user_id]) > self.pattern_window:
            self.interaction_patterns[user_id] = self.interaction_patterns[user_id][-self.pattern_window:]
        
        # 分析模式
        patterns = self._analyze_interaction_patterns(user_id)
        behavior_updates['patterns'] = patterns
        
        return behavior_updates
    
    def _analyze_interaction_patterns(self, user_id: str) -> Dict[str, Any]:
        """分析交互模式"""
        
        patterns = self.interaction_patterns.get(user_id, [])
        if not patterns:
            return {}
        
        # 计算平均内容长度
        content_lengths = [p['content_length'] for p in patterns]
        avg_length = statistics.mean(content_lengths) if content_lengths else 0
        
        # 统计交互类型分布
        type_distribution = Counter(p['content_type'] for p in patterns)
        
        # 活动时间模式
        hours = [p['timestamp'].hour for p in patterns]
        active_hours = Counter(hours)
        
        # 内容复杂度趋势
        complexity_trend = []
        for i in range(0, len(patterns), 10):  # 每10个交互计算一次
            batch = patterns[i:i+10]
            avg_complexity = statistics.mean(p['content_length'] for p in batch)
            complexity_trend.append(avg_complexity)
        
        return {
            'average_content_length': avg_length,
            'interaction_type_distribution': dict(type_distribution),
            'most_active_hours': dict(active_hours.most_common(5)),
            'complexity_trend': complexity_trend,
            'total_interactions': len(patterns)
        }
    
    async def _adapt_response_strategy(
        self,
        user_id: str,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """适应响应策略"""
        
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {}
        
        adaptation_updates = {}
        
        # 基于偏好调整响应策略
        response_strategy = self._generate_response_strategy(profile, content)
        adaptation_updates['response_strategy'] = response_strategy
        
        # 调整学习参数
        if len(self.interaction_patterns.get(user_id, [])) > 10:
            # 根据用户活跃度调整学习率
            recent_interactions = self.interaction_patterns[user_id][-10:]
            activity_level = len(recent_interactions)
            
            if activity_level > 7:  # 高活跃度
                profile.learning_rate = min(0.2, profile.learning_rate * 1.1)
            elif activity_level < 3:  # 低活跃度
                profile.learning_rate = max(0.05, profile.learning_rate * 0.9)
        
        adaptation_updates['learning_rate'] = profile.learning_rate
        
        return adaptation_updates
    
    def _generate_response_strategy(self, profile: UserProfile, content: str) -> Dict[str, Any]:
        """生成响应策略"""
        
        strategy = {
            'detail_level': 'medium',
            'response_style': 'neutral',
            'interactivity_level': 'medium',
            'personalization_factors': []
        }
        
        # 基于偏好设置详细程度
        if profile.preferences.get('detailed', 0) > 0.7:
            strategy['detail_level'] = 'high'
        elif profile.preferences.get('brief', 0) > 0.7:
            strategy['detail_level'] = 'low'
        
        # 基于偏好设置响应风格
        if profile.preferences.get('response_formal', 0) > 0.6:
            strategy['response_style'] = 'formal'
        elif profile.preferences.get('response_casual', 0) > 0.6:
            strategy['response_style'] = 'casual'
        
        # 基于交互模式调整互动性
        interaction_patterns = self.interaction_patterns.get(profile.user_id, [])
        if interaction_patterns:
            recent_types = [p['content_type'] for p in interaction_patterns[-5:]]
            if 'question' in recent_types:
                strategy['interactivity_level'] = 'high'
        
        return strategy
    
    async def _discover_patterns(self, user_id: str) -> Dict[str, Any]:
        """发现模式"""
        
        patterns = self.interaction_patterns.get(user_id, [])
        discoveries = {}
        
        if len(patterns) < 5:
            return discoveries
        
        # 发现时间模式
        time_patterns = self._discover_time_patterns(patterns)
        if time_patterns:
            discoveries['time_patterns'] = time_patterns
            self.learning_stats['pattern_discoveries'] += 1
        
        # 发现内容模式
        content_patterns = self._discover_content_patterns(patterns)
        if content_patterns:
            discoveries['content_patterns'] = content_patterns
            self.learning_stats['pattern_discoveries'] += 1
        
        # 发现交互节奏模式
        rhythm_patterns = self._discover_rhythm_patterns(patterns)
        if rhythm_patterns:
            discoveries['rhythm_patterns'] = rhythm_patterns
            self.learning_stats['pattern_discoveries'] += 1
        
        return discoveries
    
    def _discover_time_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """发现时间模式"""
        
        if not patterns:
            return {}
        
        # 分析活动时间分布
        hours = [p['timestamp'].hour for p in patterns]
        hour_counts = Counter(hours)
        
        # 找出最活跃的时间段
        most_active_hour = hour_counts.most_common(1)[0] if hour_counts else None
        
        # 分析工作日vs周末模式
        weekdays = [p['timestamp'].weekday() for p in patterns]
        weekday_counts = Counter(weekdays)
        
        return {
            'most_active_hour': most_active_hour,
            'hour_distribution': dict(hour_counts),
            'weekday_distribution': dict(weekday_counts),
            'activity_consistency': self._calculate_consistency(hours)
        }
    
    def _discover_content_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """发现内容模式"""
        
        # 分析内容长度变化
        lengths = [p['content_length'] for p in patterns]
        
        # 计算长度趋势
        if len(lengths) > 1:
            trend = np.polyfit(range(len(lengths)), lengths, 1)[0]
        else:
            trend = 0
        
        # 分析交互类型变化
        types = [p['content_type'] for p in patterns]
        type_sequence = ' -> '.join(types[-5:])  # 最近5个交互的类型序列
        
        return {
            'length_trend': trend,
            'length_variance': np.var(lengths),
            'recent_type_sequence': type_sequence,
            'dominant_type': Counter(types).most_common(1)[0] if types else None
        }
    
    def _discover_rhythm_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """发现交互节奏模式"""
        
        if len(patterns) < 2:
            return {}
        
        # 计算交互间隔
        intervals = []
        for i in range(1, len(patterns)):
            interval = (patterns[i]['timestamp'] - patterns[i-1]['timestamp']).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return {}
        
        # 分析节奏模式
        avg_interval = statistics.mean(intervals)
        interval_variance = np.var(intervals)
        
        # 判断节奏类型
        if interval_variance < 3600:  # 1小时方差
            rhythm_type = 'consistent'
        elif avg_interval < 1800:  # 30分钟
            rhythm_type = 'frequent'
        else:
            rhythm_type = 'sporadic'
        
        return {
            'average_interval_minutes': avg_interval / 60,
            'interval_variance': interval_variance,
            'rhythm_type': rhythm_type,
            'consistency_score': 1 / (1 + interval_variance / 3600)  # 归一化一致性分数
        }
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """计算一致性分数"""
        if len(values) < 2:
            return 1.0
        
        mean_val = statistics.mean(values)
        variance = np.var(values)
        
        # 一致性分数：方差越小，分数越高
        consistency = 1 / (1 + variance / (mean_val + 1))
        return consistency
    
    async def _update_knowledge_from_interaction(
        self,
        content: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """从交互中更新知识"""
        
        # 这里可以集成知识图谱更新逻辑
        # 简化实现，返回空字典
        return {
            'entities_extracted': 0,
            'relations_discovered': 0,
            'knowledge_updated': False
        }
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取个性化推荐"""
        
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {'recommendations': [], 'reason': 'no_profile'}
        
        recommendations = []
        
        # 基于偏好推荐内容类型
        top_preferences = sorted(
            profile.preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for topic, score in top_preferences:
            if score > 0.6:
                recommendations.append({
                    'type': 'content_topic',
                    'topic': topic,
                    'confidence': score,
                    'reason': f'用户对{topic}有较高偏好'
                })
        
        # 基于行为模式推荐交互方式
        patterns = self.interaction_patterns.get(user_id, [])
        if patterns:
            dominant_type = Counter(p['content_type'] for p in patterns).most_common(1)[0]
            recommendations.append({
                'type': 'interaction_style',
                'style': dominant_type[0],
                'confidence': dominant_type[1] / len(patterns),
                'reason': '基于历史交互模式'
            })
        
        # 推荐学习内容
        for gap in profile.knowledge_gaps[:2]:
            recommendations.append({
                'type': 'knowledge_gap',
                'topic': gap,
                'confidence': 0.7,
                'reason': '检测到的知识空白'
            })
        
        return {
            'recommendations': recommendations,
            'profile_summary': {
                'total_interactions': len(patterns),
                'top_preferences': top_preferences,
                'knowledge_gaps_count': len(profile.knowledge_gaps)
            }
        }
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        
        stats = self.learning_stats.copy()
        
        # 添加用户统计
        stats['total_users'] = len(self.user_profiles)
        stats['total_learning_records'] = len(self.learning_records)
        
        # 计算成功率
        if stats['total_interactions'] > 0:
            stats['adaptation_success_rate'] = stats['successful_adaptations'] / stats['total_interactions']
        else:
            stats['adaptation_success_rate'] = 0.0
        
        # 平均偏好变化
        total_preference_changes = sum(
            len(profile.preferences) for profile in self.user_profiles.values()
        )
        stats['avg_preferences_per_user'] = (
            total_preference_changes / len(self.user_profiles) 
            if self.user_profiles else 0
        )
        
        return stats
    
    async def export_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """导出用户画像"""
        
        profile = self.user_profiles.get(user_id)
        if not profile:
            return None
        
        profile_data = asdict(profile)
        
        # 添加学习历史
        user_records = [
            asdict(record) for record in self.learning_records
            if record.user_id == user_id
        ]
        profile_data['learning_history'] = user_records
        
        # 添加交互模式
        profile_data['interaction_patterns_detail'] = self.interaction_patterns.get(user_id, [])
        
        # 添加偏好变化历史
        profile_data['preference_evolution'] = self.preference_evolution.get(user_id, [])
        
        return profile_data
    
    async def clear_learning_data(self, user_id: Optional[str] = None):
        """清理学习数据"""
        
        if user_id:
            # 清理特定用户数据
            if user_id in self.user_profiles:
                del self.user_profiles[user_id]
            if user_id in self.interaction_patterns:
                del self.interaction_patterns[user_id]
            if user_id in self.preference_evolution:
                del self.preference_evolution[user_id]
            
            # 清理学习记录
            self.learning_records = [
                record for record in self.learning_records
                if record.user_id != user_id
            ]
        else:
            # 清理所有数据
            self.user_profiles.clear()
            self.interaction_patterns.clear()
            self.preference_evolution.clear()
            self.learning_records.clear()
        
        print(f"已清理用户 {user_id or '所有'} 的学习数据")