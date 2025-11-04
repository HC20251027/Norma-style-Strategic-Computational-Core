"""
用户行为分析和体验优化模块
负责分析用户行为模式并优化用户体验
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import logging
from collections import defaultdict, deque
import threading
import numpy as np
from pathlib import Path

class UserAction(Enum):
    """用户动作类型"""
    LOGIN = "login"
    LOGOUT = "logout"
    SEND_MESSAGE = "send_message"
    RECEIVE_RESPONSE = "receive_response"
    VIEW_CONVERSATION = "view_conversation"
    SEARCH = "search"
    NAVIGATE = "navigate"
    CLICK = "click"
    SCROLL = "scroll"
    UPLOAD_FILE = "upload_file"
    DOWNLOAD_FILE = "download_file"
    SETTINGS_CHANGE = "settings_change"

class UserSegment(Enum):
    """用户分群"""
    NEW_USER = "new_user"
    ACTIVE_USER = "active_user"
    POWER_USER = "power_user"
    INACTIVE_USER = "inactive_user"
    PREMIUM_USER = "premium_user"

class ExperienceMetric(Enum):
    """体验指标"""
    RESPONSE_TIME = "response_time"
    SATISFACTION_SCORE = "satisfaction_score"
    TASK_COMPLETION_RATE = "task_completion_rate"
    ERROR_RATE = "error_rate"
    ENGAGEMENT_SCORE = "engagement_score"
    RETENTION_RATE = "retention_rate"

@dataclass
class UserSession:
    """用户会话"""
    session_id: str
    user_id: str
    start_time: float
    end_time: Optional[float]
    actions: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

@dataclass
class UserBehavior:
    """用户行为"""
    id: Optional[int]
    user_id: str
    session_id: str
    action_type: UserAction
    timestamp: float
    duration: Optional[float]
    page_url: Optional[str]
    element_id: Optional[str]
    metadata: Dict[str, Any] = None

@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    segment: UserSegment
    total_sessions: int
    total_actions: int
    avg_session_duration: float
    preferred_features: List[str]
    behavior_patterns: Dict[str, Any]
    last_active: float
    satisfaction_score: float

@dataclass
class ExperienceInsight:
    """体验洞察"""
    metric_name: str
    current_value: float
    trend: str  # "improving", "declining", "stable"
    insights: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: float

class UserAnalytics:
    """用户行为分析系统"""
    
    def __init__(self, db_path: str = "user_analytics.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        
        # 用户数据缓存
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.behavior_cache: deque = deque(maxlen=10000)
        
        # 分析配置
        self.config = {
            "session_timeout": 1800,  # 30分钟
            "min_session_duration": 60,  # 1分钟
            "analysis_interval": 300,  # 5分钟
            "retention_period": 30,  # 30天
            "segment_thresholds": {
                "new_user": {"sessions": 1, "days": 7},
                "active_user": {"sessions": 5, "days": 30},
                "power_user": {"sessions": 20, "days": 90}
            }
        }
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 回调函数
        self.analytics_callbacks: List[Callable] = []
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    action_count INTEGER DEFAULT 0,
                    total_duration REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_behaviors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    duration REAL,
                    page_url TEXT,
                    element_id TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    segment TEXT NOT NULL,
                    total_sessions INTEGER NOT NULL,
                    total_actions INTEGER NOT NULL,
                    avg_session_duration REAL NOT NULL,
                    preferred_features TEXT NOT NULL,
                    behavior_patterns TEXT NOT NULL,
                    last_active REAL NOT NULL,
                    satisfaction_score REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    trend TEXT NOT NULL,
                    insights TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_behaviors_user ON user_behaviors(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_behaviors_session ON user_behaviors(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON experience_metrics(timestamp)")
    
    async def start_analytics(self):
        """启动用户分析"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("启动用户行为分析系统")
        
        # 启动分析任务
        tasks = [
            asyncio.create_task(self._process_user_behaviors()),
            asyncio.create_task(self._update_user_profiles()),
            asyncio.create_task(self._analyze_experience_metrics()),
            asyncio.create_task(self._detect_behavior_patterns()),
            asyncio.create_task(self._generate_insights()),
            asyncio.create_task(self._cleanup_expired_sessions()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop_analytics(self):
        """停止用户分析"""
        self.is_running = False
        self.logger.info("停止用户行为分析系统")
    
    def track_user_action(self, user_id: str, action_type: UserAction, 
                         session_id: str = None, metadata: Dict[str, Any] = None,
                         duration: float = None, page_url: str = None, 
                         element_id: str = None):
        """跟踪用户动作"""
        timestamp = time.time()
        
        # 如果没有会话ID，生成一个新的
        if not session_id:
            session_id = f"{user_id}_{int(timestamp)}"
        
        # 创建或更新会话
        self._update_session(user_id, session_id, timestamp, metadata)
        
        # 创建用户行为记录
        behavior = UserBehavior(
            id=None,
            user_id=user_id,
            session_id=session_id,
            action_type=action_type,
            timestamp=timestamp,
            duration=duration,
            page_url=page_url,
            element_id=element_id,
            metadata=metadata
        )
        
        # 保存到数据库
        self._save_user_behavior(behavior)
        
        # 添加到缓存
        with self.lock:
            self.behavior_cache.append(behavior)
        
        self.logger.debug(f"跟踪用户动作: {user_id} - {action_type.value}")
    
    def _update_session(self, user_id: str, session_id: str, timestamp: float, metadata: Dict = None):
        """更新用户会话"""
        with self.lock:
            if session_id not in self.active_sessions:
                # 创建新会话
                session = UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    start_time=timestamp,
                    end_time=None,
                    actions=[],
                    metadata=metadata or {}
                )
                self.active_sessions[session_id] = session
            else:
                # 更新现有会话
                session = self.active_sessions[session_id]
                session.actions.append({
                    "timestamp": timestamp,
                    "metadata": metadata
                })
    
    def _save_user_behavior(self, behavior: UserBehavior):
        """保存用户行为到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO user_behaviors 
                    (user_id, session_id, action_type, timestamp, duration, page_url, element_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    behavior.user_id,
                    behavior.session_id,
                    behavior.action_type.value,
                    behavior.timestamp,
                    behavior.duration,
                    behavior.page_url,
                    behavior.element_id,
                    json.dumps(behavior.metadata) if behavior.metadata else None
                ))
                
                behavior.id = cursor.lastrowid
        except Exception as e:
            self.logger.error(f"保存用户行为时出错: {e}")
    
    async def _process_user_behaviors(self):
        """处理用户行为数据"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["analysis_interval"])
                
                # 处理缓存中的行为数据
                with self.lock:
                    behaviors_to_process = list(self.behavior_cache)
                    self.behavior_cache.clear()
                
                for behavior in behaviors_to_process:
                    await self._process_single_behavior(behavior)
                
            except Exception as e:
                self.logger.error(f"处理用户行为时出错: {e}")
                await asyncio.sleep(60)
    
    async def _process_single_behavior(self, behavior: UserBehavior):
        """处理单个用户行为"""
        try:
            # 更新用户体验指标
            await self._update_experience_metrics(behavior)
            
            # 分析行为模式
            await self._analyze_behavior_pattern(behavior)
            
        except Exception as e:
            self.logger.error(f"处理用户行为 {behavior.id} 时出错: {e}")
    
    async def _update_experience_metrics(self, behavior: UserBehavior):
        """更新体验指标"""
        try:
            # 根据行为类型更新相应的体验指标
            if behavior.action_type == UserAction.RECEIVE_RESPONSE:
                # 更新响应时间指标
                if behavior.duration:
                    await self._record_experience_metric(
                        "response_time",
                        behavior.duration,
                        behavior.user_id,
                        behavior.session_id,
                        {"action_type": behavior.action_type.value}
                    )
            
            elif behavior.action_type == UserAction.SEND_MESSAGE:
                # 更新用户活跃度
                await self._record_experience_metric(
                    "user_activity",
                    1.0,
                    behavior.user_id,
                    behavior.session_id,
                    {"action_type": behavior.action_type.value}
                )
            
            # 更新整体满意度分数
            satisfaction_score = await self._calculate_satisfaction_score(behavior.user_id)
            await self._record_experience_metric(
                "satisfaction_score",
                satisfaction_score,
                behavior.user_id,
                behavior.session_id,
                {"calculated": True}
            )
            
        except Exception as e:
            self.logger.error(f"更新体验指标时出错: {e}")
    
    async def _record_experience_metric(self, metric_name: str, value: float, 
                                      user_id: str = None, session_id: str = None, 
                                      metadata: Dict = None):
        """记录体验指标"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experience_metrics 
                    (metric_name, metric_value, user_id, session_id, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric_name,
                    value,
                    user_id,
                    session_id,
                    time.time(),
                    json.dumps(metadata) if metadata else None
                ))
        except Exception as e:
            self.logger.error(f"记录体验指标时出错: {e}")
    
    async def _calculate_satisfaction_score(self, user_id: str) -> float:
        """计算用户满意度分数"""
        try:
            # 获取用户最近的行为数据
            recent_behaviors = await self._get_recent_user_behaviors(user_id, hours=24)
            
            if not recent_behaviors:
                return 5.0  # 默认分数
            
            # 简单算法：基于行为类型和频率计算满意度
            score_factors = {
                UserAction.SEND_MESSAGE: 1.0,
                UserAction.RECEIVE_RESPONSE: 1.2,
                UserAction.LOGIN: 0.8,
                UserAction.LOGOUT: 0.5,
                UserAction.SEARCH: 0.9,
                UserAction.NAVIGATE: 0.7,
                UserAction.CLICK: 0.6
            }
            
            total_score = 0
            total_weight = 0
            
            for behavior in recent_behaviors:
                weight = score_factors.get(behavior.action_type, 0.5)
                total_score += weight
                total_weight += 1
            
            if total_weight == 0:
                return 5.0
            
            # 归一化到1-10分
            normalized_score = (total_score / total_weight) * 10
            return min(10.0, max(1.0, normalized_score))
            
        except Exception as e:
            self.logger.error(f"计算满意度分数时出错: {e}")
            return 5.0
    
    async def _get_recent_user_behaviors(self, user_id: str, hours: int = 24) -> List[UserBehavior]:
        """获取用户最近的行为数据"""
        start_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, user_id, session_id, action_type, timestamp, duration, 
                           page_url, element_id, metadata
                    FROM user_behaviors 
                    WHERE user_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """, (user_id, start_time))
                
                behaviors = []
                for row in cursor.fetchall():
                    behaviors.append(UserBehavior(
                        id=row[0],
                        user_id=row[1],
                        session_id=row[2],
                        action_type=UserAction(row[3]),
                        timestamp=row[4],
                        duration=row[5],
                        page_url=row[6],
                        element_id=row[7],
                        metadata=json.loads(row[8]) if row[8] else None
                    ))
                
                return behaviors
        except Exception as e:
            self.logger.error(f"获取用户行为时出错: {e}")
            return []
    
    async def _analyze_behavior_pattern(self, behavior: UserBehavior):
        """分析行为模式"""
        try:
            # 获取用户的历史行为
            user_behaviors = await self._get_recent_user_behaviors(behavior.user_id, hours=168)  # 一周
            
            # 分析行为模式
            pattern_analysis = self._identify_behavior_patterns(user_behaviors)
            
            # 更新用户画像
            await self._update_user_profile(behavior.user_id, pattern_analysis)
            
        except Exception as e:
            self.logger.error(f"分析行为模式时出错: {e}")
    
    def _identify_behavior_patterns(self, behaviors: List[UserBehavior]) -> Dict[str, Any]:
        """识别行为模式"""
        if not behaviors:
            return {}
        
        # 分析会话模式
        session_durations = []
        actions_per_session = defaultdict(int)
        action_timing = defaultdict(list)
        
        current_session = None
        session_start = None
        
        for behavior in sorted(behaviors, key=lambda x: x.timestamp):
            if behavior.session_id != current_session:
                if current_session and session_start:
                    session_duration = behavior.timestamp - session_start
                    session_durations.append(session_duration)
                current_session = behavior.session_id
                session_start = behavior.timestamp
            
            actions_per_session[behavior.action_type] += 1
            hour = datetime.fromtimestamp(behavior.timestamp).hour
            action_timing[behavior.action_type].append(hour)
        
        # 计算模式特征
        patterns = {
            "avg_session_duration": statistics.mean(session_durations) if session_durations else 0,
            "most_common_actions": dict(sorted(actions_per_session.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]),
            "active_hours": self._calculate_active_hours(action_timing),
            "session_frequency": len(set(b.session_id for b in behaviors)) / 7,  # 每天会话数
            "total_actions": len(behaviors),
            "behavior_consistency": self._calculate_behavior_consistency(behaviors)
        }
        
        return patterns
    
    def _calculate_active_hours(self, action_timing: Dict[UserAction, List[int]]) -> Dict[int, int]:
        """计算活跃时段"""
        hour_counts = defaultdict(int)
        
        for action_type, hours in action_timing.items():
            for hour in hours:
                hour_counts[hour] += 1
        
        return dict(hour_counts)
    
    def _calculate_behavior_consistency(self, behaviors: List[UserBehavior]) -> float:
        """计算行为一致性"""
        if len(behaviors) < 10:
            return 0.5
        
        # 简化的一致性计算：基于行为类型的分布稳定性
        action_counts = defaultdict(int)
        for behavior in behaviors:
            action_counts[behavior.action_type] += 1
        
        total_actions = len(behaviors)
        if total_actions == 0:
            return 0.0
        
        # 计算香农熵作为一致性指标
        entropy = 0
        for count in action_counts.values():
            if count > 0:
                p = count / total_actions
                entropy -= p * np.log2(p)
        
        # 归一化到0-1
        max_entropy = np.log2(len(action_counts))
        consistency = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        return consistency
    
    async def _update_user_profiles(self):
        """更新用户画像"""
        while self.is_running:
            try:
                await asyncio.sleep(600)  # 每10分钟更新一次
                
                # 获取所有活跃用户
                active_users = await self._get_active_users()
                
                for user_id in active_users:
                    await self._update_single_user_profile(user_id)
                
            except Exception as e:
                self.logger.error(f"更新用户画像时出错: {e}")
                await asyncio.sleep(600)
    
    async def _update_single_user_profile(self, user_id: str):
        """更新单个用户画像"""
        try:
            # 获取用户统计数据
            user_stats = await self._get_user_statistics(user_id)
            
            # 确定用户分群
            segment = self._determine_user_segment(user_stats)
            
            # 创建用户画像
            profile = UserProfile(
                user_id=user_id,
                segment=segment,
                total_sessions=user_stats["total_sessions"],
                total_actions=user_stats["total_actions"],
                avg_session_duration=user_stats["avg_session_duration"],
                preferred_features=user_stats["preferred_features"],
                behavior_patterns=user_stats["behavior_patterns"],
                last_active=user_stats["last_active"],
                satisfaction_score=user_stats["satisfaction_score"]
            )
            
            # 保存到数据库
            self._save_user_profile(profile)
            
            # 更新缓存
            with self.lock:
                self.user_profiles[user_id] = profile
            
        except Exception as e:
            self.logger.error(f"更新用户画像 {user_id} 时出错: {e}")
    
    async def _get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计数据"""
        try:
            # 获取用户会话数据
            sessions = await self._get_user_sessions(user_id)
            
            # 获取用户行为数据
            behaviors = await self._get_recent_user_behaviors(user_id, hours=8760)  # 一年
            
            # 计算统计指标
            total_sessions = len(sessions)
            total_actions = len(behaviors)
            avg_session_duration = statistics.mean([s["duration"] for s in sessions]) if sessions else 0
            
            # 分析偏好功能
            action_counts = defaultdict(int)
            for behavior in behaviors:
                action_counts[behavior.action_type] += 1
            
            preferred_features = list(sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3])
            preferred_features = [action.value for action, _ in preferred_features]
            
            # 获取行为模式
            behavior_patterns = self._identify_behavior_patterns(behaviors)
            
            # 获取最后活跃时间
            last_active = max([b.timestamp for b in behaviors]) if behaviors else 0
            
            # 计算满意度分数
            satisfaction_score = await self._calculate_satisfaction_score(user_id)
            
            return {
                "total_sessions": total_sessions,
                "total_actions": total_actions,
                "avg_session_duration": avg_session_duration,
                "preferred_features": preferred_features,
                "behavior_patterns": behavior_patterns,
                "last_active": last_active,
                "satisfaction_score": satisfaction_score
            }
            
        except Exception as e:
            self.logger.error(f"获取用户统计时出错: {e}")
            return {}
    
    async def _get_user_sessions(self, user_id: str) -> List[Dict]:
        """获取用户会话数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT session_id, start_time, end_time, action_count, total_duration
                    FROM user_sessions 
                    WHERE user_id = ?
                    ORDER BY start_time DESC
                """, (user_id,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        "session_id": row[0],
                        "start_time": row[1],
                        "end_time": row[2],
                        "action_count": row[3],
                        "duration": row[4] or (row[2] - row[1] if row[2] else 0)
                    })
                
                return sessions
        except Exception as e:
            self.logger.error(f"获取用户会话时出错: {e}")
            return []
    
    def _determine_user_segment(self, user_stats: Dict[str, Any]) -> UserSegment:
        """确定用户分群"""
        total_sessions = user_stats.get("total_sessions", 0)
        last_active = user_stats.get("last_active", 0)
        days_since_active = (time.time() - last_active) / (24 * 3600) if last_active > 0 else float('inf')
        
        # 新用户
        if total_sessions <= self.config["segment_thresholds"]["new_user"]["sessions"]:
            return UserSegment.NEW_USER
        
        # 活跃用户
        if total_sessions >= self.config["segment_thresholds"]["active_user"]["sessions"]:
            return UserSegment.ACTIVE_USER
        
        # 高级用户
        if total_sessions >= self.config["segment_thresholds"]["power_user"]["sessions"]:
            return UserSegment.POWER_USER
        
        # 非活跃用户
        if days_since_active > 30:
            return UserSegment.INACTIVE_USER
        
        return UserSegment.ACTIVE_USER
    
    def _save_user_profile(self, profile: UserProfile):
        """保存用户画像"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_profiles 
                    (user_id, segment, total_sessions, total_actions, avg_session_duration,
                     preferred_features, behavior_patterns, last_active, satisfaction_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    profile.segment.value,
                    profile.total_sessions,
                    profile.total_actions,
                    profile.avg_session_duration,
                    json.dumps(profile.preferred_features),
                    json.dumps(profile.behavior_patterns),
                    profile.last_active,
                    profile.satisfaction_score,
                    time.time()
                ))
        except Exception as e:
            self.logger.error(f"保存用户画像时出错: {e}")
    
    async def _get_active_users(self) -> List[str]:
        """获取活跃用户"""
        try:
            cutoff_time = time.time() - (7 * 24 * 3600)  # 最近7天
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT user_id 
                    FROM user_behaviors 
                    WHERE timestamp > ?
                """, (cutoff_time,))
                
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"获取活跃用户时出错: {e}")
            return []
    
    async def _analyze_experience_metrics(self):
        """分析体验指标"""
        while self.is_running:
            try:
                await asyncio.sleep(900)  # 每15分钟分析一次
                
                # 获取关键体验指标
                metrics_to_analyze = [
                    "response_time",
                    "satisfaction_score",
                    "user_activity",
                    "task_completion_rate"
                ]
                
                for metric_name in metrics_to_analyze:
                    await self._analyze_single_metric(metric_name)
                
            except Exception as e:
                self.logger.error(f"分析体验指标时出错: {e}")
                await asyncio.sleep(900)
    
    async def _analyze_single_metric(self, metric_name: str):
        """分析单个体验指标"""
        try:
            # 获取历史数据
            end_time = time.time()
            start_time = end_time - (24 * 3600)  # 最近24小时
            
            values = await self._get_metric_history(metric_name, start_time, end_time)
            
            if len(values) < 10:
                return
            
            # 计算趋势
            trend = self._calculate_trend(values)
            
            # 生成洞察
            insights = self._generate_metric_insights(metric_name, values, trend)
            
            # 生成推荐
            recommendations = self._generate_metric_recommendations(metric_name, values, trend)
            
            # 计算置信度
            confidence = self._calculate_analysis_confidence(values)
            
            # 保存洞察
            insight = ExperienceInsight(
                metric_name=metric_name,
                current_value=values[-1] if values else 0,
                trend=trend,
                insights=insights,
                recommendations=recommendations,
                confidence=confidence,
                timestamp=time.time()
            )
            
            await self._save_experience_insight(insight)
            
        except Exception as e:
            self.logger.error(f"分析指标 {metric_name} 时出错: {e}")
    
    async def _get_metric_history(self, metric_name: str, start_time: float, end_time: float) -> List[float]:
        """获取指标历史数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT metric_value 
                    FROM experience_metrics 
                    WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                """, (metric_name, start_time, end_time))
                
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"获取指标历史时出错: {e}")
            return []
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 5:
            return "stable"
        
        # 计算前一半和后一半的平均值
        mid_point = len(values) // 2
        first_half_avg = statistics.mean(values[:mid_point])
        second_half_avg = statistics.mean(values[mid_point:])
        
        change_ratio = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg != 0 else 0
        
        if change_ratio > 0.1:
            return "improving"
        elif change_ratio < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _generate_metric_insights(self, metric_name: str, values: List[float], trend: str) -> List[str]:
        """生成指标洞察"""
        insights = []
        
        if metric_name == "response_time":
            avg_response_time = statistics.mean(values)
            if avg_response_time > 2.0:
                insights.append("平均响应时间较长，可能影响用户体验")
            elif avg_response_time < 0.5:
                insights.append("响应时间表现良好")
            
            if trend == "improving":
                insights.append("响应时间正在改善")
            elif trend == "declining":
                insights.append("响应时间正在恶化，需要关注")
        
        elif metric_name == "satisfaction_score":
            avg_satisfaction = statistics.mean(values)
            if avg_satisfaction > 7.0:
                insights.append("用户满意度较高")
            elif avg_satisfaction < 5.0:
                insights.append("用户满意度偏低，需要改进")
            
            if trend == "improving":
                insights.append("用户满意度正在提升")
            elif trend == "declining":
                insights.append("用户满意度正在下降")
        
        return insights
    
    def _generate_metric_recommendations(self, metric_name: str, values: List[float], trend: str) -> List[str]:
        """生成指标推荐"""
        recommendations = []
        
        if metric_name == "response_time":
            if trend == "declining":
                recommendations.append("优化系统性能，减少响应时间")
                recommendations.append("检查服务器负载，考虑扩容")
            elif trend == "stable":
                recommendations.append("保持当前性能水平")
        
        elif metric_name == "satisfaction_score":
            if trend == "declining":
                recommendations.append("调查用户不满原因，改进产品功能")
                recommendations.append("加强用户支持，提升服务质量")
            elif trend == "improving":
                recommendations.append("继续保持当前策略")
        
        return recommendations
    
    def _calculate_analysis_confidence(self, values: List[float]) -> float:
        """计算分析置信度"""
        if len(values) < 10:
            return 0.5
        
        # 基于数据量和稳定性计算置信度
        data_quality = min(1.0, len(values) / 100)  # 数据量评分
        
        if len(values) > 1:
            stability = 1.0 - (statistics.stdev(values) / statistics.mean(values))  # 稳定性评分
            stability = max(0.0, min(1.0, stability))
        else:
            stability = 0.5
        
        return (data_quality + stability) / 2
    
    async def _save_experience_insight(self, insight: ExperienceInsight):
        """保存体验洞察"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experience_insights 
                    (metric_name, current_value, trend, insights, recommendations, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.metric_name,
                    insight.current_value,
                    insight.trend,
                    json.dumps(insight.insights),
                    json.dumps(insight.recommendations),
                    insight.confidence,
                    insight.timestamp
                ))
        except Exception as e:
            self.logger.error(f"保存体验洞察时出错: {e}")
    
    async def _detect_behavior_patterns(self):
        """检测行为模式"""
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # 每30分钟检测一次
                
                # 检测异常行为模式
                await self._detect_anomalous_behaviors()
                
                # 检测用户流失风险
                await self._detect_churn_risk()
                
                # 检测使用高峰时段
                await self._detect_usage_patterns()
                
            except Exception as e:
                self.logger.error(f"检测行为模式时出错: {e}")
                await asyncio.sleep(1800)
    
    async def _detect_anomalous_behaviors(self):
        """检测异常行为"""
        # 简化实现
        pass
    
    async def _detect_churn_risk(self):
        """检测用户流失风险"""
        # 简化实现
        pass
    
    async def _detect_usage_patterns(self):
        """检测使用模式"""
        # 简化实现
        pass
    
    async def _generate_insights(self):
        """生成综合洞察"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # 每小时生成一次
                
                # 生成用户行为洞察
                await self._generate_user_behavior_insights()
                
                # 生成体验优化建议
                await self._generate_optimization_suggestions()
                
            except Exception as e:
                self.logger.error(f"生成洞察时出错: {e}")
                await asyncio.sleep(3600)
    
    async def _generate_user_behavior_insights(self):
        """生成用户行为洞察"""
        # 获取用户分群统计
        segment_stats = await self._get_user_segment_statistics()
        
        # 分析用户行为趋势
        behavior_trends = await self._analyze_behavior_trends()
        
        self.logger.info(f"用户行为洞察: {segment_stats}")
    
    async def _generate_optimization_suggestions(self):
        """生成优化建议"""
        # 基于分析结果生成优化建议
        pass
    
    async def _get_user_segment_statistics(self) -> Dict[str, int]:
        """获取用户分群统计"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT segment, COUNT(*) as count
                    FROM user_profiles
                    GROUP BY segment
                """)
                
                return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            self.logger.error(f"获取用户分群统计时出错: {e}")
            return {}
    
    async def _analyze_behavior_trends(self) -> Dict[str, Any]:
        """分析行为趋势"""
        # 简化实现
        return {}
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # 每5分钟清理一次
                
                current_time = time.time()
                expired_sessions = []
                
                with self.lock:
                    for session_id, session in self.active_sessions.items():
                        if current_time - session.start_time > self.config["session_timeout"]:
                            expired_sessions.append(session_id)
                
                # 清理过期会话
                for session_id in expired_sessions:
                    await self._finalize_session(session_id)
                    del self.active_sessions[session_id]
                
            except Exception as e:
                self.logger.error(f"清理过期会话时出错: {e}")
                await asyncio.sleep(300)
    
    async def _finalize_session(self, session_id: str):
        """完成会话"""
        try:
            if session_id not in self.active_sessions:
                return
            
            session = self.active_sessions[session_id]
            session.end_time = time.time()
            duration = session.end_time - session.start_time
            
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_sessions 
                    (session_id, user_id, start_time, end_time, action_count, total_duration, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.start_time,
                    session.end_time,
                    len(session.actions),
                    duration,
                    json.dumps(session.metadata) if session.metadata else None
                ))
            
            self.logger.debug(f"完成会话: {session_id}")
            
        except Exception as e:
            self.logger.error(f"完成会话时出错: {e}")
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        while self.is_running:
            try:
                await asyncio.sleep(86400)  # 每天清理一次
                
                cutoff_time = time.time() - (self.config["retention_period"] * 24 * 3600)
                
                with sqlite3.connect(self.db_path) as conn:
                    # 清理旧行为数据
                    conn.execute("DELETE FROM user_behaviors WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧体验指标
                    conn.execute("DELETE FROM experience_metrics WHERE timestamp < ?", (cutoff_time,))
                    
                    # 清理旧洞察
                    conn.execute("DELETE FROM experience_insights WHERE timestamp < ?", (cutoff_time,))
                
                self.logger.info("清理旧用户分析数据完成")
                
            except Exception as e:
                self.logger.error(f"清理旧数据时出错: {e}")
                await asyncio.sleep(86400)
    
    def add_analytics_callback(self, callback: Callable):
        """添加分析回调函数"""
        self.analytics_callbacks.append(callback)
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        with self.lock:
            return self.user_profiles.get(user_id)
    
    def get_user_behavior_history(self, user_id: str, hours: int = 24) -> List[UserBehavior]:
        """获取用户行为历史"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, user_id, session_id, action_type, timestamp, duration, 
                       page_url, element_id, metadata
                FROM user_behaviors 
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (user_id, start_time))
            
            behaviors = []
            for row in cursor.fetchall():
                behaviors.append(UserBehavior(
                    id=row[0],
                    user_id=row[1],
                    session_id=row[2],
                    action_type=UserAction(row[3]),
                    timestamp=row[4],
                    duration=row[5],
                    page_url=row[6],
                    element_id=row[7],
                    metadata=json.loads(row[8]) if row[8] else None
                ))
            
            return behaviors
    
    def get_experience_insights(self, hours: int = 24) -> List[ExperienceInsight]:
        """获取体验洞察"""
        start_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_name, current_value, trend, insights, recommendations, confidence, timestamp
                FROM experience_insights 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (start_time,))
            
            insights = []
            for row in cursor.fetchall():
                insights.append(ExperienceInsight(
                    metric_name=row[0],
                    current_value=row[1],
                    trend=row[2],
                    insights=json.loads(row[3]),
                    recommendations=json.loads(row[4]),
                    confidence=row[5],
                    timestamp=row[6]
                ))
            
            return insights
    
    def get_user_segments_summary(self) -> Dict[str, Any]:
        """获取用户分群摘要"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT segment, COUNT(*) as count, AVG(satisfaction_score) as avg_satisfaction
                FROM user_profiles
                GROUP BY segment
            """)
            
            summary = {}
            for row in cursor.fetchall():
                summary[row[0]] = {
                    "count": row[1],
                    "avg_satisfaction": row[2]
                }
            
            return summary

# 使用示例
async def main():
    """主函数示例"""
    analytics = UserAnalytics()
    
    # 添加分析回调
    async def analytics_callback(data: Any):
        print(f"用户分析更新: {data}")
    
    analytics.add_analytics_callback(analytics_callback)
    
    try:
        await analytics.start_analytics()
        
        # 模拟用户行为
        analytics.track_user_action("user123", UserAction.LOGIN)
        analytics.track_user_action("user123", UserAction.SEND_MESSAGE, 
                                  metadata={"message_length": 100})
        analytics.track_user_action("user123", UserAction.RECEIVE_RESPONSE, 
                                  duration=1.5)
        
        # 运行一段时间
        await asyncio.sleep(60)
        
    finally:
        await analytics.stop_analytics()

if __name__ == "__main__":
    asyncio.run(main())