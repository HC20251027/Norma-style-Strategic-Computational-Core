#!/usr/bin/env python3
"""
性能评估器
负责评估多智能体系统的协作性能和效果

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import numpy as np

from ..utils.logger import MultiAgentLogger


class MetricType(Enum):
    """指标类型枚举"""
    EFFICIENCY = "efficiency"          # 效率指标
    QUALITY = "quality"                # 质量指标
    RELIABILITY = "reliability"        # 可靠性指标
    SCALABILITY = "scalability"        # 可扩展性指标
    LATENCY = "latency"                # 延迟指标
    THROUGHPUT = "throughput"          # 吞吐量指标
    RESOURCE_UTILIZATION = "resource_utilization"  # 资源利用率


class EvaluationPeriod(Enum):
    """评估周期枚举"""
    REAL_TIME = "real_time"            # 实时
    HOURLY = "hourly"                  # 每小时
    DAILY = "daily"                    # 每天
    WEEKLY = "weekly"                  # 每周
    MONTHLY = "monthly"                # 每月


@dataclass
class PerformanceMetric:
    """性能指标"""
    metric_id: str
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class EvaluationResult:
    """评估结果"""
    evaluation_id: str
    evaluation_type: str
    period: EvaluationPeriod
    metrics: List[PerformanceMetric]
    score: float  # 综合评分 0-1
    grade: str    # 等级 (A, B, C, D, F)
    recommendations: List[str]
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AgentPerformance:
    """智能体性能"""
    agent_id: str
    tasks_completed: int
    tasks_failed: int
    average_response_time: float
    success_rate: float
    resource_usage: Dict[str, float]
    quality_score: float
    last_updated: datetime
    
    @property
    def overall_score(self) -> float:
        """综合评分"""
        return (self.success_rate * 0.4 + 
                self.quality_score * 0.3 + 
                (1.0 - min(self.average_response_time / 10.0, 1.0)) * 0.3)


@dataclass
class CollaborationPerformance:
    """协作性能"""
    collaboration_id: str
    participants: List[str]
    duration: float
    efficiency: float
    quality: float
    success: bool
    conflicts_detected: int
    resource_consumption: Dict[str, float]
    
    @property
    def overall_score(self) -> float:
        """协作综合评分"""
        return (self.efficiency * 0.4 + 
                self.quality * 0.4 + 
                (1.0 if self.success else 0.0) * 0.2)


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self):
        """初始化性能评估器"""
        self.evaluator_id = str(uuid.uuid4())
        
        # 初始化日志
        self.logger = MultiAgentLogger("performance_evaluator")
        
        # 性能数据存储
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.agent_performances: Dict[str, AgentPerformance] = {}
        self.collaboration_performances: Dict[str, CollaborationPerformance] = {}
        
        # 评估配置
        self.evaluation_config = {
            MetricType.EFFICIENCY: {"weight": 0.25, "threshold": 0.8},
            MetricType.QUALITY: {"weight": 0.25, "threshold": 0.85},
            MetricType.RELIABILITY: {"weight": 0.20, "threshold": 0.90},
            MetricType.LATENCY: {"weight": 0.15, "threshold": 2.0},  # 秒
            MetricType.THROUGHPUT: {"weight": 0.10, "threshold": 100},  # 任务/分钟
            MetricType.RESOURCE_UTILIZATION: {"weight": 0.05, "threshold": 0.75}
        }
        
        # 评分标准
        self.grading_scale = {
            "A": (0.9, 1.0),
            "B": (0.8, 0.9),
            "C": (0.7, 0.8),
            "D": (0.6, 0.7),
            "F": (0.0, 0.6)
        }
        
        # 统计信息
        self.metrics = {
            "evaluations_performed": 0,
            "total_metrics_collected": 0,
            "average_system_score": 0.0,
            "performance_trends": defaultdict(list),
            "bottlenecks_detected": [],
            "optimization_suggestions": []
        }
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger.info(f"性能评估器 {self.evaluator_id} 已初始化")
    
    async def initialize(self) -> bool:
        """初始化性能评估器"""
        try:
            self.logger.info("初始化性能评估器...")
            
            # 启动后台任务
            asyncio.create_task(self._real_time_monitor())
            asyncio.create_task(self._periodic_evaluator())
            asyncio.create_task(self._trend_analyzer())
            asyncio.create_task(self._bottleneck_detector())
            
            self.logger.info("性能评估器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"性能评估器初始化失败: {e}")
            return False
    
    async def collect_metric(
        self,
        metric_type: MetricType,
        name: str,
        value: float,
        unit: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """收集性能指标
        
        Args:
            metric_type: 指标类型
            name: 指标名称
            value: 指标值
            unit: 单位
            context: 上下文信息
            
        Returns:
            指标ID
        """
        try:
            metric_id = str(uuid.uuid4())
            
            metric = PerformanceMetric(
                metric_id=metric_id,
                metric_type=metric_type,
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                context=context or {}
            )
            
            # 存储指标
            metric_key = f"{metric_type.value}_{name}"
            self.metrics_history[metric_key].append(metric)
            
            self.metrics["total_metrics_collected"] += 1
            
            # 发送指标收集事件
            await self._emit_event("metric.collected", {
                "metric_id": metric_id,
                "metric_type": metric_type.value,
                "name": name,
                "value": value,
                "timestamp": metric.timestamp.isoformat()
            })
            
            return metric_id
            
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
            raise
    
    async def update_agent_performance(
        self,
        agent_id: str,
        tasks_completed: int,
        tasks_failed: int,
        average_response_time: float,
        resource_usage: Dict[str, float],
        quality_score: float
    ):
        """更新智能体性能"""
        try:
            success_rate = tasks_completed / max(tasks_completed + tasks_failed, 1)
            
            performance = AgentPerformance(
                agent_id=agent_id,
                tasks_completed=tasks_completed,
                tasks_failed=tasks_failed,
                average_response_time=average_response_time,
                success_rate=success_rate,
                resource_usage=resource_usage,
                quality_score=quality_score,
                last_updated=datetime.now()
            )
            
            self.agent_performances[agent_id] = performance
            
            # 收集相关指标
            await self.collect_metric(
                MetricType.EFFICIENCY,
                f"agent_{agent_id}_success_rate",
                success_rate,
                "ratio",
                {"agent_id": agent_id}
            )
            
            await self.collect_metric(
                MetricType.LATENCY,
                f"agent_{agent_id}_response_time",
                average_response_time,
                "seconds",
                {"agent_id": agent_id}
            )
            
            await self.collect_metric(
                MetricType.QUALITY,
                f"agent_{agent_id}_quality_score",
                quality_score,
                "ratio",
                {"agent_id": agent_id}
            )
            
        except Exception as e:
            self.logger.error(f"更新智能体性能失败 {agent_id}: {e}")
    
    async def update_collaboration_performance(
        self,
        collaboration_id: str,
        participants: List[str],
        duration: float,
        efficiency: float,
        quality: float,
        success: bool,
        conflicts_detected: int,
        resource_consumption: Dict[str, float]
    ):
        """更新协作性能"""
        try:
            performance = CollaborationPerformance(
                collaboration_id=collaboration_id,
                participants=participants,
                duration=duration,
                efficiency=efficiency,
                quality=quality,
                success=success,
                conflicts_detected=conflicts_detected,
                resource_consumption=resource_consumption
            )
            
            self.collaboration_performances[collaboration_id] = performance
            
            # 收集相关指标
            await self.collect_metric(
                MetricType.EFFICIENCY,
                f"collaboration_{collaboration_id}_efficiency",
                efficiency,
                "ratio",
                {"collaboration_id": collaboration_id}
            )
            
            await self.collect_metric(
                MetricType.QUALITY,
                f"collaboration_{collaboration_id}_quality",
                quality,
                "ratio",
                {"collaboration_id": collaboration_id}
            )
            
            await self.collect_metric(
                MetricType.LATENCY,
                f"collaboration_{collaboration_id}_duration",
                duration,
                "seconds",
                {"collaboration_id": collaboration_id}
            )
            
        except Exception as e:
            self.logger.error(f"更新协作性能失败 {collaboration_id}: {e}")
    
    async def evaluate_system_performance(
        self,
        agents: Dict[str, Any],
        collaboration_history: List[Dict[str, Any]],
        active_collaborations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估系统整体性能
        
        Args:
            agents: 智能体信息
            collaboration_history: 协作历史
            active_collaborations: 活跃协作
            
        Returns:
            性能评估结果
        """
        try:
            evaluation_id = str(uuid.uuid4())
            
            # 计算各项性能指标
            efficiency_score = await self._calculate_efficiency_score(agents, collaboration_history)
            quality_score = await self._calculate_quality_score(agents, collaboration_history)
            reliability_score = await self._calculate_reliability_score(agents, collaboration_history)
            latency_score = await self._calculate_latency_score(agents, collaboration_history)
            throughput_score = await self._calculate_throughput_score(agents, collaboration_history)
            resource_score = await self._calculate_resource_utilization_score(agents)
            
            # 计算综合评分
            overall_score = (
                efficiency_score * self.evaluation_config[MetricType.EFFICIENCY]["weight"] +
                quality_score * self.evaluation_config[MetricType.QUALITY]["weight"] +
                reliability_score * self.evaluation_config[MetricType.RELIABILITY]["weight"] +
                latency_score * self.evaluation_config[MetricType.LATENCY]["weight"] +
                throughput_score * self.evaluation_config[MetricType.THROUGHPUT]["weight"] +
                resource_score * self.evaluation_config[MetricType.RESOURCE_UTILIZATION]["weight"]
            )
            
            # 确定等级
            grade = self._determine_grade(overall_score)
            
            # 生成建议
            recommendations = await self._generate_recommendations(
                efficiency_score, quality_score, reliability_score, 
                latency_score, throughput_score, resource_score
            )
            
            # 创建评估结果
            result = EvaluationResult(
                evaluation_id=evaluation_id,
                evaluation_type="system_performance",
                period=EvaluationPeriod.REAL_TIME,
                metrics=[
                    PerformanceMetric("", MetricType.EFFICIENCY, "efficiency", efficiency_score, "ratio", datetime.now()),
                    PerformanceMetric("", MetricType.QUALITY, "quality", quality_score, "ratio", datetime.now()),
                    PerformanceMetric("", MetricType.RELIABILITY, "reliability", reliability_score, "ratio", datetime.now()),
                    PerformanceMetric("", MetricType.LATENCY, "latency", latency_score, "ratio", datetime.now()),
                    PerformanceMetric("", MetricType.THROUGHPUT, "throughput", throughput_score, "ratio", datetime.now()),
                    PerformanceMetric("", MetricType.RESOURCE_UTILIZATION, "resource_utilization", resource_score, "ratio", datetime.now())
                ],
                score=overall_score,
                grade=grade,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # 更新统计
            self.metrics["evaluations_performed"] += 1
            self.metrics["average_system_score"] = (
                (self.metrics["average_system_score"] * (self.metrics["evaluations_performed"] - 1) + overall_score) / 
                self.metrics["evaluations_performed"]
            )
            
            # 记录性能趋势
            self.metrics["performance_trends"]["overall_score"].append((datetime.now(), overall_score))
            
            # 发送评估完成事件
            await self._emit_event("evaluation.completed", {
                "evaluation_id": evaluation_id,
                "overall_score": overall_score,
                "grade": grade,
                "recommendations_count": len(recommendations)
            })
            
            self.logger.info(f"系统性能评估完成: {overall_score:.2f} ({grade})")
            
            return {
                "evaluation_id": evaluation_id,
                "overall_score": overall_score,
                "grade": grade,
                "metrics": {
                    "efficiency": efficiency_score,
                    "quality": quality_score,
                    "reliability": reliability_score,
                    "latency": latency_score,
                    "throughput": throughput_score,
                    "resource_utilization": resource_score
                },
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"评估系统性能失败: {e}")
            raise
    
    async def _calculate_efficiency_score(
        self,
        agents: Dict[str, Any],
        collaboration_history: List[Dict[str, Any]]
    ) -> float:
        """计算效率评分"""
        if not collaboration_history:
            return 0.5
        
        # 计算任务完成效率
        completed_collaborations = [c for c in collaboration_history if c.get("result", {}).get("success", False)]
        efficiency_scores = []
        
        for collab in completed_collaborations:
            # 基于预期时间vs实际时间的效率
            expected_duration = collab.get("estimated_duration", 60)
            actual_duration = collab.get("result", {}).get("duration", 60)
            
            if expected_duration > 0:
                efficiency = min(1.0, expected_duration / actual_duration)
                efficiency_scores.append(efficiency)
        
        return statistics.mean(efficiency_scores) if efficiency_scores else 0.5
    
    async def _calculate_quality_score(
        self,
        agents: Dict[str, Any],
        collaboration_history: List[Dict[str, Any]]
    ) -> float:
        """计算质量评分"""
        if not collaboration_history:
            return 0.5
        
        # 基于任务成功率和结果质量
        successful_collaborations = [c for c in collaboration_history if c.get("result", {}).get("success", False)]
        
        if not successful_collaborations:
            return 0.0
        
        # 计算平均质量分数
        quality_scores = []
        for collab in successful_collaborations:
            result = collab.get("result", {})
            quality = result.get("quality_score", 0.8)  # 默认质量分数
            quality_scores.append(quality)
        
        return statistics.mean(quality_scores) if quality_scores else 0.5
    
    async def _calculate_reliability_score(
        self,
        agents: Dict[str, Any],
        collaboration_history: List[Dict[str, Any]]
    ) -> float:
        """计算可靠性评分"""
        if not collaboration_history:
            return 0.5
        
        # 计算成功率
        total_collaborations = len(collaboration_history)
        successful_collaborations = len([c for c in collaboration_history if c.get("result", {}).get("success", False)])
        
        success_rate = successful_collaborations / total_collaborations if total_collaborations > 0 else 0
        
        # 考虑一致性（方差）
        if len(collaboration_history) > 1:
            durations = [c.get("result", {}).get("duration", 60) for c in collaboration_history]
            duration_variance = statistics.variance(durations) if len(durations) > 1 else 0
            consistency_score = max(0, 1.0 - min(duration_variance / 100.0, 1.0))  # 归一化方差
        else:
            consistency_score = 1.0
        
        # 综合可靠性评分
        reliability_score = (success_rate * 0.7 + consistency_score * 0.3)
        
        return reliability_score
    
    async def _calculate_latency_score(
        self,
        agents: Dict[str, Any],
        collaboration_history: List[Dict[str, Any]]
    ) -> float:
        """计算延迟评分"""
        if not collaboration_history:
            return 0.5
        
        # 计算平均响应时间
        response_times = []
        for collab in collaboration_history:
            result = collab.get("result", {})
            duration = result.get("duration", 60)
            response_times.append(duration)
        
        if not response_times:
            return 0.5
        
        avg_response_time = statistics.mean(response_times)
        
        # 将响应时间转换为评分（越短越好）
        # 假设2秒以内为满分，30秒以上为0分
        latency_score = max(0, min(1.0, (30 - avg_response_time) / 28))
        
        return latency_score
    
    async def _calculate_throughput_score(
        self,
        agents: Dict[str, Any],
        collaboration_history: List[Dict[str, Any]]
    ) -> float:
        """计算吞吐量评分"""
        if not collaboration_history:
            return 0.5
        
        # 计算单位时间内的任务处理量
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        recent_collaborations = [
            c for c in collaboration_history 
            if datetime.fromisoformat(c.get("executed_at", now.isoformat())) > one_hour_ago
        ]
        
        throughput = len(recent_collaborations)  # 每小时任务数
        
        # 将吞吐量转换为评分
        # 假设100任务/小时为满分
        throughput_score = min(1.0, throughput / 100.0)
        
        return throughput_score
    
    async def _calculate_resource_utilization_score(self, agents: Dict[str, Any]) -> float:
        """计算资源利用率评分"""
        if not agents:
            return 0.5
        
        # 计算平均资源利用率
        utilization_scores = []
        
        for agent_id, agent_info in agents.items():
            if "resource_usage" in agent_info:
                usage = agent_info["resource_usage"]
                
                # 计算综合利用率
                cpu_usage = usage.get("cpu_percent", 0) / 100.0
                memory_usage = usage.get("memory_percent", 0) / 100.0
                
                # 理想利用率在70-80%之间
                ideal_utilization = 0.75
                cpu_score = 1.0 - abs(cpu_usage - ideal_utilization)
                memory_score = 1.0 - abs(memory_usage - ideal_utilization)
                
                avg_score = (cpu_score + memory_score) / 2
                utilization_scores.append(max(0, avg_score))
        
        return statistics.mean(utilization_scores) if utilization_scores else 0.5
    
    def _determine_grade(self, score: float) -> str:
        """确定等级"""
        for grade, (min_score, max_score) in self.grading_scale.items():
            if min_score <= score < max_score:
                return grade
        return "F"  # 默认等级
    
    async def _generate_recommendations(
        self,
        efficiency_score: float,
        quality_score: float,
        reliability_score: float,
        latency_score: float,
        throughput_score: float,
        resource_score: float
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 效率建议
        if efficiency_score < 0.7:
            recommendations.append("优化任务分配算法，提高执行效率")
            recommendations.append("考虑重新设计协作流程，减少不必要的步骤")
        
        # 质量建议
        if quality_score < 0.8:
            recommendations.append("加强智能体质量控制机制")
            recommendations.append("改进结果验证和审核流程")
        
        # 可靠性建议
        if reliability_score < 0.8:
            recommendations.append("增强错误处理和恢复机制")
            recommendations.append("增加冗余和备份策略")
        
        # 延迟建议
        if latency_score < 0.6:
            recommendations.append("优化网络通信和消息传递")
            recommendations.append("减少不必要的同步操作")
        
        # 吞吐量建议
        if throughput_score < 0.5:
            recommendations.append("增加并行处理能力")
            recommendations.append("优化资源分配策略")
        
        # 资源利用率建议
        if resource_score < 0.6:
            recommendations.append("优化资源分配，避免过度分配或分配不足")
            recommendations.append("实施动态资源调整机制")
        
        # 通用建议
        if len(recommendations) == 0:
            recommendations.append("系统性能良好，继续保持当前配置")
        
        return recommendations
    
    async def _real_time_monitor(self):
        """实时监控后台任务"""
        while True:
            try:
                await asyncio.sleep(10)  # 每10秒监控一次
                
                # 检查关键指标
                await self._check_critical_metrics()
                
            except Exception as e:
                self.logger.error(f"实时监控任务出错: {e}")
    
    async def _periodic_evaluator(self):
        """定期评估后台任务"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟评估一次
                
                # 执行定期评估
                await self._perform_periodic_evaluation()
                
            except Exception as e:
                self.logger.error(f"定期评估任务出错: {e}")
    
    async def _trend_analyzer(self):
        """趋势分析后台任务"""
        while True:
            try:
                await asyncio.sleep(600)  # 每10分钟分析一次
                
                # 分析性能趋势
                await self._analyze_performance_trends()
                
            except Exception as e:
                self.logger.error(f"趋势分析任务出错: {e}")
    
    async def _bottleneck_detector(self):
        """瓶颈检测后台任务"""
        while True:
            try:
                await asyncio.sleep(900)  # 每15分钟检测一次
                
                # 检测性能瓶颈
                await self._detect_bottlenecks()
                
            except Exception as e:
                self.logger.error(f"瓶颈检测任务出错: {e}")
    
    async def _check_critical_metrics(self):
        """检查关键指标"""
        # 检查最近的效率指标
        efficiency_metrics = list(self.metrics_history.get("efficiency_efficiency", []))
        if efficiency_metrics:
            latest_efficiency = efficiency_metrics[-1].value
            if latest_efficiency < 0.5:
                await self._emit_event("performance.alert", {
                    "type": "low_efficiency",
                    "value": latest_efficiency,
                    "threshold": 0.5
                })
    
    async def _perform_periodic_evaluation(self):
        """执行定期评估"""
        # 这里可以执行更详细的定期评估
        # 简化实现
        pass
    
    async def _analyze_performance_trends(self):
        """分析性能趋势"""
        for metric_type, trend_data in self.metrics["performance_trends"].items():
            if len(trend_data) > 10:
                # 计算趋势斜率
                timestamps = [t for t, _ in trend_data]
                scores = [s for _, s in trend_data]
                
                if len(scores) > 1:
                    # 简单的线性趋势分析
                    recent_scores = scores[-10:]
                    if len(recent_scores) >= 2:
                        trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                        
                        if abs(trend) > 0.1:  # 趋势变化超过10%
                            await self._emit_event("performance.trend_detected", {
                                "metric": metric_type,
                                "trend": "improving" if trend > 0 else "declining",
                                "magnitude": abs(trend)
                            })
    
    async def _detect_bottlenecks(self):
        """检测性能瓶颈"""
        bottlenecks = []
        
        # 检测延迟瓶颈
        latency_metrics = list(self.metrics_history.get("latency_latency", []))
        if latency_metrics:
            recent_latencies = [m.value for m in latency_metrics[-10:]]
            if recent_latencies:
                avg_latency = statistics.mean(recent_latencies)
                if avg_latency > 5.0:  # 平均延迟超过5秒
                    bottlenecks.append("高延迟问题")
        
        # 检测资源瓶颈
        resource_metrics = list(self.metrics_history.get("resource_utilization_resource_utilization", []))
        if resource_metrics:
            recent_utilization = [m.value for m in resource_metrics[-10:]]
            if recent_utilization:
                avg_utilization = statistics.mean(recent_utilization)
                if avg_utilization < 0.3:  # 资源利用率过低
                    bottlenecks.append("资源利用不足")
                elif avg_utilization > 0.9:  # 资源利用率过高
                    bottlenecks.append("资源过度使用")
        
        if bottlenecks:
            self.metrics["bottlenecks_detected"].extend(bottlenecks)
            
            await self._emit_event("performance.bottlenecks_detected", {
                "bottlenecks": bottlenecks,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            "evaluator_id": self.evaluator_id,
            "total_metrics_collected": self.metrics["total_metrics_collected"],
            "evaluations_performed": self.metrics["evaluations_performed"],
            "average_system_score": self.metrics["average_system_score"],
            "active_agents": len(self.agent_performances),
            "completed_collaborations": len(self.collaboration_performances),
            "recent_bottlenecks": self.metrics["bottlenecks_detected"][-5:],  # 最近5个瓶颈
            "performance_trends": {
                metric: len(trend) for metric, trend in self.metrics["performance_trends"].items()
            }
        }
    
    def on_event(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """发送事件"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"事件回调执行失败 {event_type}: {e}")
    
    async def shutdown(self):
        """关闭性能评估器"""
        try:
            self.logger.info("关闭性能评估器...")
            
            # 清理数据
            self.metrics_history.clear()
            self.agent_performances.clear()
            self.collaboration_performances.clear()
            
            self.logger.info("性能评估器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭性能评估器时出错: {e}")