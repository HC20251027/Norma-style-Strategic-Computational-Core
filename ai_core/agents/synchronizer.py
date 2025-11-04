#!/usr/bin/env python3
"""
智能体同步器
负责智能体间的状态同步、时钟同步和协调同步

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from ..utils.logger import MultiAgentLogger
from ..communication.message_bus import MessageBus, Message, MessageType


class SyncType(Enum):
    """同步类型枚举"""
    STATE_SYNC = "state_sync"        # 状态同步
    CLOCK_SYNC = "clock_sync"        # 时钟同步
    CONFIG_SYNC = "config_sync"      # 配置同步
    CAPABILITY_SYNC = "capability_sync"  # 能力同步
    PERFORMANCE_SYNC = "performance_sync"  # 性能同步


class SyncStrategy(Enum):
    """同步策略枚举"""
    PERIODIC = "periodic"            # 定期同步
    EVENT_DRIVEN = "event_driven"    # 事件驱动
    DEMAND_BASED = "demand_based"    # 需求驱动
    QUORUM_BASED = "quorum_based"    # 仲裁同步
    LEADER_BASED = "leader_based"    # 基于领导者的同步


@dataclass
class SyncRequest:
    """同步请求"""
    request_id: str
    sync_type: SyncType
    requester_id: str
    target_agents: List[str]
    sync_data: Dict[str, Any]
    priority: int = 2  # 1-5，5为最高优先级
    timeout: float = 30.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SyncState:
    """同步状态"""
    agent_id: str
    sync_type: SyncType
    last_sync_time: datetime
    sync_version: int
    state_data: Dict[str, Any]
    checksum: str
    is_leader: bool = False
    leader_id: Optional[str] = None
    
    def __post_init__(self):
        if self.leader_id is None:
            self.leader_id = self.agent_id


@dataclass
class SyncResult:
    """同步结果"""
    request_id: str
    success: bool
    synced_agents: List[str]
    failed_agents: List[str]
    sync_duration: float
    conflicts: List[Dict[str, Any]]
    final_state: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class AgentSynchronizer:
    """智能体同步器"""
    
    def __init__(self, message_bus: MessageBus):
        """初始化智能体同步器
        
        Args:
            message_bus: 消息总线
        """
        self.synchronizer_id = str(uuid.uuid4())
        self.message_bus = message_bus
        
        # 初始化日志
        self.logger = MultiAgentLogger("agent_synchronizer")
        
        # 同步管理
        self.sync_states: Dict[str, Dict[str, SyncState]] = defaultdict(dict)  # agent_id -> sync_type -> state
        self.sync_requests: Dict[str, SyncRequest] = {}
        self.sync_results: Dict[str, SyncResult] = {}
        
        # 同步策略配置
        self.sync_strategies = {
            SyncType.STATE_SYNC: SyncStrategy.PERIODIC,
            SyncType.CLOCK_SYNC: SyncStrategy.EVENT_DRIVEN,
            SyncType.CONFIG_SYNC: SyncStrategy.QUORUM_BASED,
            SyncType.CAPABILITY_SYNC: SyncStrategy.LEADER_BASED,
            SyncType.PERFORMANCE_SYNC: SyncStrategy.PERIODIC
        }
        
        # 领导者选举
        self.leaders: Dict[SyncType, str] = {}  # sync_type -> leader_agent_id
        self.leader_heartbeats: Dict[str, datetime] = {}
        
        # 同步配置
        self.sync_config = {
            SyncType.STATE_SYNC: {"interval": 60, "timeout": 10},
            SyncType.CLOCK_SYNC: {"interval": 30, "timeout": 5},
            SyncType.CONFIG_SYNC: {"interval": 300, "timeout": 30},
            SyncType.CAPABILITY_SYNC: {"interval": 120, "timeout": 15},
            SyncType.PERFORMANCE_SYNC: {"interval": 90, "timeout": 20}
        }
        
        # 统计信息
        self.metrics = {
            "sync_requests_total": 0,
            "sync_requests_success": 0,
            "sync_requests_failed": 0,
            "average_sync_duration": 0.0,
            "conflicts_detected": 0,
            "leader_elections": 0,
            "sync_efficiency": 0.0
        }
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # 轮询计数器
        self.sync_counter = 0
        
        self.logger.info(f"智能体同步器 {self.synchronizer_id} 已初始化")
    
    async def initialize(self) -> bool:
        """初始化智能体同步器"""
        try:
            self.logger.info("初始化智能体同步器...")
            
            # 订阅消息总线事件
            await self.message_bus.subscribe("sync.request", self._handle_sync_request)
            await self.message_bus.subscribe("sync.response", self._handle_sync_response)
            await self.message_bus.subscribe("sync.heartbeat", self._handle_sync_heartbeat)
            await self.message_bus.subscribe("agent.join", self._handle_agent_join)
            await self.message_bus.subscribe("agent.leave", self._handle_agent_leave)
            
            # 启动后台任务
            asyncio.create_task(self._periodic_sync_scheduler())
            asyncio.create_task(self._leader_monitor())
            asyncio.create_task(self._conflict_resolver())
            asyncio.create_task(self._clock_synchronizer())
            asyncio.create_task(self._sync_metrics_collector())
            
            self.logger.info("智能体同步器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"智能体同步器初始化失败: {e}")
            return False
    
    async def request_sync(
        self,
        sync_type: SyncType,
        requester_id: str,
        target_agents: List[str],
        sync_data: Dict[str, Any],
        priority: int = 2,
        timeout: float = 30.0
    ) -> str:
        """请求同步
        
        Args:
            sync_type: 同步类型
            requester_id: 请求者ID
            target_agents: 目标智能体列表
            sync_data: 同步数据
            priority: 优先级
            timeout: 超时时间
            
        Returns:
            同步请求ID
        """
        try:
            request_id = str(uuid.uuid4())
            
            # 创建同步请求
            sync_request = SyncRequest(
                request_id=request_id,
                sync_type=sync_type,
                requester_id=requester_id,
                target_agents=target_agents,
                sync_data=sync_data,
                priority=priority,
                timeout=timeout
            )
            
            self.sync_requests[request_id] = sync_request
            self.metrics["sync_requests_total"] += 1
            
            # 根据同步策略处理请求
            strategy = self.sync_strategies.get(sync_type, SyncStrategy.PERIODIC)
            
            if strategy == SyncStrategy.PERIODIC:
                await self._handle_periodic_sync(sync_request)
            elif strategy == SyncStrategy.EVENT_DRIVEN:
                await self._handle_event_driven_sync(sync_request)
            elif strategy == SyncStrategy.DEMAND_BASED:
                await self._handle_demand_based_sync(sync_request)
            elif strategy == SyncStrategy.QUORUM_BASED:
                await self._handle_quorum_based_sync(sync_request)
            elif strategy == SyncStrategy.LEADER_BASED:
                await self._handle_leader_based_sync(sync_request)
            
            # 发送同步请求事件
            await self._emit_event("sync.requested", {
                "request_id": request_id,
                "sync_type": sync_type.value,
                "requester_id": requester_id,
                "target_count": len(target_agents),
                "strategy": strategy.value
            })
            
            self.logger.info(f"同步请求已创建: {sync_type.value} {request_id}")
            return request_id
            
        except Exception as e:
            self.logger.error(f"创建同步请求失败: {e}")
            raise
    
    async def _handle_periodic_sync(self, sync_request: SyncRequest):
        """处理定期同步"""
        # 立即执行一次同步
        await self._execute_sync(sync_request)
        
        # 设置定期同步任务
        sync_type = sync_request.sync_type
        config = self.sync_config.get(sync_type, {})
        interval = config.get("interval", 60)
        
        # 创建定期同步任务
        asyncio.create_task(self._periodic_sync_task(sync_request, interval))
    
    async def _periodic_sync_task(self, sync_request: SyncRequest, interval: float):
        """定期同步任务"""
        while sync_request.request_id in self.sync_requests:
            try:
                await asyncio.sleep(interval)
                
                # 检查请求是否仍然有效
                if sync_request.request_id not in self.sync_requests:
                    break
                
                # 执行同步
                await self._execute_sync(sync_request)
                
            except Exception as e:
                self.logger.error(f"定期同步任务出错: {e}")
    
    async def _handle_event_driven_sync(self, sync_request: SyncRequest):
        """处理事件驱动同步"""
        # 立即执行同步
        await self._execute_sync(sync_request)
    
    async def _handle_demand_based_sync(self, sync_request: SyncRequest):
        """处理需求驱动同步"""
        # 检查是否确实需要同步
        if await self._check_sync_need(sync_request):
            await self._execute_sync(sync_request)
    
    async def _handle_quorum_based_sync(self, sync_request: SyncRequest):
        """处理仲裁同步"""
        target_count = len(sync_request.target_agents)
        quorum = target_count // 2 + 1  # 多数仲裁
        
        # 发送同步请求给所有目标智能体
        responses = await self._broadcast_sync_request(sync_request)
        
        # 检查是否达到仲裁数量
        successful_responses = [r for r in responses if r.get("success", False)]
        
        if len(successful_responses) >= quorum:
            # 达到仲裁，执行同步
            await self._finalize_quorum_sync(sync_request, successful_responses)
        else:
            # 未达到仲裁，同步失败
            await self._fail_sync(sync_request, "Quorum not reached")
    
    async def _handle_leader_based_sync(self, sync_request: SyncRequest):
        """处理基于领导者的同步"""
        sync_type = sync_request.sync_type
        
        # 确保有领导者
        if sync_type not in self.leaders:
            await self._elect_leader(sync_type, sync_request.target_agents)
        
        leader_id = self.leaders.get(sync_type)
        if leader_id:
            # 将同步请求转发给领导者
            await self._forward_to_leader(sync_request, leader_id)
        else:
            # 没有领导者，执行普通同步
            await self._execute_sync(sync_request)
    
    async def _execute_sync(self, sync_request: SyncRequest) -> SyncResult:
        """执行同步"""
        start_time = time.time()
        
        try:
            synced_agents = []
            failed_agents = []
            conflicts = []
            
            # 广播同步请求
            responses = await self._broadcast_sync_request(sync_request)
            
            # 处理响应
            for response in responses:
                agent_id = response.get("agent_id")
                success = response.get("success", False)
                
                if success:
                    synced_agents.append(agent_id)
                    
                    # 检查冲突
                    if "conflict" in response:
                        conflicts.append(response["conflict"])
                else:
                    failed_agents.append(agent_id)
            
            # 解决冲突
            if conflicts:
                final_state = await self._resolve_conflicts(sync_request, conflicts)
                self.metrics["conflicts_detected"] += len(conflicts)
            else:
                final_state = sync_request.sync_data
            
            # 创建同步结果
            sync_duration = time.time() - start_time
            result = SyncResult(
                request_id=sync_request.request_id,
                success=len(synced_agents) > 0,
                synced_agents=synced_agents,
                failed_agents=failed_agents,
                sync_duration=sync_duration,
                conflicts=conflicts,
                final_state=final_state
            )
            
            self.sync_results[sync_request.request_id] = result
            
            # 更新统计
            if result.success:
                self.metrics["sync_requests_success"] += 1
            else:
                self.metrics["sync_requests_failed"] += 1
            
            # 更新平均同步时间
            total_syncs = self.metrics["sync_requests_success"] + self.metrics["sync_requests_failed"]
            if total_syncs > 0:
                self.metrics["average_sync_duration"] = (
                    (self.metrics["average_sync_duration"] * (total_syncs - 1) + sync_duration) / total_syncs
                )
            
            # 发送同步完成事件
            await self._emit_event("sync.completed", {
                "request_id": sync_request.request_id,
                "result": asdict(result)
            })
            
            self.logger.info(f"同步完成: {sync_request.request_id}, 成功: {len(synced_agents)}, 失败: {len(failed_agents)}")
            return result
            
        except Exception as e:
            self.logger.error(f"执行同步失败: {e}")
            
            # 创建失败结果
            sync_duration = time.time() - start_time
            result = SyncResult(
                request_id=sync_request.request_id,
                success=False,
                synced_agents=[],
                failed_agents=sync_request.target_agents,
                sync_duration=sync_duration,
                conflicts=[],
                error_message=str(e)
            )
            
            self.sync_results[sync_request.request_id] = result
            self.metrics["sync_requests_failed"] += 1
            
            return result
    
    async def _broadcast_sync_request(self, sync_request: SyncRequest) -> List[Dict[str, Any]]:
        """广播同步请求"""
        responses = []
        
        # 创建消息
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_id=self.synchronizer_id,
            receiver_id=None,  # 广播
            topic=f"sync_{sync_request.sync_type.value}",
            content={
                "request_id": sync_request.request_id,
                "sync_data": sync_request.sync_data,
                "requester_id": sync_request.requester_id
            },
            correlation_id=sync_request.request_id
        )
        
        # 发送消息
        await self.message_bus.publish_message(message)
        
        # 等待响应（简化实现）
        # 实际实现中需要更复杂的响应收集机制
        for agent_id in sync_request.target_agents:
            # 模拟响应
            response = {
                "agent_id": agent_id,
                "success": True,
                "sync_data": sync_request.sync_data
            }
            responses.append(response)
        
        return responses
    
    async def _resolve_conflicts(self, sync_request: SyncRequest, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解决同步冲突"""
        # 简化的冲突解决策略
        # 实际实现中需要更复杂的冲突解决算法
        
        if not conflicts:
            return sync_request.sync_data
        
        # 选择最新的数据
        latest_conflict = max(conflicts, key=lambda x: x.get("timestamp", ""))
        
        # 合并冲突数据
        resolved_data = sync_request.sync_data.copy()
        resolved_data.update(latest_conflict.get("data", {}))
        
        self.logger.info(f"冲突已解决: {len(conflicts)} 个冲突")
        
        return resolved_data
    
    async def _check_sync_need(self, sync_request: SyncRequest) -> bool:
        """检查是否需要同步"""
        # 简化的需求检查
        # 实际实现中需要检查状态差异
        
        sync_type = sync_request.sync_type
        agent_id = sync_request.requester_id
        
        # 检查上次同步时间
        if agent_id in self.sync_states:
            agent_states = self.sync_states[agent_id]
            if sync_type in agent_states:
                last_sync = agent_states[sync_type].last_sync_time
                time_since_sync = (datetime.now() - last_sync).total_seconds()
                
                config = self.sync_config.get(sync_type, {})
                interval = config.get("interval", 60)
                
                return time_since_sync > interval
        
        return True
    
    async def _elect_leader(self, sync_type: SyncType, candidate_agents: List[str]):
        """选举领导者"""
        if not candidate_agents:
            return
        
        # 简单的领导者选举：选择第一个候选者
        leader_id = candidate_agents[0]
        
        self.leaders[sync_type] = leader_id
        self.leader_heartbeats[leader_id] = datetime.now()
        self.metrics["leader_elections"] += 1
        
        # 更新所有智能体的状态
        for agent_id in candidate_agents:
            if agent_id in self.sync_states:
                agent_states = self.sync_states[agent_id]
                if sync_type in agent_states:
                    agent_states[sync_type].is_leader = (agent_id == leader_id)
                    agent_states[sync_type].leader_id = leader_id
        
        # 发送领导者选举事件
        await self._emit_event("leader.elected", {
            "sync_type": sync_type.value,
            "leader_id": leader_id,
            "candidates": candidate_agents
        })
        
        self.logger.info(f"领导者已选举: {sync_type.value} -> {leader_id}")
    
    async def _forward_to_leader(self, sync_request: SyncRequest, leader_id: str):
        """转发给领导者"""
        # 将请求转发给领导者智能体
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender_id=self.synchronizer_id,
            receiver_id=leader_id,
            topic=f"sync_leader_{sync_request.sync_type.value}",
            content={
                "original_request_id": sync_request.request_id,
                "sync_data": sync_request.sync_data,
                "requester_id": sync_request.requester_id,
                "target_agents": sync_request.target_agents
            }
        )
        
        await self.message_bus.publish_message(message)
    
    async def _finalize_quorum_sync(self, sync_request: SyncRequest, successful_responses: List[Dict[str, Any]]):
        """完成仲裁同步"""
        # 收集所有成功的响应数据
        all_sync_data = [sync_request.sync_data]
        for response in successful_responses:
            if "sync_data" in response:
                all_sync_data.append(response["sync_data"])
        
        # 合并数据
        final_data = self._merge_sync_data(all_sync_data)
        
        # 广播最终结果
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST,
            sender_id=self.synchronizer_id,
            receiver_id=None,
            topic=f"sync_final_{sync_request.sync_type.value}",
            content={
                "request_id": sync_request.request_id,
                "final_data": final_data,
                "synced_agents": [r.get("agent_id") for r in successful_responses]
            }
        )
        
        await self.message_bus.publish_message(message)
    
    def _merge_sync_data(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并同步数据"""
        merged = {}
        
        for data in data_list:
            if isinstance(data, dict):
                merged.update(data)
        
        return merged
    
    async def _fail_sync(self, sync_request: SyncRequest, reason: str):
        """标记同步失败"""
        result = SyncResult(
            request_id=sync_request.request_id,
            success=False,
            synced_agents=[],
            failed_agents=sync_request.target_agents,
            sync_duration=0.0,
            conflicts=[],
            error_message=reason
        )
        
        self.sync_results[sync_request.request_id] = result
        self.metrics["sync_requests_failed"] += 1
        
        await self._emit_event("sync.failed", {
            "request_id": sync_request.request_id,
            "reason": reason
        })
    
    async def _periodic_sync_scheduler(self):
        """定期同步调度器"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒调度一次
                
                # 为每个同步类型执行定期同步
                for sync_type, config in self.sync_config.items():
                    if self.sync_strategies[sync_type] == SyncStrategy.PERIODIC:
                        interval = config.get("interval", 60)
                        if self.sync_counter % (interval // 30) == 0:
                            await self._trigger_periodic_sync(sync_type)
                
                self.sync_counter += 1
                
            except Exception as e:
                self.logger.error(f"定期同步调度器出错: {e}")
    
    async def _trigger_periodic_sync(self, sync_type: SyncType):
        """触发定期同步"""
        # 收集所有相关智能体
        relevant_agents = set()
        for agent_states in self.sync_states.values():
            if sync_type in agent_states:
                relevant_agents.add(list(agent_states[sync_type].state_data.keys()))
        
        if relevant_agents:
            # 创建定期同步请求
            await self.request_sync(
                sync_type=sync_type,
                requester_id="system",
                target_agents=list(relevant_agents),
                sync_data={"type": "periodic_sync"},
                priority=1,
                timeout=10.0
            )
    
    async def _leader_monitor(self):
        """领导者监控后台任务"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                current_time = datetime.now()
                
                # 检查领导者心跳
                expired_leaders = []
                for sync_type, leader_id in self.leaders.items():
                    last_heartbeat = self.leader_heartbeats.get(leader_id)
                    if last_heartbeat and (current_time - last_heartbeat).total_seconds() > 90:
                        expired_leaders.append(sync_type)
                
                # 处理过期的领导者
                for sync_type in expired_leaders:
                    self.logger.warning(f"领导者 {self.leaders[sync_type]} 已过期，重新选举")
                    del self.leaders[sync_type]
                    del self.leader_heartbeats[self.leaders.get(sync_type, "")]
                
            except Exception as e:
                self.logger.error(f"领导者监控任务出错: {e}")
    
    async def _conflict_resolver(self):
        """冲突解决器后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次冲突
                
                # 分析最近的同步结果
                recent_results = [
                    result for result in self.sync_results.values()
                    if result.conflicts and (datetime.now() - datetime.now()).total_seconds() < 3600
                ]
                
                if recent_results:
                    await self._analyze_conflict_patterns(recent_results)
                
            except Exception as e:
                self.logger.error(f"冲突解决器任务出错: {e}")
    
    async def _analyze_conflict_patterns(self, results: List[SyncResult]):
        """分析冲突模式"""
        conflict_types = defaultdict(int)
        
        for result in results:
            for conflict in result.conflicts:
                conflict_type = conflict.get("type", "unknown")
                conflict_types[conflict_type] += 1
        
        # 发送冲突分析事件
        await self._emit_event("sync.conflicts_analyzed", {
            "conflict_types": dict(conflict_types),
            "total_conflicts": sum(conflict_types.values())
        })
        
        if conflict_types:
            self.logger.info(f"冲突模式分析: {dict(conflict_types)}")
    
    async def _clock_synchronizer(self):
        """时钟同步器"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟同步一次
                
                # 收集所有智能体的时钟信息
                clock_data = {}
                for agent_id, agent_states in self.sync_states.items():
                    if SyncType.CLOCK_SYNC in agent_states:
                        state = agent_states[SyncType.CLOCK_SYNC]
                        clock_data[agent_id] = {
                            "local_time": state.last_sync_time,
                            "offset": state.state_data.get("clock_offset", 0)
                        }
                
                # 计算平均时钟偏移
                if clock_data:
                    offsets = [data["offset"] for data in clock_data.values() if "offset" in data]
                    if offsets:
                        avg_offset = sum(offsets) / len(offsets)
                        
                        # 广播时钟同步信息
                        await self._emit_event("sync.clock_sync", {
                            "average_offset": avg_offset,
                            "synced_agents": list(clock_data.keys())
                        })
                
            except Exception as e:
                self.logger.error(f"时钟同步器任务出错: {e}")
    
    async def _sync_metrics_collector(self):
        """同步指标收集器"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟收集一次
                
                # 计算同步效率
                total_requests = self.metrics["sync_requests_total"]
                successful_requests = self.metrics["sync_requests_success"]
                
                if total_requests > 0:
                    self.metrics["sync_efficiency"] = successful_requests / total_requests
                
                # 发送指标事件
                await self._emit_event("sync.metrics_collected", {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": dict(self.metrics)
                })
                
            except Exception as e:
                self.logger.error(f"同步指标收集器任务出错: {e}")
    
    async def _handle_sync_request(self, message: Message):
        """处理同步请求"""
        content = message.content
        request_id = content.get("request_id")
        
        if request_id in self.sync_requests:
            sync_request = self.sync_requests[request_id]
            
            # 执行同步
            result = await self._execute_sync(sync_request)
            
            # 发送响应
            response_message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.RESPONSE,
                sender_id=self.synchronizer_id,
                receiver_id=message.sender_id,
                topic=message.topic,
                content={
                    "request_id": request_id,
                    "result": asdict(result)
                },
                correlation_id=message.correlation_id
            )
            
            await self.message_bus.publish_message(response_message)
    
    async def _handle_sync_response(self, message: Message):
        """处理同步响应"""
        # 响应处理逻辑已在广播请求中简化实现
        pass
    
    async def _handle_sync_heartbeat(self, message: Message):
        """处理同步心跳"""
        agent_id = message.sender_id
        sync_type_str = message.topic.split("_")[-1]  # 提取同步类型
        
        try:
            sync_type = SyncType(sync_type_str)
            self.leader_heartbeats[agent_id] = datetime.now()
            
            # 如果是领导者，更新领导者状态
            if self.leaders.get(sync_type) == agent_id:
                # 更新领导者状态
                pass
                
        except ValueError:
            self.logger.warning(f"未知的同步类型: {sync_type_str}")
    
    async def _handle_agent_join(self, message: Message):
        """处理智能体加入"""
        agent_id = message.content.get("agent_id")
        if agent_id:
            # 初始化智能体的同步状态
            for sync_type in SyncType:
                self.sync_states[agent_id][sync_type] = SyncState(
                    agent_id=agent_id,
                    sync_type=sync_type,
                    last_sync_time=datetime.now(),
                    sync_version=0,
                    state_data={},
                    checksum=""
                )
    
    async def _handle_agent_leave(self, message: Message):
        """处理智能体离开"""
        agent_id = message.content.get("agent_id")
        if agent_id:
            # 清理智能体的同步状态
            if agent_id in self.sync_states:
                del self.sync_states[agent_id]
            
            # 检查是否是领导者
            expired_leaders = [
                sync_type for sync_type, leader_id in self.leaders.items()
                if leader_id == agent_id
            ]
            
            for sync_type in expired_leaders:
                del self.leaders[sync_type]
                if agent_id in self.leader_heartbeats:
                    del self.leader_heartbeats[agent_id]
    
    def get_sync_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """获取同步状态"""
        if agent_id:
            # 返回特定智能体的同步状态
            agent_states = self.sync_states.get(agent_id, {})
            return {
                "agent_id": agent_id,
                "sync_states": {
                    sync_type.value: {
                        "last_sync_time": state.last_sync_time.isoformat(),
                        "sync_version": state.sync_version,
                        "is_leader": state.is_leader,
                        "leader_id": state.leader_id
                    }
                    for sync_type, state in agent_states.items()
                }
            }
        else:
            # 返回全局同步状态
            return {
                "synchronizer_id": self.synchronizer_id,
                "total_agents": len(self.sync_states),
                "active_sync_requests": len(self.sync_requests),
                "completed_sync_results": len(self.sync_results),
                "leaders": {st.value: leader for st, leader in self.leaders.items()},
                "metrics": dict(self.metrics),
                "sync_strategies": {st.value: strategy.value for st, strategy in self.sync_strategies.items()}
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
        
        # 通过消息总线发送事件
        if self.message_bus:
            await self.message_bus.publish_event(event_type, data)
    
    async def shutdown(self):
        """关闭智能体同步器"""
        try:
            self.logger.info("关闭智能体同步器...")
            
            # 取消所有同步请求
            self.sync_requests.clear()
            
            # 清理状态
            self.sync_states.clear()
            self.leaders.clear()
            self.leader_heartbeats.clear()
            
            self.logger.info("智能体同步器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭智能体同步器时出错: {e}")