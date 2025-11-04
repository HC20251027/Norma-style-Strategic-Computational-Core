"""
对话协调智能体 - 负责对话管理和多智能体协调
"""
import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from ..core.base_agent import BaseAgent, Task, TaskPriority, AgentMessage


@dataclass
class ConversationContext:
    """对话上下文"""
    conversation_id: str
    user_id: str
    session_id: str
    context_data: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    participants: List[str]
    state: str  # active, paused, ended


@dataclass
class MessageRouting:
    """消息路由"""
    message_id: str
    conversation_id: str
    sender_id: str
    target_agents: List[str]
    routing_strategy: str  # broadcast, round_robin, priority, load_balanced
    priority: TaskPriority
    timestamp: datetime


class ConversationCoordinatorAgent(BaseAgent):
    """对话协调智能体"""
    
    def __init__(self, agent_id: str = None, config: Dict[str, Any] = None):
        super().__init__(
            agent_id=agent_id or f"conversation-coordinator-{int(time.time())}",
            agent_type="conversation_coordinator",
            config=config or {}
        )
        self.active_conversations = {}  # 活跃对话
        self.conversation_queue = defaultdict(deque)  # 对话队列
        self.message_router = {}  # 消息路由器
        self.agent_availability = {}  # 智能体可用性
        self.conversation_history = defaultdict(list)  # 对话历史
        self.coordination_rules = self.config.get("coordination_rules", {})
        self.routing_strategies = ["broadcast", "round_robin", "priority", "load_balanced"]
        self.logger = logging.getLogger("agent.conversation.coordinator")
        
    async def initialize(self) -> bool:
        """初始化对话协调智能体"""
        try:
            self.logger.info("初始化对话协调智能体...")
            
            # 设置协调能力
            self.capabilities = [
                "conversation_management",
                "multi_agent_coordination",
                "message_routing",
                "context_tracking",
                "load_balancing",
                "conflict_resolution",
                "session_management",
                "dialogue_flow_control"
            ]
            
            # 启动协调循环
            asyncio.create_task(self._coordination_loop())
            
            # 启动对话监控
            asyncio.create_task(self._conversation_monitor())
            
            # 启动负载均衡
            asyncio.create_task(self._load_balancer())
            
            self.logger.info("对话协调智能体初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            return False
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """处理协调任务"""
        try:
            task_type = task.payload.get("type")
            
            if task_type == "create_conversation":
                return await self._create_conversation(task.payload)
            elif task_type == "join_conversation":
                return await self._join_conversation(task.payload)
            elif task_type == "route_message":
                return await self._route_message(task.payload)
            elif task_type == "coordinate_agents":
                return await self._coordinate_agents(task.payload)
            elif task_type == "resolve_conflict":
                return await self._resolve_conflict(task.payload)
            elif task_type == "get_conversation_status":
                return await self._get_conversation_status(task.payload.get("conversation_id"))
            elif task_type == "update_context":
                return await self._update_context(task.payload)
            elif task_type == "end_conversation":
                return await self._end_conversation(task.payload.get("conversation_id"))
            elif task_type == "get_agent_availability":
                return await self._get_agent_availability()
            elif task_type == "set_routing_strategy":
                return await self._set_routing_strategy(task.payload)
            else:
                return {"status": "error", "message": f"未知的任务类型: {task_type}"}
                
        except Exception as e:
            self.logger.error(f"任务处理失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self) -> bool:
        """关闭对话协调智能体"""
        try:
            self.logger.info("关闭对话协调智能体...")
            
            # 结束所有活跃对话
            for conv_id in list(self.active_conversations.keys()):
                await self._end_conversation(conv_id)
            
            await super().stop()
            self.logger.info("对话协调智能体已关闭")
            return True
        except Exception as e:
            self.logger.error(f"关闭失败: {e}")
            return False
    
    async def _create_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """创建对话"""
        try:
            user_id = payload.get("user_id")
            participants = payload.get("participants", [])
            context_data = payload.get("context_data", {})
            
            if not user_id:
                return {"status": "error", "message": "用户ID不能为空"}
            
            conversation_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            
            # 创建对话上下文
            conversation = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id,
                context_data=context_data,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                participants=participants + [self.agent_id],
                state="active"
            )
            
            # 注册对话
            self.active_conversations[conversation_id] = conversation
            
            # 初始化对话历史
            self.conversation_history[conversation_id] = []
            
            self.logger.info(f"创建对话: {conversation_id}")
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "session_id": session_id,
                "participants": conversation.participants
            }
            
        except Exception as e:
            self.logger.error(f"创建对话失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _join_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """加入对话"""
        try:
            conversation_id = payload.get("conversation_id")
            agent_id = payload.get("agent_id")
            
            if not conversation_id or not agent_id:
                return {"status": "error", "message": "对话ID和智能体ID不能为空"}
            
            if conversation_id not in self.active_conversations:
                return {"status": "error", "message": f"对话不存在: {conversation_id}"}
            
            conversation = self.active_conversations[conversation_id]
            
            # 添加参与者
            if agent_id not in conversation.participants:
                conversation.participants.append(agent_id)
                conversation.last_updated = datetime.now()
                
                self.logger.info(f"智能体 {agent_id} 加入对话 {conversation_id}")
                
                return {
                    "status": "success",
                    "conversation_id": conversation_id,
                    "participants": conversation.participants
                }
            else:
                return {"status": "info", "message": "智能体已在对话中"}
                
        except Exception as e:
            self.logger.error(f"加入对话失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _route_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """路由消息"""
        try:
            conversation_id = payload.get("conversation_id")
            sender_id = payload.get("sender_id")
            message_content = payload.get("message")
            target_agents = payload.get("target_agents", [])
            routing_strategy = payload.get("routing_strategy", "broadcast")
            priority = TaskPriority(payload.get("priority", 2))
            
            if not conversation_id or not sender_id or not message_content:
                return {"status": "error", "message": "必要参数缺失"}
            
            if conversation_id not in self.active_conversations:
                return {"status": "error", "message": f"对话不存在: {conversation_id}"}
            
            # 创建消息路由
            message_routing = MessageRouting(
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                sender_id=sender_id,
                target_agents=target_agents,
                routing_strategy=routing_strategy,
                priority=priority,
                timestamp=datetime.now()
            )
            
            # 执行路由策略
            if routing_strategy == "broadcast":
                return await self._broadcast_routing(message_routing, message_content)
            elif routing_strategy == "round_robin":
                return await self._round_robin_routing(message_routing, message_content)
            elif routing_strategy == "priority":
                return await self._priority_routing(message_routing, message_content)
            elif routing_strategy == "load_balanced":
                return await self._load_balanced_routing(message_routing, message_content)
            else:
                return {"status": "error", "message": f"未知的路由策略: {routing_strategy}"}
                
        except Exception as e:
            self.logger.error(f"消息路由失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _broadcast_routing(self, routing: MessageRouting, message_content: str) -> Dict[str, Any]:
        """广播路由"""
        try:
            conversation = self.active_conversations[routing.conversation_id]
            delivered_count = 0
            
            # 广播给所有参与者（除了发送者）
            for agent_id in conversation.participants:
                if agent_id != routing.sender_id:
                    # 模拟消息发送
                    success = await self._deliver_message(agent_id, message_content, routing)
                    if success:
                        delivered_count += 1
            
            return {
                "status": "success",
                "routing_strategy": "broadcast",
                "delivered_count": delivered_count,
                "total_participants": len(conversation.participants) - 1
            }
            
        except Exception as e:
            self.logger.error(f"广播路由失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _round_robin_routing(self, routing: MessageRouting, message_content: str) -> Dict[str, Any]:
        """轮询路由"""
        try:
            conversation = self.active_conversations[routing.conversation_id]
            available_agents = [agent for agent in conversation.participants if agent != routing.sender_id]
            
            if not available_agents:
                return {"status": "error", "message": "没有可用的智能体"}
            
            # 简单的轮询实现（实际中需要维护轮询状态）
            selected_agent = available_agents[len(routing.message_id) % len(available_agents)]
            
            success = await self._deliver_message(selected_agent, message_content, routing)
            
            return {
                "status": "success",
                "routing_strategy": "round_robin",
                "selected_agent": selected_agent,
                "delivered": success
            }
            
        except Exception as e:
            self.logger.error(f"轮询路由失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _priority_routing(self, routing: MessageRouting, message_content: str) -> Dict[str, Any]:
        """优先级路由"""
        try:
            conversation = self.active_conversations[routing.conversation_id]
            available_agents = [agent for agent in conversation.participants if agent != routing.sender_id]
            
            # 根据优先级选择智能体
            if routing.priority == TaskPriority.CRITICAL:
                # 关键任务选择最可靠的智能体
                selected_agent = available_agents[0] if available_agents else None
            elif routing.priority == TaskPriority.HIGH:
                # 高优先级选择负载最低的智能体
                selected_agent = await self._select_lowest_load_agent(available_agents)
            else:
                # 普通优先级使用轮询
                selected_agent = available_agents[len(routing.message_id) % len(available_agents)] if available_agents else None
            
            if selected_agent:
                success = await self._deliver_message(selected_agent, message_content, routing)
                return {
                    "status": "success",
                    "routing_strategy": "priority",
                    "selected_agent": selected_agent,
                    "priority": routing.priority.name,
                    "delivered": success
                }
            else:
                return {"status": "error", "message": "没有可用的智能体"}
                
        except Exception as e:
            self.logger.error(f"优先级路由失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _load_balanced_routing(self, routing: MessageRouting, message_content: str) -> Dict[str, Any]:
        """负载均衡路由"""
        try:
            conversation = self.active_conversations[routing.conversation_id]
            available_agents = [agent for agent in conversation.participants if agent != routing.sender_id]
            
            # 选择负载最低的智能体
            selected_agent = await self._select_lowest_load_agent(available_agents)
            
            if selected_agent:
                success = await self._deliver_message(selected_agent, message_content, routing)
                return {
                    "status": "success",
                    "routing_strategy": "load_balanced",
                    "selected_agent": selected_agent,
                    "delivered": success
                }
            else:
                return {"status": "error", "message": "没有可用的智能体"}
                
        except Exception as e:
            self.logger.error(f"负载均衡路由失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _coordinate_agents(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """协调智能体"""
        try:
            coordination_type = payload.get("coordination_type")
            agents = payload.get("agents", [])
            task_description = payload.get("task_description")
            
            if not coordination_type or not agents:
                return {"status": "error", "message": "协调类型和智能体列表不能为空"}
            
            coordination_result = {
                "coordination_id": str(uuid.uuid4()),
                "coordination_type": coordination_type,
                "agents": agents,
                "task_description": task_description,
                "status": "initiated",
                "participants": [],
                "start_time": datetime.now().isoformat()
            }
            
            # 根据协调类型执行不同逻辑
            if coordination_type == "collaborative":
                # 协作模式：所有智能体共同完成一个任务
                coordination_result["participants"] = agents
                coordination_result["status"] = "collaborating"
                
            elif coordination_type == "sequential":
                # 顺序模式：智能体按顺序执行任务
                coordination_result["participants"] = agents
                coordination_result["status"] = "sequencing"
                
            elif coordination_type == "competitive":
                # 竞争模式：多个智能体竞争执行任务
                coordination_result["participants"] = agents[:1]  # 选择一个
                coordination_result["status"] = "competing"
            
            self.logger.info(f"启动智能体协调: {coordination_result['coordination_id']}")
            
            return {
                "status": "success",
                "coordination": coordination_result
            }
            
        except Exception as e:
            self.logger.error(f"智能体协调失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _resolve_conflict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """解决冲突"""
        try:
            conflict_type = payload.get("conflict_type")
            conflicting_agents = payload.get("conflicting_agents", [])
            conflict_details = payload.get("conflict_details", {})
            
            if not conflict_type or not conflicting_agents:
                return {"status": "error", "message": "冲突类型和冲突智能体不能为空"}
            
            resolution_strategy = self.coordination_rules.get(f"{conflict_type}_resolution", "arbitration")
            
            resolution = {
                "conflict_id": str(uuid.uuid4()),
                "conflict_type": conflict_type,
                "conflicting_agents": conflicting_agents,
                "resolution_strategy": resolution_strategy,
                "resolution_details": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # 根据冲突类型选择解决策略
            if conflict_type == "resource_contention":
                # 资源争用：按优先级分配
                resolution["resolution_details"] = {
                    "strategy": "priority_based_allocation",
                    "winner": conflicting_agents[0],  # 简化实现
                    "reason": "基于优先级分配"
                }
                
            elif conflict_type == "message_conflict":
                # 消息冲突：消息排序
                resolution["resolution_details"] = {
                    "strategy": "timestamp_ordering",
                    "order": "chronological",
                    "reason": "按时间戳排序"
                }
                
            elif conflict_type == "task_overlap":
                # 任务重叠：任务分配
                resolution["resolution_details"] = {
                    "strategy": "task_splitting",
                    "assigned_tasks": {agent: f"task_{i}" for i, agent in enumerate(conflicting_agents)},
                    "reason": "任务分割分配"
                }
            
            self.logger.info(f"解决冲突: {resolution['conflict_id']}")
            
            return {
                "status": "success",
                "resolution": resolution
            }
            
        except Exception as e:
            self.logger.error(f"冲突解决失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话状态"""
        try:
            if conversation_id not in self.active_conversations:
                return {"status": "error", "message": f"对话不存在: {conversation_id}"}
            
            conversation = self.active_conversations[conversation_id]
            history = self.conversation_history.get(conversation_id, [])
            
            return {
                "status": "success",
                "conversation": asdict(conversation),
                "message_count": len(history),
                "participants_count": len(conversation.participants),
                "duration": (datetime.now() - conversation.created_at).total_seconds()
            }
            
        except Exception as e:
            self.logger.error(f"获取对话状态失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _update_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """更新上下文"""
        try:
            conversation_id = payload.get("conversation_id")
            context_updates = payload.get("context_updates", {})
            
            if not conversation_id or not context_updates:
                return {"status": "error", "message": "对话ID和上下文更新不能为空"}
            
            if conversation_id not in self.active_conversations:
                return {"status": "error", "message": f"对话不存在: {conversation_id}"}
            
            conversation = self.active_conversations[conversation_id]
            conversation.context_data.update(context_updates)
            conversation.last_updated = datetime.now()
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "updated_context": context_updates
            }
            
        except Exception as e:
            self.logger.error(f"更新上下文失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """结束对话"""
        try:
            if conversation_id not in self.active_conversations:
                return {"status": "error", "message": f"对话不存在: {conversation_id}"}
            
            conversation = self.active_conversations[conversation_id]
            conversation.state = "ended"
            conversation.last_updated = datetime.now()
            
            # 移动到历史记录
            self.conversation_history[conversation_id] = list(self.conversation_history[conversation_id])
            
            # 从活跃对话中移除
            del self.active_conversations[conversation_id]
            
            self.logger.info(f"结束对话: {conversation_id}")
            
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "end_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"结束对话失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_agent_availability(self) -> Dict[str, Any]:
        """获取智能体可用性"""
        try:
            return {
                "status": "success",
                "agent_availability": self.agent_availability,
                "total_agents": len(self.agent_availability),
                "available_agents": len([a for a in self.agent_availability.values() if a.get("available", False)])
            }
            
        except Exception as e:
            self.logger.error(f"获取智能体可用性失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _set_routing_strategy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """设置路由策略"""
        try:
            strategy_name = payload.get("strategy_name")
            strategy_config = payload.get("strategy_config", {})
            
            if strategy_name not in self.routing_strategies:
                return {"status": "error", "message": f"不支持的路由策略: {strategy_name}"}
            
            self.coordination_rules[f"{strategy_name}_config"] = strategy_config
            
            return {
                "status": "success",
                "strategy_name": strategy_name,
                "strategy_config": strategy_config
            }
            
        except Exception as e:
            self.logger.error(f"设置路由策略失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _deliver_message(self, agent_id: str, message_content: str, routing: MessageRouting) -> bool:
        """交付消息"""
        try:
            # 模拟消息交付
            message = AgentMessage(
                id=str(uuid.uuid4()),
                sender_id=routing.sender_id,
                receiver_id=agent_id,
                message_type="coordinated_message",
                payload={"content": message_content, "conversation_id": routing.conversation_id},
                timestamp=datetime.now(),
                priority=routing.priority
            )
            
            # 添加到对话历史
            self.conversation_history[routing.conversation_id].append(asdict(message))
            
            # 模拟成功交付
            return True
            
        except Exception as e:
            self.logger.error(f"消息交付失败: {e}")
            return False
    
    async def _select_lowest_load_agent(self, agents: List[str]) -> Optional[str]:
        """选择负载最低的智能体"""
        if not agents:
            return None
        
        # 简化实现：随机选择
        # 实际实现中应该查询每个智能体的负载情况
        return agents[0]
    
    async def _coordination_loop(self):
        """协调循环"""
        while self.is_running:
            try:
                # 定期检查和协调对话
                await asyncio.sleep(30)  # 30秒检查一次
                
                # 清理过期的对话
                cutoff_time = datetime.now() - timedelta(hours=24)
                expired_conversations = [
                    conv_id for conv_id, conv in self.active_conversations.items()
                    if conv.last_updated < cutoff_time
                ]
                
                for conv_id in expired_conversations:
                    await self._end_conversation(conv_id)
                
            except Exception as e:
                self.logger.error(f"协调循环错误: {e}")
                await asyncio.sleep(30)
    
    async def _conversation_monitor(self):
        """对话监控器"""
        while self.is_running:
            try:
                # 监控对话状态
                for conv_id, conversation in self.active_conversations.items():
                    # 检查对话活跃度
                    if (datetime.now() - conversation.last_updated).total_seconds() > 3600:  # 1小时无活动
                        conversation.state = "idle"
                
                await asyncio.sleep(60)  # 1分钟监控一次
                
            except Exception as e:
                self.logger.error(f"对话监控错误: {e}")
                await asyncio.sleep(60)
    
    async def _load_balancer(self):
        """负载均衡器"""
        while self.is_running:
            try:
                # 定期更新智能体负载信息
                await asyncio.sleep(120)  # 2分钟更新一次
                
                # 这里应该实现真实的负载查询
                # 简化实现：模拟负载数据
                for agent_id in self.agent_availability:
                    self.agent_availability[agent_id]["load"] = 0.5  # 模拟负载
                
            except Exception as e:
                self.logger.error(f"负载均衡错误: {e}")
                await asyncio.sleep(120)