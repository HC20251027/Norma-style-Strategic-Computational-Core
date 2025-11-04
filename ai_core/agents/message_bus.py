#!/usr/bin/env python3
"""
消息总线
提供智能体间异步消息传递的通信基础设施

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pickle
import zlib

from ..utils.logger import MultiAgentLogger


class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    SYNC = "sync"
    ERROR = "error"


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """消息定义"""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: Optional[str]  # None表示广播
    topic: str
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = None
    ttl: Optional[float] = None  # 生存时间（秒）
    correlation_id: Optional[str] = None  # 关联ID，用于请求-响应模式
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl
    
    @property
    def size_bytes(self) -> int:
        """消息大小（字节）"""
        try:
            serialized = json.dumps(asdict(self), default=str)
            return len(serialized.encode('utf-8'))
        except:
            return 0


@dataclass
class Subscription:
    """订阅信息"""
    subscription_id: str
    subscriber_id: str
    topic: str
    callback: Callable
    filter_func: Optional[Callable] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class MessageBus:
    """消息总线"""
    
    def __init__(self, buffer_size: int = 1000):
        """初始化消息总线
        
        Args:
            buffer_size: 消息缓冲区大小
        """
        self.bus_id = str(uuid.uuid4())
        self.buffer_size = buffer_size
        
        # 初始化日志
        self.logger = MultiAgentLogger("message_bus")
        
        # 订阅管理
        self.subscriptions: Dict[str, Subscription] = {}
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.subscriber_topics: Dict[str, Set[str]] = defaultdict(set)
        
        # 消息队列
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.broadcast_queue = deque(maxlen=buffer_size)
        
        # 统计信息
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_broadcast": 0,
            "subscriptions_created": 0,
            "subscriptions_removed": 0,
            "queue_sizes": defaultdict(int),
            "average_latency": 0.0,
            "throughput_per_second": 0.0
        }
        
        # 性能监控
        self.message_timestamps = deque(maxlen=1000)
        self.latency_history = deque(maxlen=100)
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger.info(f"消息总线 {self.bus_id} 已初始化")
    
    async def initialize(self) -> bool:
        """初始化消息总线"""
        try:
            self.logger.info("初始化消息总线...")
            
            # 启动后台任务
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._cleanup_expired_messages())
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._health_monitor())
            
            self.logger.info("消息总线初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"消息总线初始化失败: {e}")
            return False
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable,
        subscriber_id: Optional[str] = None,
        filter_func: Optional[Callable] = None
    ) -> str:
        """订阅主题
        
        Args:
            topic: 主题名称
            callback: 回调函数
            subscriber_id: 订阅者ID，如果为None则使用回调函数的标识
            filter_func: 消息过滤函数
            
        Returns:
            订阅ID
        """
        try:
            # 生成订阅ID
            subscription_id = str(uuid.uuid4())
            
            # 确定订阅者ID
            if subscriber_id is None:
                subscriber_id = getattr(callback, '__name__', f"subscriber_{subscription_id[:8]}")
            
            # 创建订阅
            subscription = Subscription(
                subscription_id=subscription_id,
                subscriber_id=subscriber_id,
                topic=topic,
                callback=callback,
                filter_func=filter_func
            )
            
            # 注册订阅
            self.subscriptions[subscription_id] = subscription
            self.topic_subscribers[topic].add(subscriber_id)
            self.subscriber_topics[subscriber_id].add(topic)
            
            self.metrics["subscriptions_created"] += 1
            
            # 发送订阅事件
            await self._emit_event("subscription.created", {
                "subscription_id": subscription_id,
                "topic": topic,
                "subscriber_id": subscriber_id
            })
            
            self.logger.info(f"订阅已创建: {topic} -> {subscriber_id}")
            return subscription_id
            
        except Exception as e:
            self.logger.error(f"创建订阅失败: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        try:
            if subscription_id not in self.subscriptions:
                self.logger.warning(f"订阅 {subscription_id} 不存在")
                return False
            
            subscription = self.subscriptions[subscription_id]
            
            # 移除订阅
            del self.subscriptions[subscription_id]
            self.topic_subscribers[subscription.topic].discard(subscription.subscriber_id)
            self.subscriber_topics[subscription.subscriber_id].discard(subscription.topic)
            
            # 清理空的主题和订阅者
            if not self.topic_subscribers[subscription.topic]:
                del self.topic_subscribers[subscription.topic]
            
            if not self.subscriber_topics[subscription.subscriber_id]:
                del self.subscriber_topics[subscription.subscriber_id]
            
            self.metrics["subscriptions_removed"] += 1
            
            # 发送取消订阅事件
            await self._emit_event("subscription.removed", {
                "subscription_id": subscription_id,
                "topic": subscription.topic,
                "subscriber_id": subscription.subscriber_id
            })
            
            self.logger.info(f"订阅已取消: {subscription.topic} -> {subscription.subscriber_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"取消订阅失败: {e}")
            return False
    
    async def publish_message(self, message: Message) -> bool:
        """发布消息"""
        try:
            # 检查消息是否过期
            if message.is_expired:
                self.logger.warning(f"消息 {message.message_id} 已过期，丢弃")
                return False
            
            # 序列化消息（压缩）
            serialized_message = await self._serialize_message(message)
            
            if message.receiver_id:
                # 点对点消息
                await self._deliver_direct_message(message, serialized_message)
            else:
                # 广播消息
                await self._deliver_broadcast_message(message, serialized_message)
            
            self.metrics["messages_sent"] += 1
            self.message_timestamps.append(datetime.now())
            
            return True
            
        except Exception as e:
            self.logger.error(f"发布消息失败: {e}")
            return False
    
    async def publish_event(self, topic: str, data: Any, sender_id: str = "system") -> bool:
        """发布事件"""
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EVENT,
            sender_id=sender_id,
            receiver_id=None,  # 广播
            topic=topic,
            content=data,
            priority=MessagePriority.NORMAL
        )
        
        return await self.publish_message(message)
    
    async def send_request(
        self,
        receiver_id: str,
        topic: str,
        content: Any,
        sender_id: str,
        timeout: float = 30.0
    ) -> Optional[Message]:
        """发送请求并等待响应"""
        try:
            correlation_id = str(uuid.uuid4())
            
            # 创建请求消息
            request_message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.REQUEST,
                sender_id=sender_id,
                receiver_id=receiver_id,
                topic=topic,
                content=content,
                priority=MessagePriority.HIGH,
                correlation_id=correlation_id
            )
            
            # 等待响应的事件
            response_future = asyncio.Future()
            
            # 创建临时订阅来接收响应
            def response_handler(message: Message):
                if (message.message_type == MessageType.RESPONSE and 
                    message.correlation_id == correlation_id):
                    if not response_future.done():
                        response_future.set_result(message)
            
            # 订阅响应
            response_subscription_id = await self.subscribe(
                topic=f"response_{receiver_id}",
                callback=response_handler,
                subscriber_id=sender_id
            )
            
            # 发送请求
            await self.publish_message(request_message)
            
            # 等待响应或超时
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                self.logger.warning(f"请求超时: {correlation_id}")
                return None
            finally:
                # 清理临时订阅
                await self.unsubscribe(response_subscription_id)
                
        except Exception as e:
            self.logger.error(f"发送请求失败: {e}")
            return None
    
    async def _deliver_direct_message(self, message: Message, serialized_message: bytes):
        """投递直接消息"""
        receiver_id = message.receiver_id
        
        # 投递给指定接收者
        queue_key = f"direct_{receiver_id}"
        self.message_queues[queue_key].append(serialized_message)
        
        # 查找相关的订阅
        for subscription in self.subscriptions.values():
            if (subscription.subscriber_id == receiver_id and 
                (subscription.topic == message.topic or subscription.topic == f"direct_{receiver_id}")):
                
                # 检查过滤器
                if subscription.filter_func and not subscription.filter_func(message):
                    continue
                
                # 异步调用回调
                asyncio.create_task(self._invoke_callback(subscription, message))
    
    async def _deliver_broadcast_message(self, message: Message, serialized_message: bytes):
        """投递广播消息"""
        # 添加到广播队列
        self.broadcast_queue.append(serialized_message)
        
        # 投递给所有订阅者
        subscribers = self.topic_subscribers.get(message.topic, set())
        
        for subscriber_id in subscribers:
            queue_key = f"direct_{subscriber_id}"
            self.message_queues[queue_key].append(serialized_message)
        
        # 查找相关订阅
        for subscription in self.subscriptions.values():
            if subscription.topic == message.topic:
                # 检查过滤器
                if subscription.filter_func and not subscription.filter_func(message):
                    continue
                
                # 异步调用回调
                asyncio.create_task(self._invoke_callback(subscription, message))
        
        self.metrics["messages_broadcast"] += len(subscribers)
    
    async def _invoke_callback(self, subscription: Subscription, message: Message):
        """调用订阅回调"""
        try:
            start_time = datetime.now()
            
            # 调用回调函数
            if asyncio.iscoroutinefunction(subscription.callback):
                await subscription.callback(message)
            else:
                subscription.callback(message)
            
            # 计算延迟
            latency = (datetime.now() - start_time).total_seconds()
            self.latency_history.append(latency)
            
            self.metrics["messages_received"] += 1
            
        except Exception as e:
            self.logger.error(f"调用回调失败 {subscription.subscription_id}: {e}")
    
    async def _serialize_message(self, message: Message) -> bytes:
        """序列化消息"""
        try:
            # 转换为字典
            message_dict = asdict(message)
            
            # JSON序列化
            json_str = json.dumps(message_dict, default=str)
            
            # 压缩
            compressed = zlib.compress(json_str.encode('utf-8'))
            
            return compressed
            
        except Exception as e:
            self.logger.error(f"序列化消息失败: {e}")
            return b""
    
    async def _deserialize_message(self, data: bytes) -> Optional[Message]:
        """反序列化消息"""
        try:
            # 解压
            decompressed = zlib.decompress(data)
            
            # JSON反序列化
            json_str = decompressed.decode('utf-8')
            message_dict = json.loads(json_str)
            
            # 转换为Message对象
            # 处理datetime字段
            if 'timestamp' in message_dict and isinstance(message_dict['timestamp'], str):
                message_dict['timestamp'] = datetime.fromisoformat(message_dict['timestamp'])
            
            # 处理枚举字段
            if 'message_type' in message_dict:
                message_dict['message_type'] = MessageType(message_dict['message_type'])
            
            if 'priority' in message_dict:
                message_dict['priority'] = MessagePriority(message_dict['priority'])
            
            return Message(**message_dict)
            
        except Exception as e:
            self.logger.error(f"反序列化消息失败: {e}")
            return None
    
    async def _message_processor(self):
        """消息处理器后台任务"""
        while True:
            try:
                await asyncio.sleep(0.01)  # 10ms处理间隔
                
                # 处理各个消息队列
                for queue_key, queue in self.message_queues.items():
                    while queue:
                        try:
                            # 获取消息
                            message_data = queue.popleft()
                            
                            # 反序列化消息
                            message = await self._deserialize_message(message_data)
                            if message:
                                # 检查消息是否过期
                                if not message.is_expired:
                                    # 投递给订阅者
                                    await self._deliver_to_subscribers(message)
                                else:
                                    self.logger.debug(f"丢弃过期消息: {message.message_id}")
                            
                        except Exception as e:
                            self.logger.error(f"处理消息队列 {queue_key} 出错: {e}")
                
                # 处理广播队列
                while self.broadcast_queue:
                    try:
                        message_data = self.broadcast_queue.popleft()
                        message = await self._deserialize_message(message_data)
                        if message and not message.is_expired:
                            await self._deliver_broadcast(message)
                    except Exception as e:
                        self.logger.error(f"处理广播队列出错: {e}")
                
            except Exception as e:
                self.logger.error(f"消息处理器出错: {e}")
    
    async def _deliver_to_subscribers(self, message: Message):
        """投递给订阅者"""
        # 查找相关订阅
        for subscription in self.subscriptions.values():
            if subscription.topic == message.topic:
                # 检查过滤器
                if subscription.filter_func and not subscription.filter_func(message):
                    continue
                
                # 异步调用回调
                asyncio.create_task(self._invoke_callback(subscription, message))
    
    async def _deliver_broadcast(self, message: Message):
        """投递广播"""
        subscribers = self.topic_subscribers.get(message.topic, set())
        
        for subscriber_id in subscribers:
            queue_key = f"direct_{subscriber_id}"
            serialized = await self._serialize_message(message)
            self.message_queues[queue_key].append(serialized)
        
        # 投递给订阅者
        await self._deliver_to_subscribers(message)
        
        self.metrics["messages_broadcast"] += len(subscribers)
    
    async def _cleanup_expired_messages(self):
        """清理过期消息后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                current_time = datetime.now()
                expired_count = 0
                
                # 清理各个队列中的过期消息
                for queue_key, queue in self.message_queues.items():
                    expired_messages = []
                    for message_data in queue:
                        message = await self._deserialize_message(message_data)
                        if message and message.is_expired:
                            expired_messages.append(message_data)
                    
                    # 移除过期消息
                    for expired_msg in expired_messages:
                        queue.remove(expired_msg)
                        expired_count += 1
                
                if expired_count > 0:
                    self.logger.info(f"清理了 {expired_count} 个过期消息")
                
            except Exception as e:
                self.logger.error(f"清理过期消息任务出错: {e}")
    
    async def _metrics_collector(self):
        """指标收集后台任务"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒收集一次
                
                # 计算吞吐量
                current_time = datetime.now()
                recent_messages = [
                    ts for ts in self.message_timestamps 
                    if (current_time - ts).total_seconds() <= 30
                ]
                
                self.metrics["throughput_per_second"] = len(recent_messages) / 30.0
                
                # 计算平均延迟
                if self.latency_history:
                    self.metrics["average_latency"] = sum(self.latency_history) / len(self.latency_history)
                
                # 更新队列大小统计
                for queue_key, queue in self.message_queues.items():
                    self.metrics["queue_sizes"][queue_key] = len(queue)
                
                self.metrics["queue_sizes"]["broadcast"] = len(self.broadcast_queue)
                
                # 发送指标事件
                await self._emit_event("metrics.updated", {
                    "timestamp": current_time.isoformat(),
                    "metrics": dict(self.metrics)
                })
                
            except Exception as e:
                self.logger.error(f"指标收集任务出错: {e}")
    
    async def _health_monitor(self):
        """健康监控后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                # 检查队列大小
                large_queues = []
                for queue_key, queue in self.message_queues.items():
                    if len(queue) > self.buffer_size * 0.8:
                        large_queues.append(queue_key)
                
                if large_queues:
                    self.logger.warning(f"检测到大队列: {large_queues}")
                    
                    await self._emit_event("health.warning", {
                        "type": "large_queues",
                        "queues": large_queues,
                        "buffer_utilization": max(len(q) for q in self.message_queues.values()) / self.buffer_size
                    })
                
                # 检查订阅数量
                total_subscriptions = len(self.subscriptions)
                if total_subscriptions > 1000:
                    self.logger.warning(f"订阅数量过多: {total_subscriptions}")
                    
                    await self._emit_event("health.warning", {
                        "type": "too_many_subscriptions",
                        "subscription_count": total_subscriptions
                    })
                
            except Exception as e:
                self.logger.error(f"健康监控任务出错: {e}")
    
    def get_subscription_info(self, subscriber_id: str) -> Dict[str, Any]:
        """获取订阅信息"""
        topics = self.subscriber_topics.get(subscriber_id, set())
        
        return {
            "subscriber_id": subscriber_id,
            "subscribed_topics": list(topics),
            "subscription_count": len(topics)
        }
    
    def get_topic_subscribers(self, topic: str) -> List[str]:
        """获取主题的订阅者"""
        return list(self.topic_subscribers.get(topic, set()))
    
    def get_bus_status(self) -> Dict[str, Any]:
        """获取总线状态"""
        return {
            "bus_id": self.bus_id,
            "total_subscriptions": len(self.subscriptions),
            "total_topics": len(self.topic_subscribers),
            "total_subscribers": len(self.subscriber_topics),
            "message_queues": len(self.message_queues),
            "metrics": dict(self.metrics),
            "queue_sizes": dict(self.metrics["queue_sizes"]),
            "buffer_size": self.buffer_size,
            "average_latency": self.metrics["average_latency"],
            "throughput_per_second": self.metrics["throughput_per_second"]
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
        """关闭消息总线"""
        try:
            self.logger.info("关闭消息总线...")
            
            # 取消所有订阅
            subscription_ids = list(self.subscriptions.keys())
            for subscription_id in subscription_ids:
                await self.unsubscribe(subscription_id)
            
            # 清空队列
            for queue in self.message_queues.values():
                queue.clear()
            self.broadcast_queue.clear()
            
            self.logger.info("消息总线已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭消息总线时出错: {e}")