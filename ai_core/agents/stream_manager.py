"""
流式生成模块
支持实时流式响应和流管理
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..utils import measure_time

logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    """流事件类型"""
    START = "start"
    TOKEN = "token"
    COMPLETION = "completion"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    META = "meta"

@dataclass
class StreamEvent:
    """流事件"""
    type: StreamEventType
    data: Any
    timestamp: float
    request_id: str
    sequence: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'request_id': self.request_id,
            'sequence': self.sequence
        }

@dataclass
class StreamConfig:
    """流配置"""
    chunk_size: int = 1024
    buffer_size: int = 4096
    heartbeat_interval: float = 30.0
    timeout: float = 300.0
    max_connections: int = 100
    enable_compression: bool = True
    compression_threshold: int = 1024
    retry_attempts: int = 3
    retry_delay: float = 1.0

class StreamConnection:
    """流连接"""
    
    def __init__(self, connection_id: str, request_id: str, config: StreamConfig):
        self.connection_id = connection_id
        self.request_id = request_id
        self.config = config
        self.created_at = time.time()
        self.last_activity = time.time()
        self.is_active = True
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.buffer_size)
        self.callbacks: List[Callable] = []
        self.metadata: Dict[str, Any] = {}
        
        # 统计信息
        self.events_sent = 0
        self.bytes_sent = 0
        self.errors_count = 0
    
    async def send_event(self, event: StreamEvent):
        """发送事件"""
        if not self.is_active:
            return False
        
        try:
            # 添加到队列
            await asyncio.wait_for(
                self.event_queue.put(event),
                timeout=1.0
            )
            
            self.last_activity = time.time()
            self.events_sent += 1
            
            # 通知回调
            for callback in self.callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.warning(f"流回调失败: {e}")
            
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"流队列满，丢弃事件: {event.type.value}")
            return False
        except Exception as e:
            logger.error(f"发送流事件失败: {e}")
            self.errors_count += 1
            return False
    
    async def receive_event(self) -> Optional[StreamEvent]:
        """接收事件"""
        try:
            event = await asyncio.wait_for(
                self.event_queue.get(),
                timeout=1.0
            )
            return event
        except asyncio.TimeoutError:
            return None
    
    def add_callback(self, callback: Callable):
        """添加回调"""
        self.callbacks.append(callback)
    
    def close(self):
        """关闭连接"""
        self.is_active = False
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return (time.time() - self.last_activity) > self.config.timeout
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'connection_id': self.connection_id,
            'request_id': self.request_id,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'is_active': self.is_active,
            'events_sent': self.events_sent,
            'bytes_sent': self.bytes_sent,
            'errors_count': self.errors_count,
            'queue_size': self.event_queue.qsize(),
            'uptime': time.time() - self.created_at
        }

class StreamManager:
    """流管理器"""
    
    def __init__(self, config: StreamConfig = None):
        self.config = config or StreamConfig()
        self.connections: Dict[str, StreamConnection] = {}
        self.active_streams: Dict[str, asyncio.Task] = {}
        self.sequence_counters: Dict[str, int] = {}
        
        # 统计信息
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_events': 0,
            'total_bytes': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # 清理任务
        self.cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        logger.info("流管理器初始化完成")
    
    def _start_cleanup_task(self):
        """启动清理任务"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                await self._cleanup_expired_connections()
                await asyncio.sleep(60)  # 每分钟清理一次
            except Exception as e:
                logger.error(f"清理循环错误: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_expired_connections(self):
        """清理过期连接"""
        expired_ids = []
        
        for conn_id, connection in self.connections.items():
            if connection.is_expired() or not connection.is_active:
                expired_ids.append(conn_id)
        
        for conn_id in expired_ids:
            await self.close_connection(conn_id)
        
        if expired_ids:
            logger.info(f"清理了{len(expired_ids)}个过期连接")
    
    async def create_connection(self, request_id: str, metadata: Dict[str, Any] = None) -> str:
        """创建流连接"""
        if len(self.connections) >= self.config.max_connections:
            raise Exception("达到最大连接数限制")
        
        connection_id = f"stream_{int(time.time() * 1000)}_{len(self.connections)}"
        
        connection = StreamConnection(connection_id, request_id, self.config)
        if metadata:
            connection.metadata.update(metadata)
        
        self.connections[connection_id] = connection
        self.sequence_counters[request_id] = 0
        
        self.stats['total_connections'] += 1
        self.stats['active_connections'] += 1
        
        logger.info(f"创建流连接: {connection_id} (请求: {request_id})")
        return connection_id
    
    async def close_connection(self, connection_id: str):
        """关闭流连接"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.close()
            
            # 取消相关任务
            if connection_id in self.active_streams:
                task = self.active_streams[connection_id]
                if not task.done():
                    task.cancel()
                del self.active_streams[connection_id]
            
            # 更新统计
            self.stats['active_connections'] = max(0, self.stats['active_connections'] - 1)
            
            # 移除连接
            del self.connections[connection_id]
            
            logger.info(f"关闭流连接: {connection_id}")
    
    async def send_event(
        self,
        connection_id: str,
        event_type: StreamEventType,
        data: Any,
        request_id: str = None
    ) -> bool:
        """发送事件"""
        if connection_id not in self.connections:
            logger.warning(f"连接不存在: {connection_id}")
            return False
        
        connection = self.connections[connection_id]
        
        # 生成序列号
        if request_id is None:
            request_id = connection.request_id
        
        sequence = self.sequence_counters.get(request_id, 0)
        self.sequence_counters[request_id] = sequence + 1
        
        # 创建事件
        event = StreamEvent(
            type=event_type,
            data=data,
            timestamp=time.time(),
            request_id=request_id,
            sequence=sequence
        )
        
        # 发送事件
        success = await connection.send_event(event)
        
        if success:
            self.stats['total_events'] += 1
            self.stats['total_bytes'] += len(str(data))
        
        return success
    
    async def stream_response(
        self,
        connection_id: str,
        data_generator: AsyncGenerator[Any, None],
        request_id: str = None
    ):
        """流式响应"""
        if connection_id not in self.connections:
            raise Exception(f"连接不存在: {connection_id}")
        
        connection = self.connections[connection_id]
        if request_id is None:
            request_id = connection.request_id
        
        # 发送开始事件
        await self.send_event(connection_id, StreamEventType.START, {
            'message': 'Stream started',
            'request_id': request_id
        }, request_id)
        
        try:
            # 流式发送数据
            async for chunk in data_generator:
                if not connection.is_active:
                    break
                
                await self.send_event(
                    connection_id,
                    StreamEventType.TOKEN,
                    {'content': chunk},
                    request_id
                )
                
                # 控制发送速率
                await asyncio.sleep(0.01)
            
            # 发送完成事件
            await self.send_event(connection_id, StreamEventType.COMPLETION, {
                'message': 'Stream completed'
            }, request_id)
            
        except Exception as e:
            logger.error(f"流式响应错误: {e}")
            await self.send_event(connection_id, StreamEventType.ERROR, {
                'error': str(e)
            }, request_id)
    
    async def stream_text(
        self,
        connection_id: str,
        text: str,
        request_id: str = None,
        chunk_size: int = None
    ):
        """流式发送文本"""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        # 文本分块
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        async def text_generator():
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)  # 控制发送速率
        
        await self.stream_response(connection_id, text_generator(), request_id)
    
    async def stream_json(
        self,
        connection_id: str,
        json_data: Dict[str, Any],
        request_id: str = None
    ):
        """流式发送JSON数据"""
        async def json_generator():
            yield json.dumps(json_data, ensure_ascii=False)
        
        await self.stream_response(connection_id, json_generator(), request_id)
    
    async def send_heartbeat(self, connection_id: str):
        """发送心跳"""
        await self.send_event(connection_id, StreamEventType.HEARTBEAT, {
            'timestamp': time.time()
        })
    
    async def get_connection_events(self, connection_id: str) -> AsyncGenerator[StreamEvent, None]:
        """获取连接事件流"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        while connection.is_active:
            event = await connection.receive_event()
            if event:
                yield event
            else:
                await asyncio.sleep(0.1)  # 避免忙等待
    
    def get_connection(self, connection_id: str) -> Optional[StreamConnection]:
        """获取连接"""
        return self.connections.get(connection_id)
    
    def get_all_connections(self) -> Dict[str, StreamConnection]:
        """获取所有连接"""
        return self.connections.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - self.stats['start_time']
        
        # 连接详情
        connection_stats = {}
        for conn_id, conn in self.connections.items():
            connection_stats[conn_id] = conn.get_stats()
        
        return {
            'uptime': uptime,
            'global_stats': self.stats.copy(),
            'config': asdict(self.config),
            'active_connections': len(self.connections),
            'connection_details': connection_stats,
            'memory_usage': {
                'connections_count': len(self.connections),
                'total_events_queued': sum(conn.event_queue.qsize() for conn in self.connections.values())
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        healthy_connections = 0
        total_connections = len(self.connections)
        
        for conn in self.connections.values():
            if conn.is_active and not conn.is_expired():
                healthy_connections += 1
        
        return {
            'healthy': True,
            'total_connections': total_connections,
            'healthy_connections': healthy_connections,
            'health_ratio': healthy_connections / total_connections if total_connections > 0 else 1.0,
            'timestamp': time.time()
        }
    
    async def shutdown(self):
        """关闭流管理器"""
        logger.info("开始关闭流管理器")
        
        # 关闭所有连接
        for connection_id in list(self.connections.keys()):
            await self.close_connection(connection_id)
        
        # 取消清理任务
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("流管理器已关闭")
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"流配置已更新: {kwargs}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return asdict(self.config)