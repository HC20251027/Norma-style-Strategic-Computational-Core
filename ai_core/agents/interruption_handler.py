"""
中断处理器
实现任务的中断、恢复和状态管理功能
"""

import asyncio
import uuid
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import signal


class InterruptType(Enum):
    """中断类型"""
    USER_REQUEST = "user_request"
    SYSTEM_ERROR = "system_error"
    RESOURCE_LIMIT = "resource_limit"
    TIMEOUT = "timeout"
    PRIORITY_CHANGE = "priority_change"
    DEPENDENCY_FAILURE = "dependency_failure"


class InterruptStatus(Enum):
    """中断状态"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class InterruptRequest:
    """中断请求"""
    
    def __init__(self, 
                 request_id: str,
                 task_id: str,
                 interrupt_type: InterruptType,
                 reason: str,
                 priority: int = 1,
                 timeout: Optional[int] = None):
        self.request_id = request_id
        self.task_id = task_id
        self.interrupt_type = interrupt_type
        self.reason = reason
        self.priority = priority
        self.timeout = timeout
        self.created_at = datetime.now()
        self.status = InterruptStatus.PENDING
        self.processed_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None


class InterruptionHandler:
    """中断处理器"""
    
    def __init__(self, 
                 max_concurrent_interrupts: int = 5,
                 default_timeout: int = 30):
        self.max_concurrent_interrupts = max_concurrent_interrupts
        self.default_timeout = default_timeout
        
        # 中断管理
        self.interrupt_queue: List[InterruptRequest] = []
        self.active_interrupts: Dict[str, InterruptRequest] = {}
        self.interrupt_history: Dict[str, List[InterruptRequest]] = {}
        
        # 任务中断状态
        self.task_interrupt_states: Dict[str, Dict[str, Any]] = {}
        
        # 中断策略
        self.interrupt_strategies = {
            InterruptType.USER_REQUEST: self._handle_user_interrupt,
            InterruptType.SYSTEM_ERROR: self._handle_system_error,
            InterruptType.RESOURCE_LIMIT: self._handle_resource_limit,
            InterruptType.TIMEOUT: self._handle_timeout,
            InterruptType.PRIORITY_CHANGE: self._handle_priority_change,
            InterruptType.DEPENDENCY_FAILURE: self._handle_dependency_failure
        }
        
        # 回调函数
        self.on_interrupt_request: Optional[Callable] = None
        self.on_interrupt_complete: Optional[Callable] = None
        self.on_task_interrupted: Optional[Callable] = None
        self.on_task_resumed: Optional[Callable] = None
        
        # 统计信息
        self.stats = {
            'total_interrupt_requests': 0,
            'successful_interrupts': 0,
            'rejected_interrupts': 0,
            'average_interrupt_time': 0,
            'tasks_interrupted': 0,
            'tasks_resumed': 0
        }
        
        # 中断处理循环
        self.processing_loop_running = False
        self.processing_loop_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def request_interrupt(self, 
                              task_id: str,
                              interrupt_type: InterruptType,
                              reason: str,
                              priority: int = 1,
                              timeout: Optional[int] = None) -> str:
        """请求中断任务"""
        request_id = str(uuid.uuid4())
        
        interrupt_request = InterruptRequest(
            request_id=request_id,
            task_id=task_id,
            interrupt_type=interrupt_type,
            reason=reason,
            priority=priority,
            timeout=timeout or self.default_timeout
        )
        
        # 添加到队列
        self.interrupt_queue.append(interrupt_request)
        
        # 按优先级排序
        self.interrupt_queue.sort(key=lambda x: x.priority, reverse=True)
        
        # 限制队列大小
        if len(self.interrupt_queue) > self.max_concurrent_interrupts * 2:
            self.interrupt_queue = self.interrupt_queue[-self.max_concurrent_interrupts:]
        
        self.stats['total_interrupt_requests'] += 1
        
        # 启动处理循环
        if not self.processing_loop_running:
            await self._start_processing_loop()
        
        self.logger.info(f"Interrupt request {request_id} created for task {task_id}")
        return request_id
    
    async def handle_interrupt(self, session_id: str, message: str) -> Dict[str, Any]:
        """处理用户中断请求"""
        # 解析中断意图
        interrupt_intent = await self._parse_interrupt_intent(message)
        
        if not interrupt_intent:
            return {
                'success': False,
                'message': '无法解析中断请求'
            }
        
        # 创建中断请求
        request_id = await self.request_interrupt(
            task_id=interrupt_intent['task_id'],
            interrupt_type=InterruptType.USER_REQUEST,
            reason=interrupt_intent['reason'],
            priority=interrupt_intent.get('priority', 1)
        )
        
        return {
            'success': True,
            'request_id': request_id,
            'message': f'已请求中断任务 {interrupt_intent["task_id"][:8]}...'
        }
    
    async def can_interrupt(self, task_id: str) -> bool:
        """检查任务是否可以被中断"""
        # 检查任务状态
        if task_id not in self.task_interrupt_states:
            return True
        
        state = self.task_interrupt_states[task_id]
        
        # 如果任务已经被中断且未恢复，则不能再次中断
        if state.get('interrupted', False) and not state.get('resumed', False):
            return False
        
        # 检查是否有活跃的中断请求
        for interrupt in self.active_interrupts.values():
            if interrupt.task_id == task_id and interrupt.status == InterruptStatus.ACTIVE:
                return False
        
        return True
    
    async def interrupt_task(self, task_id: str, reason: str = '') -> bool:
        """中断任务"""
        if not await self.can_interrupt(task_id):
            return False
        
        # 创建中断请求
        request_id = await self.request_interrupt(
            task_id=task_id,
            interrupt_type=InterruptType.USER_REQUEST,
            reason=reason or '用户请求中断'
        )
        
        # 等待中断完成
        result = await self._wait_for_interrupt_completion(request_id, timeout=30)
        
        if result and result.get('success'):
            # 更新任务中断状态
            self.task_interrupt_states[task_id] = {
                'interrupted': True,
                'interrupted_at': datetime.now(),
                'interrupt_reason': reason,
                'resumed': False
            }
            
            self.stats['tasks_interrupted'] += 1
            
            if self.on_task_interrupted:
                await self.on_task_interrupted(task_id, reason)
        
        return result.get('success', False) if result else False
    
    async def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        if task_id not in self.task_interrupt_states:
            return False
        
        state = self.task_interrupt_states[task_id]
        
        if not state.get('interrupted', False):
            return False
        
        # 更新状态
        state['resumed'] = True
        state['resumed_at'] = datetime.now()
        
        self.stats['tasks_resumed'] += 1
        
        if self.on_task_resumed:
            await self.on_task_resumed(task_id)
        
        self.logger.info(f"Resumed task {task_id}")
        return True
    
    async def get_interrupt_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取中断请求状态"""
        # 检查活跃中断
        if request_id in self.active_interrupts:
            interrupt = self.active_interrupts[request_id]
            return {
                'request_id': request_id,
                'task_id': interrupt.task_id,
                'status': interrupt.status.value,
                'interrupt_type': interrupt.interrupt_type.value,
                'reason': interrupt.reason,
                'created_at': interrupt.created_at.isoformat(),
                'processed_at': interrupt.processed_at.isoformat() if interrupt.processed_at else None,
                'completed_at': interrupt.completed_at.isoformat() if interrupt.completed_at else None
            }
        
        # 检查历史记录
        for task_history in self.interrupt_history.values():
            for interrupt in task_history:
                if interrupt.request_id == request_id:
                    return {
                        'request_id': request_id,
                        'task_id': interrupt.task_id,
                        'status': interrupt.status.value,
                        'interrupt_type': interrupt.interrupt_type.value,
                        'reason': interrupt.reason,
                        'created_at': interrupt.created_at.isoformat(),
                        'processed_at': interrupt.processed_at.isoformat() if interrupt.processed_at else None,
                        'completed_at': interrupt.completed_at.isoformat() if interrupt.completed_at else None,
                        'result': interrupt.result,
                        'error': interrupt.error
                    }
        
        return None
    
    async def get_task_interrupt_history(self, task_id: str) -> List[Dict[str, Any]]:
        """获取任务的中断历史"""
        if task_id not in self.interrupt_history:
            return []
        
        history = []
        for interrupt in self.interrupt_history[task_id]:
            history.append({
                'request_id': interrupt.request_id,
                'interrupt_type': interrupt.interrupt_type.value,
                'reason': interrupt.reason,
                'status': interrupt.status.value,
                'created_at': interrupt.created_at.isoformat(),
                'completed_at': interrupt.completed_at.isoformat() if interrupt.completed_at else None
            })
        
        return history
    
    async def cancel_interrupt_request(self, request_id: str) -> bool:
        """取消中断请求"""
        # 从队列中移除
        for i, interrupt in enumerate(self.interrupt_queue):
            if interrupt.request_id == request_id:
                del self.interrupt_queue[i]
                return True
        
        # 如果正在处理，则标记为已取消
        if request_id in self.active_interrupts:
            interrupt = self.active_interrupts[request_id]
            interrupt.status = InterruptStatus.COMPLETED
            interrupt.completed_at = datetime.now()
            interrupt.result = {'cancelled': True}
            return True
        
        return False
    
    async def _parse_interrupt_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """解析中断意图"""
        # 简化的意图解析逻辑
        intent = {
            'task_id': None,
            'reason': message,
            'priority': 1
        }
        
        # 提取任务ID（假设在消息中）
        import re
        task_id_pattern = r'task[:\s]*([a-f0-9\-]{8,})'
        match = re.search(task_id_pattern, message, re.IGNORECASE)
        if match:
            intent['task_id'] = match.group(1)
        else:
            # 如果没有指定任务ID，尝试从上下文获取
            intent['task_id'] = 'default_task'
        
        # 确定优先级
        urgent_keywords = ['紧急', '立即', '马上', 'urgent', 'immediate']
        if any(keyword in message for keyword in urgent_keywords):
            intent['priority'] = 5
        
        return intent
    
    async def _start_processing_loop(self):
        """启动中断处理循环"""
        if self.processing_loop_running:
            return
        
        self.processing_loop_running = True
        self.processing_loop_task = asyncio.create_task(self._processing_loop())
    
    async def _stop_processing_loop(self):
        """停止中断处理循环"""
        self.processing_loop_running = False
        if self.processing_loop_task:
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                pass
    
    async def _processing_loop(self):
        """中断处理循环"""
        while self.processing_loop_running:
            try:
                # 处理队列中的中断请求
                while self.interrupt_queue and len(self.active_interrupts) < self.max_concurrent_interrupts:
                    interrupt = self.interrupt_queue.pop(0)
                    await self._process_interrupt(interrupt)
                
                # 检查超时
                current_time = datetime.now()
                timeout_requests = []
                
                for request_id, interrupt in self.active_interrupts.items():
                    if interrupt.timeout:
                        elapsed = (current_time - interrupt.created_at).total_seconds()
                        if elapsed > interrupt.timeout:
                            timeout_requests.append(request_id)
                
                # 处理超时请求
                for request_id in timeout_requests:
                    await self._handle_timeout_interrupt(request_id)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in interrupt processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_interrupt(self, interrupt: InterruptRequest):
        """处理单个中断请求"""
        interrupt.status = InterruptStatus.ACTIVE
        interrupt.processed_at = datetime.now()
        
        self.active_interrupts[interrupt.request_id] = interrupt
        
        try:
            # 根据中断类型选择处理策略
            if interrupt.interrupt_type in self.interrupt_strategies:
                strategy = self.interrupt_strategies[interrupt.interrupt_type]
                result = await strategy(interrupt)
                
                interrupt.status = InterruptStatus.COMPLETED
                interrupt.result = result
                interrupt.completed_at = datetime.now()
                
                self.stats['successful_interrupts'] += 1
                
            else:
                # 默认处理
                result = await self._default_interrupt_handler(interrupt)
                interrupt.result = result
                interrupt.status = InterruptStatus.COMPLETED
                interrupt.completed_at = datetime.now()
            
            # 移动到历史记录
            if interrupt.task_id not in self.interrupt_history:
                self.interrupt_history[interrupt.task_id] = []
            self.interrupt_history[interrupt.task_id].append(interrupt)
            
            # 限制历史记录大小
            if len(self.interrupt_history[interrupt.task_id]) > 20:
                self.interrupt_history[interrupt.task_id] = self.interrupt_history[interrupt.task_id][-10:]
            
            if self.on_interrupt_complete:
                await self.on_interrupt_complete(interrupt)
            
        except Exception as e:
            interrupt.status = InterruptStatus.FAILED
            interrupt.error = str(e)
            interrupt.completed_at = datetime.now()
            
            self.stats['rejected_interrupts'] += 1
            self.logger.error(f"Failed to process interrupt {interrupt.request_id}: {e}")
        
        finally:
            # 从活跃中断中移除
            if interrupt.request_id in self.active_interrupts:
                del self.active_interrupts[interrupt.request_id]
    
    async def _handle_user_interrupt(self, interrupt: InterruptRequest) -> Dict[str, Any]:
        """处理用户中断"""
        # 模拟中断处理时间
        await asyncio.sleep(0.1)
        
        return {
            'success': True,
            'action': 'task_interrupted',
            'message': f'任务 {interrupt.task_id} 已被中断'
        }
    
    async def _handle_system_error(self, interrupt: InterruptRequest) -> Dict[str, Any]:
        """处理系统错误中断"""
        return {
            'success': True,
            'action': 'task_failed',
            'message': f'任务 {interrupt.task_id} 因系统错误被中断'
        }
    
    async def _handle_resource_limit(self, interrupt: InterruptRequest) -> Dict[str, Any]:
        """处理资源限制中断"""
        return {
            'success': True,
            'action': 'task_paused',
            'message': f'任务 {interrupt.task_id} 因资源限制被暂停'
        }
    
    async def _handle_timeout(self, interrupt: InterruptRequest) -> Dict[str, Any]:
        """处理超时中断"""
        return {
            'success': True,
            'action': 'task_cancelled',
            'message': f'任务 {interrupt.task_id} 因超时被取消'
        }
    
    async def _handle_priority_change(self, interrupt: InterruptRequest) -> Dict[str, Any]:
        """处理优先级变更中断"""
        return {
            'success': True,
            'action': 'priority_updated',
            'message': f'任务 {interrupt.task_id} 优先级已更新'
        }
    
    async def _handle_dependency_failure(self, interrupt: InterruptRequest) -> Dict[str, Any]:
        """处理依赖失败中断"""
        return {
            'success': True,
            'action': 'task_failed',
            'message': f'任务 {interrupt.task_id} 因依赖失败被中断'
        }
    
    async def _default_interrupt_handler(self, interrupt: InterruptRequest) -> Dict[str, Any]:
        """默认中断处理器"""
        await asyncio.sleep(0.05)
        return {
            'success': True,
            'action': 'interrupt_processed',
            'message': f'中断请求 {interrupt.request_id} 已处理'
        }
    
    async def _handle_timeout_interrupt(self, request_id: str):
        """处理超时中断"""
        if request_id in self.active_interrupts:
            interrupt = self.active_interrupts[request_id]
            interrupt.status = InterruptStatus.EXPIRED
            interrupt.completed_at = datetime.now()
            interrupt.result = {'timeout': True}
            
            self.stats['rejected_interrupts'] += 1
            
            # 移动到历史记录
            if interrupt.task_id not in self.interrupt_history:
                self.interrupt_history[interrupt.task_id] = []
            self.interrupt_history[interrupt.task_id].append(interrupt)
            
            del self.active_interrupts[request_id]
    
    async def _wait_for_interrupt_completion(self, request_id: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """等待中断完成"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = await self.get_interrupt_status(request_id)
            if status and status['status'] in ['completed', 'failed', 'expired']:
                return status
            
            await asyncio.sleep(0.1)
        
        return None
    
    def register_callbacks(self,
                          on_interrupt_request: Optional[Callable] = None,
                          on_interrupt_complete: Optional[Callable] = None,
                          on_task_interrupted: Optional[Callable] = None,
                          on_task_resumed: Optional[Callable] = None):
        """注册回调函数"""
        self.on_interrupt_request = on_interrupt_request
        self.on_interrupt_complete = on_interrupt_complete
        self.on_task_interrupted = on_task_interrupted
        self.on_task_resumed = on_task_resumed
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'queued_interrupts': len(self.interrupt_queue),
            'active_interrupts': len(self.active_interrupts),
            'total_history_entries': sum(len(history) for history in self.interrupt_history.values())
        }