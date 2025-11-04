"""
流畅交互管理器
核心管理器，协调所有交互组件，提供统一的流畅交互体验
"""

import asyncio
import uuid
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging

from .task_manager import MultiTaskManager
from .context_manager import ContextManager
from .progress_tracker import ProgressTracker
from .interruption_handler import InterruptionHandler
from .interaction_flow import InteractionFlowOptimizer


class InteractionState(Enum):
    """交互状态枚举"""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_INPUT = "waiting_input"
    INTERRUPTED = "interrupted"
    PAUSED = "paused"
    COMPLETED = "completed"


class SmoothInteractionManager:
    """流畅交互管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.session_id = str(uuid.uuid4())
        self.state = InteractionState.IDLE
        
        # 初始化各个组件
        self.task_manager = MultiTaskManager()
        self.context_manager = ContextManager()
        self.progress_tracker = ProgressTracker()
        self.interruption_handler = InterruptionHandler()
        self.flow_optimizer = InteractionFlowOptimizer()
        
        # 回调函数
        self.on_state_change: Optional[Callable] = None
        self.on_progress_update: Optional[Callable] = None
        self.on_message: Optional[Callable] = None
        
        # 活跃会话
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # 性能统计
        self.performance_stats = {
            'total_interactions': 0,
            'avg_response_time': 0,
            'task_completion_rate': 0,
            'user_satisfaction': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def start_interaction(self, user_id: str, initial_context: Optional[Dict[str, Any]] = None) -> str:
        """开始交互会话"""
        session_id = str(uuid.uuid4())
        
        # 初始化会话
        session = {
            'session_id': session_id,
            'user_id': user_id,
            'start_time': datetime.now(),
            'state': InteractionState.IDLE,
            'context': initial_context or {},
            'active_tasks': [],
            'message_history': [],
            'preferences': {}
        }
        
        self.active_sessions[session_id] = session
        await self.context_manager.create_session(session_id, initial_context)
        
        self.state = InteractionState.PROCESSING
        await self._notify_state_change()
        
        self.logger.info(f"Started interaction session {session_id} for user {user_id}")
        return session_id
    
    async def process_message(self, session_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理用户消息"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        message_data = {
            'id': str(uuid.uuid4()),
            'content': message,
            'timestamp': datetime.now(),
            'metadata': metadata or {},
            'type': 'user'
        }
        
        # 添加到历史记录
        session['message_history'].append(message_data)
        
        # 更新上下文
        await self.context_manager.update_context(session_id, {
            'last_message': message,
            'message_count': len(session['message_history']),
            'last_activity': datetime.now()
        })
        
        # 分析消息意图
        intent = await self._analyze_intent(message)
        
        # 优化交互流程
        optimized_flow = await self.flow_optimizer.optimize_flow(session_id, intent, session['context'])
        
        # 处理任务
        response = await self._handle_message(session_id, message, intent, optimized_flow)
        
        # 记录响应时间
        response_time = (datetime.now() - message_data['timestamp']).total_seconds()
        await self._update_performance_stats(response_time)
        
        return response
    
    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """分析用户消息意图"""
        # 简化的意图分析逻辑
        intent = {
            'type': 'general',
            'confidence': 0.8,
            'entities': [],
            'action': 'respond',
            'priority': 'normal'
        }
        
        # 检查是否包含任务相关关键词
        task_keywords = ['做', '执行', '处理', '分析', '创建', '生成', '计算']
        if any(keyword in message for keyword in task_keywords):
            intent['type'] = 'task'
            intent['action'] = 'execute'
            intent['priority'] = 'high'
        
        # 检查是否包含中断相关关键词
        interrupt_keywords = ['停止', '暂停', '取消', '中断']
        if any(keyword in message for keyword in interrupt_keywords):
            intent['type'] = 'interrupt'
            intent['action'] = 'interrupt'
            intent['priority'] = 'urgent'
        
        return intent
    
    async def _handle_message(self, session_id: str, message: str, intent: Dict[str, Any], flow: Dict[str, Any]) -> Dict[str, Any]:
        """处理消息的核心逻辑"""
        session = self.active_sessions[session_id]
        
        if intent['action'] == 'interrupt':
            # 处理中断请求
            await self.interruption_handler.handle_interrupt(session_id, message)
            session['state'] = InteractionState.INTERRUPTED
            return {
                'response': '已中断当前任务，请告诉我您想要执行什么新任务？',
                'state': 'interrupted',
                'active_tasks': await self.task_manager.get_active_tasks(session_id)
            }
        
        elif intent['action'] == 'execute':
            # 创建新任务
            task_id = await self.task_manager.create_task(
                session_id=session_id,
                task_type='user_request',
                description=message,
                priority=intent['priority']
            )
            
            # 启动任务
            await self.task_manager.start_task(task_id)
            
            # 跟踪进度
            await self.progress_tracker.start_tracking(task_id, session_id)
            
            session['active_tasks'].append(task_id)
            session['state'] = InteractionState.PROCESSING
            
            return {
                'response': f'正在处理您的请求，任务ID: {task_id[:8]}...',
                'state': 'processing',
                'task_id': task_id,
                'estimated_time': await self.task_manager.get_estimated_time(task_id)
            }
        
        else:
            # 常规响应
            return {
                'response': await self._generate_response(message, session['context']),
                'state': 'idle',
                'suggestions': await self.flow_optimizer.get_suggestions(session_id)
            }
    
    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """生成响应消息"""
        # 简化的响应生成逻辑
        responses = [
            '我理解您的需求，让我来帮您处理。',
            '好的，请稍等，我正在为您分析。',
            '明白了，我来为您处理这个问题。',
            '好的，请告诉我您希望我如何帮助您。'
        ]
        
        import random
        return random.choice(responses)
    
    async def get_task_progress(self, session_id: str, task_id: str) -> Dict[str, Any]:
        """获取任务进度"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        progress = await self.progress_tracker.get_progress(task_id)
        return {
            'task_id': task_id,
            'progress': progress,
            'status': await self.task_manager.get_task_status(task_id),
            'can_interrupt': await self.interruption_handler.can_interrupt(task_id)
        }
    
    async def interrupt_task(self, session_id: str, task_id: str, reason: str = '') -> bool:
        """中断任务"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        success = await self.interruption_handler.interrupt_task(task_id, reason)
        if success:
            session = self.active_sessions[session_id]
            if task_id in session['active_tasks']:
                session['active_tasks'].remove(task_id)
            
            session['state'] = InteractionState.INTERRUPTED
            await self._notify_state_change()
        
        return success
    
    async def resume_task(self, session_id: str, task_id: str) -> bool:
        """恢复任务"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        success = await self.interruption_handler.resume_task(task_id)
        if success:
            session = self.active_sessions[session_id]
            if task_id not in session['active_tasks']:
                session['active_tasks'].append(task_id)
            
            session['state'] = InteractionState.PROCESSING
            await self._notify_state_change()
        
        return success
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        active_tasks = []
        
        for task_id in session['active_tasks']:
            task_status = await self.task_manager.get_task_status(task_id)
            progress = await self.progress_tracker.get_progress(task_id)
            active_tasks.append({
                'task_id': task_id,
                'status': task_status,
                'progress': progress
            })
        
        return {
            'session_id': session_id,
            'user_id': session['user_id'],
            'state': session['state'].value,
            'start_time': session['start_time'],
            'active_tasks': active_tasks,
            'message_count': len(session['message_history']),
            'context': await self.context_manager.get_context(session_id)
        }
    
    async def end_interaction(self, session_id: str) -> Dict[str, Any]:
        """结束交互会话"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # 清理任务
        for task_id in session['active_tasks']:
            await self.task_manager.cancel_task(task_id)
            await self.progress_tracker.stop_tracking(task_id)
        
        # 生成会话报告
        report = {
            'session_id': session_id,
            'duration': (datetime.now() - session['start_time']).total_seconds(),
            'message_count': len(session['message_history']),
            'tasks_completed': len(session['active_tasks']),
            'performance_stats': self.performance_stats
        }
        
        # 清理会话
        await self.context_manager.cleanup_session(session_id)
        del self.active_sessions[session_id]
        
        self.logger.info(f"Ended interaction session {session_id}")
        return report
    
    async def _notify_state_change(self):
        """通知状态变化"""
        if self.on_state_change:
            await self.on_state_change(self.state)
    
    async def _update_performance_stats(self, response_time: float):
        """更新性能统计"""
        self.performance_stats['total_interactions'] += 1
        
        # 计算平均响应时间
        total_time = self.performance_stats['avg_response_time'] * (self.performance_stats['total_interactions'] - 1)
        self.performance_stats['avg_response_time'] = (total_time + response_time) / self.performance_stats['total_interactions']
    
    def register_callbacks(self, 
                         on_state_change: Optional[Callable] = None,
                         on_progress_update: Optional[Callable] = None,
                         on_message: Optional[Callable] = None):
        """注册回调函数"""
        self.on_state_change = on_state_change
        self.on_progress_update = on_progress_update
        self.on_message = on_message
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'active_sessions': len(self.active_sessions),
            'total_tasks': await self.task_manager.get_total_tasks(),
            'performance_stats': self.performance_stats,
            'system_health': 'healthy',
            'uptime': '运行正常'
        }