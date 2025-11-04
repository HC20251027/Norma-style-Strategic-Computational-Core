"""
对话上下文管理器
负责在异步任务中保持和管理对话上下文，确保任务执行期间上下文的一致性
"""

import asyncio
import json
import pickle
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid


class ContextType(Enum):
    """上下文类型"""
    USER_PREFERENCE = "user_preference"
    CONVERSATION_HISTORY = "conversation_history"
    TASK_CONTEXT = "task_context"
    SYSTEM_STATE = "system_state"
    WORKFLOW_STATE = "workflow_state"
    USER_PROFILE = "user_profile"


class ContextManager:
    """对话上下文管理器"""
    
    def __init__(self, max_context_size: int = 1000, ttl_hours: int = 24):
        self.max_context_size = max_context_size
        self.ttl_hours = ttl_hours
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        
        # 上下文存储
        self.context_storage: Dict[str, Dict[str, Any]] = {}
        
        # 异步锁
        self.locks: Dict[str, asyncio.Lock] = {}
        
        # 上下文处理器
        self.context_processors: Dict[ContextType, Callable] = {}
        
        # 统计信息
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'context_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def create_session(self, session_id: str, initial_context: Optional[Dict[str, Any]] = None) -> bool:
        """创建新的会话上下文"""
        if session_id in self.sessions:
            return False
        
        session = {
            'session_id': session_id,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'context': initial_context or {},
            'context_history': [],
            'metadata': {
                'user_agent': None,
                'ip_address': None,
                'language': 'zh-CN',
                'timezone': 'Asia/Shanghai'
            },
            'workflow_state': {},
            'user_profile': {},
            'preferences': {}
        }
        
        self.sessions[session_id] = session
        self.context_cache[session_id] = session['context'].copy()
        self.locks[session_id] = asyncio.Lock()
        
        self.stats['total_sessions'] += 1
        self.stats['active_sessions'] += 1
        
        self.logger.info(f"Created session context {session_id}")
        return True
    
    async def get_context(self, session_id: str, context_type: Optional[ContextType] = None) -> Dict[str, Any]:
        """获取会话上下文"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session['last_accessed'] = datetime.now()
        
        if context_type:
            return session['context'].get(context_type.value, {})
        else:
            return session['context']
    
    async def update_context(self, 
                           session_id: str, 
                           updates: Dict[str, Any],
                           context_type: Optional[ContextType] = None) -> bool:
        """更新会话上下文"""
        if session_id not in self.sessions:
            return False
        
        async with self.locks[session_id]:
            session = self.sessions[session_id]
            
            # 保存历史记录
            if session['context']:
                session['context_history'].append({
                    'timestamp': datetime.now(),
                    'context': session['context'].copy()
                })
            
            # 限制历史记录大小
            if len(session['context_history']) > self.max_context_size:
                session['context_history'] = session['context_history'][-self.max_context_size//2:]
            
            # 更新上下文
            if context_type:
                session['context'][context_type.value] = updates
            else:
                session['context'].update(updates)
            
            # 更新缓存
            self.context_cache[session_id] = session['context'].copy()
            
            # 处理上下文更新
            await self._process_context_update(session_id, updates, context_type)
            
            self.stats['context_updates'] += 1
            self.logger.debug(f"Updated context for session {session_id}")
            return True
    
    async def _process_context_update(self, 
                                    session_id: str, 
                                    updates: Dict[str, Any],
                                    context_type: Optional[ContextType]):
        """处理上下文更新"""
        if context_type and context_type in self.context_processors:
            processor = self.context_processors[context_type]
            await processor(session_id, updates)
        else:
            # 默认处理器
            await self._default_context_processor(session_id, updates)
    
    async def _default_context_processor(self, session_id: str, updates: Dict[str, Any]):
        """默认上下文处理器"""
        session = self.sessions[session_id]
        
        # 更新用户偏好
        if 'preferences' in updates:
            session['preferences'].update(updates['preferences'])
        
        # 更新用户资料
        if 'user_profile' in updates:
            session['user_profile'].update(updates['user_profile'])
        
        # 更新工作流状态
        if 'workflow_state' in updates:
            session['workflow_state'].update(updates['workflow_state'])
    
    async def add_conversation_turn(self, 
                                  session_id: str, 
                                  user_message: str, 
                                  assistant_response: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加对话轮次"""
        if session_id not in self.sessions:
            return False
        
        async with self.locks[session_id]:
            session = self.sessions[session_id]
            
            turn = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now(),
                'user_message': user_message,
                'assistant_response': assistant_response,
                'metadata': metadata or {}
            }
            
            # 初始化对话历史
            if 'conversation_history' not in session['context']:
                session['context']['conversation_history'] = []
            
            # 添加轮次
            session['context']['conversation_history'].append(turn)
            
            # 限制历史长度
            if len(session['context']['conversation_history']) > self.max_context_size:
                session['context']['conversation_history'] = session['context']['conversation_history'][-self.max_context_size//2:]
            
            # 更新缓存
            self.context_cache[session_id] = session['context'].copy()
            
            self.logger.debug(f"Added conversation turn for session {session_id}")
            return True
    
    async def get_conversation_history(self, 
                                     session_id: str, 
                                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取对话历史"""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        history = session['context'].get('conversation_history', [])
        
        if limit:
            return history[-limit:]
        
        return history
    
    async def update_task_context(self, 
                                session_id: str, 
                                task_id: str, 
                                task_data: Dict[str, Any]) -> bool:
        """更新任务上下文"""
        if session_id not in self.sessions:
            return False
        
        async with self.locks[session_id]:
            session = self.sessions[session_id]
            
            # 初始化任务上下文
            if 'task_context' not in session['context']:
                session['context']['task_context'] = {}
            
            # 更新任务数据
            session['context']['task_context'][task_id] = {
                **task_data,
                'last_updated': datetime.now()
            }
            
            # 清理过期的任务上下文
            await self._cleanup_expired_task_context(session_id)
            
            # 更新缓存
            self.context_cache[session_id] = session['context'].copy()
            
            return True
    
    async def get_task_context(self, session_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务上下文"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        task_context = session['context'].get('task_context', {})
        return task_context.get(task_id)
    
    async def _cleanup_expired_task_context(self, session_id: str):
        """清理过期的任务上下文"""
        session = self.sessions[session_id]
        task_context = session['context'].get('task_context', {})
        
        cutoff_time = datetime.now() - timedelta(hours=self.ttl_hours)
        
        expired_tasks = []
        for task_id, task_data in task_context.items():
            if task_data.get('last_updated', datetime.now()) < cutoff_time:
                expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            del task_context[task_id]
    
    async def save_context_snapshot(self, session_id: str, snapshot_name: str) -> bool:
        """保存上下文快照"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        snapshot = {
            'name': snapshot_name,
            'timestamp': datetime.now(),
            'context': session['context'].copy(),
            'workflow_state': session['workflow_state'].copy(),
            'user_profile': session['user_profile'].copy(),
            'preferences': session['preferences'].copy()
        }
        
        # 初始化快照存储
        if 'snapshots' not in session:
            session['snapshots'] = []
        
        session['snapshots'].append(snapshot)
        
        # 限制快照数量
        if len(session['snapshots']) > 10:
            session['snapshots'] = session['snapshots'][-10:]
        
        self.logger.info(f"Saved context snapshot {snapshot_name} for session {session_id}")
        return True
    
    async def restore_context_snapshot(self, session_id: str, snapshot_name: str) -> bool:
        """恢复上下文快照"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        snapshots = session.get('snapshots', [])
        
        # 查找快照
        snapshot = None
        for s in snapshots:
            if s['name'] == snapshot_name:
                snapshot = s
                break
        
        if not snapshot:
            return False
        
        # 恢复上下文
        async with self.locks[session_id]:
            session['context'] = snapshot['context'].copy()
            session['workflow_state'] = snapshot['workflow_state'].copy()
            session['user_profile'] = snapshot['user_profile'].copy()
            session['preferences'] = snapshot['preferences'].copy()
            
            # 更新缓存
            self.context_cache[session_id] = session['context'].copy()
        
        self.logger.info(f"Restored context snapshot {snapshot_name} for session {session_id}")
        return True
    
    async def merge_contexts(self, 
                           target_session_id: str, 
                           source_session_id: str,
                           merge_strategy: str = 'overwrite') -> bool:
        """合并会话上下文"""
        if target_session_id not in self.sessions or source_session_id not in self.sessions:
            return False
        
        target_session = self.sessions[target_session_id]
        source_session = self.sessions[source_session_id]
        
        async with self.locks[target_session_id]:
            if merge_strategy == 'overwrite':
                target_session['context'].update(source_session['context'])
            elif merge_strategy == 'merge':
                # 深度合并
                await self._deep_merge_dict(target_session['context'], source_session['context'])
            elif merge_strategy == 'append':
                # 追加模式（适用于列表等）
                await self._append_merge(target_session['context'], source_session['context'])
            
            # 更新缓存
            self.context_cache[target_session_id] = target_session['context'].copy()
        
        self.logger.info(f"Merged context from {source_session_id} to {target_session_id}")
        return True
    
    async def _deep_merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """深度合并字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                await self._deep_merge_dict(target[key], value)
            else:
                target[key] = value
    
    async def _append_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """追加模式合并"""
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], list) and isinstance(value, list):
                    target[key].extend(value)
                elif isinstance(target[key], str) and isinstance(value, str):
                    target[key] += value
                else:
                    target[key] = value
            else:
                target[key] = value
    
    async def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """获取上下文摘要"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        context = session['context']
        
        summary = {
            'session_id': session_id,
            'created_at': session['created_at'].isoformat(),
            'last_accessed': session['last_accessed'].isoformat(),
            'context_types': list(context.keys()),
            'conversation_turns': len(context.get('conversation_history', [])),
            'active_tasks': len(context.get('task_context', {})),
            'user_profile_complete': bool(session['user_profile']),
            'preferences_count': len(session['preferences']),
            'snapshot_count': len(session.get('snapshots', []))
        }
        
        return summary
    
    async def cleanup_session(self, session_id: str) -> bool:
        """清理会话"""
        if session_id not in self.sessions:
            return False
        
        # 保存最终快照
        await self.save_context_snapshot(session_id, f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 清理资源
        del self.sessions[session_id]
        if session_id in self.context_cache:
            del self.context_cache[session_id]
        if session_id in self.locks:
            del self.locks[session_id]
        
        self.stats['active_sessions'] -= 1
        self.logger.info(f"Cleaned up session {session_id}")
        return True
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        cutoff_time = datetime.now() - timedelta(hours=self.ttl_hours)
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session['last_accessed'] < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.cleanup_session(session_id)
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def register_context_processor(self, context_type: ContextType, processor: Callable):
        """注册上下文处理器"""
        self.context_processors[context_type] = processor
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'cache_size': len(self.context_cache),
            'active_locks': len(self.locks),
            'session_count': len(self.sessions)
        }