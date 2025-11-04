"""
流水线状态管理器

负责管理流水线的状态、进度跟踪和性能监控
"""

import asyncio
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import asdict
import logging

from .models import (
    PipelineState, 
    PipelineStatus, 
    ProcessingStage, 
    PipelineMetrics,
    ErrorType,
    PipelineRequest
)


class PipelineStateManager:
    """流水线状态管理器"""
    
    def __init__(self, max_history_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self._states: Dict[str, PipelineState] = {}
        self._metrics = PipelineMetrics()
        self._max_history_size = max_history_size
        self._state_history: List[Dict] = []
        self._lock = asyncio.Lock()
    
    async def create_state(self, request: PipelineRequest) -> PipelineState:
        """创建新的流水线状态"""
        async with self._lock:
            state = PipelineState(
                request_id=request.id,
                status=PipelineStatus.IDLE,
                current_stage=ProcessingStage.RECEIVED,
                start_time=datetime.now(),
                metadata=request.metadata
            )
            self._states[request.id] = state
            self._metrics.total_requests += 1
            return state
    
    async def update_stage(self, request_id: str, stage: ProcessingStage, 
                          progress: float = None, metadata: Dict = None) -> bool:
        """更新处理阶段"""
        async with self._lock:
            if request_id not in self._states:
                self.logger.warning(f"Request {request_id} not found in state manager")
                return False
            
            state = self._states[request_id]
            old_stage = state.current_stage
            state.current_stage = stage
            
            if progress is not None:
                state.progress = progress
            
            if metadata:
                state.metadata.update(metadata)
            
            # 记录阶段处理时间
            current_time = datetime.now()
            if old_stage != stage:
                if old_stage.value not in state.processing_times:
                    state.processing_times[old_stage.value] = 0
                # 这里可以添加更精确的时间跟踪
            
            # 保存历史记录
            await self._save_state_history(request_id, old_stage, stage)
            
            self.logger.info(f"Updated stage for {request_id}: {old_stage.value} -> {stage.value}")
            return True
    
    async def set_status(self, request_id: str, status: PipelineStatus, 
                        error_message: str = None, error_type: ErrorType = None) -> bool:
        """设置流水线状态"""
        async with self._lock:
            if request_id not in self._states:
                return False
            
            state = self._states[request_id]
            old_status = state.status
            state.status = status
            
            if error_message:
                state.error_message = error_message
            
            if error_type:
                state.error_type = error_type
            
            # 如果是完成或失败状态，设置结束时间
            if status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED, PipelineStatus.CANCELLED]:
                state.end_time = datetime.now()
                await self._update_metrics(state, old_status)
            
            return True
    
    async def set_speech_result(self, request_id: str, text: str, confidence: float = None) -> bool:
        """设置语音识别结果"""
        async with self._lock:
            if request_id not in self._states:
                return False
            
            state = self._states[request_id]
            state.speech_text = text
            if confidence:
                state.metadata['speech_confidence'] = confidence
            
            return True
    
    async def add_tool_call(self, request_id: str, tool_call) -> bool:
        """添加工具调用"""
        async with self._lock:
            if request_id not in self._states:
                return False
            
            state = self._states[request_id]
            state.tool_calls.append(tool_call)
            return True
    
    async def add_execution_result(self, request_id: str, result) -> bool:
        """添加工具执行结果"""
        async with self._lock:
            if request_id not in self._states:
                return False
            
            state = self._states[request_id]
            state.execution_results.append(result)
            
            # 更新工具调用状态
            for call in state.tool_calls:
                if call.call_id == result.call_id:
                    call.metadata['execution_result'] = result
                    break
            
            return True
    
    async def set_final_result(self, request_id: str, text: str = None, 
                              audio: bytes = None) -> bool:
        """设置最终结果"""
        async with self._lock:
            if request_id not in self._states:
                return False
            
            state = self._states[request_id]
            if text:
                state.final_text = text
            if audio:
                state.response_audio = audio
            
            return True
    
    async def get_state(self, request_id: str) -> Optional[PipelineState]:
        """获取流水线状态"""
        async with self._lock:
            return self._states.get(request_id)
    
    async def get_all_states(self) -> Dict[str, PipelineState]:
        """获取所有流水线状态"""
        async with self._lock:
            return self._states.copy()
    
    async def get_metrics(self) -> PipelineMetrics:
        """获取性能指标"""
        async with self._lock:
            return self._metrics
    
    async def cleanup_old_states(self, max_age_hours: int = 24) -> int:
        """清理过旧的状态"""
        async with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            removed_count = 0
            
            to_remove = []
            for request_id, state in self._states.items():
                if state.end_time and state.end_time < cutoff_time:
                    to_remove.append(request_id)
            
            for request_id in to_remove:
                del self._states[request_id]
                removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old states")
            
            return removed_count
    
    async def _save_state_history(self, request_id: str, old_stage: ProcessingStage, 
                                 new_stage: ProcessingStage):
        """保存状态历史"""
        history_entry = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'old_stage': old_stage.value,
            'new_stage': new_stage.value
        }
        
        self._state_history.append(history_entry)
        
        # 保持历史记录在限制范围内
        if len(self._state_history) > self._max_history_size:
            self._state_history = self._state_history[-self._max_history_size:]
    
    async def _update_metrics(self, state: PipelineState, old_status: PipelineStatus):
        """更新性能指标"""
        if state.end_time and state.start_time:
            processing_time = (state.end_time - state.start_time).total_seconds()
            
            # 更新平均处理时间
            if self._metrics.successful_requests + self._metrics.failed_requests > 0:
                total_completed = self._metrics.successful_requests + self._metrics.failed_requests
                self._metrics.average_processing_time = (
                    (self._metrics.average_processing_time * (total_completed - 1) + processing_time) 
                    / total_completed
                )
            
            # 更新成功/失败计数
            if state.status == PipelineStatus.COMPLETED:
                self._metrics.successful_requests += 1
            elif state.status == PipelineStatus.FAILED:
                self._metrics.failed_requests += 1
                if state.error_type:
                    self._metrics.error_counts[state.error_type] = \
                        self._metrics.error_counts.get(state.error_type, 0) + 1
        
        self._metrics.last_updated = datetime.now()
    
    async def get_processing_time(self, request_id: str, stage: ProcessingStage) -> Optional[float]:
        """获取特定阶段的处理时间"""
        async with self._lock:
            if request_id not in self._states:
                return None
            
            state = self._states[request_id]
            return state.processing_times.get(stage.value)
    
    async def get_active_requests(self) -> List[str]:
        """获取活跃的请求ID列表"""
        async with self._lock:
            active_ids = []
            for request_id, state in self._states.items():
                if state.status == PipelineStatus.PROCESSING:
                    active_ids.append(request_id)
            return active_ids
    
    async def cancel_request(self, request_id: str) -> bool:
        """取消请求"""
        async with self._lock:
            if request_id not in self._states:
                return False
            
            state = self._states[request_id]
            if state.status == PipelineStatus.PROCESSING:
                state.status = PipelineStatus.CANCELLED
                state.end_time = datetime.now()
                return True
            
            return False