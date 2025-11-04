"""
语音数据缓冲和处理模块

提供音频数据的缓冲、队列管理和预处理功能
"""

import numpy as np
import queue
import threading
import time
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """音频数据块"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int
    
    @property
    def duration(self) -> float:
        """获取音频时长（秒）"""
        return len(self.data) / self.sample_rate

class AudioBuffer:
    """音频缓冲区管理器"""
    
    def __init__(self, 
                 max_size: int = 4096,
                 timeout: float = 2.0,
                 min_size: int = 1024,
                 pre_recording_duration: float = 1.0):
        """
        初始化音频缓冲区
        
        Args:
            max_size: 最大缓冲区大小（样本数）
            timeout: 缓冲区超时时间（秒）
            min_size: 最小缓冲区大小（样本数）
            pre_recording_duration: 预录制时长（秒）
        """
        self.max_size = max_size
        self.timeout = timeout
        self.min_size = min_size
        self.pre_recording_duration = pre_recording_duration
        
        # 音频数据队列
        self.audio_queue = queue.Queue(maxsize=100)
        self.pre_record_buffer = deque(maxlen=int(pre_recording_duration * 16000))  # 假设16kHz采样率
        
        # 缓冲区状态
        self.is_recording = False
        self.is_processing = False
        self.buffer_lock = threading.Lock()
        self.last_activity_time = time.time()
        
        # 回调函数
        self.on_audio_ready: Optional[Callable] = None
        self.on_buffer_full: Optional[Callable] = None
        self.on_timeout: Optional[Callable] = None
        
        # 统计信息
        self.total_chunks = 0
        self.total_duration = 0.0
        self.avg_chunk_duration = 0.0
        
        logger.info(f"音频缓冲区初始化完成: max_size={max_size}, timeout={timeout}s")
    
    def add_audio_chunk(self, chunk: AudioChunk) -> bool:
        """
        添加音频数据块到缓冲区
        
        Args:
            chunk: 音频数据块
            
        Returns:
            bool: 是否成功添加
        """
        try:
            with self.buffer_lock:
                # 添加到预录制缓冲区
                if len(self.pre_record_buffer) + len(chunk.data) > self.pre_record_buffer.maxlen:
                    # 移除旧数据
                    excess = len(self.pre_record_buffer) + len(chunk.data) - self.pre_record_buffer.maxlen
                    for _ in range(excess):
                        if self.pre_record_buffer:
                            self.pre_record_buffer.popleft()
                
                self.pre_record_buffer.extend(chunk.data)
                
                # 添加到主队列
                if not self.audio_queue.full():
                    self.audio_queue.put_nowait(chunk)
                    self.total_chunks += 1
                    self.total_duration += chunk.duration
                    self.avg_chunk_duration = self.total_duration / self.total_chunks
                    self.last_activity_time = time.time()
                    
                    # 检查缓冲区状态
                    self._check_buffer_status()
                    return True
                else:
                    logger.warning("音频队列已满，丢弃音频块")
                    return False
                    
        except Exception as e:
            logger.error(f"添加音频块失败: {e}")
            return False
    
    def get_audio_data(self, blocking: bool = True, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """
        从缓冲区获取音频数据
        
        Args:
            blocking: 是否阻塞等待
            timeout: 超时时间
            
        Returns:
            AudioChunk: 音频数据块，如果超时返回None
        """
        try:
            if blocking:
                return self.audio_queue.get(timeout=timeout)
            else:
                return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_pre_recorded_data(self) -> Optional[np.ndarray]:
        """
        获取预录制的音频数据
        
        Returns:
            np.ndarray: 预录制的音频数据
        """
        with self.buffer_lock:
            if self.pre_record_buffer:
                return np.array(self.pre_record_buffer, dtype=np.int16)
            return None
    
    def clear_buffer(self) -> None:
        """清空缓冲区"""
        with self.buffer_lock:
            # 清空队列
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # 清空预录制缓冲区
            self.pre_record_buffer.clear()
            
            # 重置统计信息
            self.total_chunks = 0
            self.total_duration = 0.0
            self.avg_chunk_duration = 0.0
            
            logger.info("音频缓冲区已清空")
    
    def get_buffer_size(self) -> Tuple[int, int]:
        """
        获取缓冲区大小
        
        Returns:
            Tuple[int, int]: (队列大小, 预录制缓冲区大小)
        """
        with self.buffer_lock:
            queue_size = self.audio_queue.qsize()
            pre_record_size = len(self.pre_record_buffer)
            return queue_size, pre_record_size
    
    def is_buffer_ready(self) -> bool:
        """
        检查缓冲区是否准备好处理
        
        Returns:
            bool: 是否准备好
        """
        with self.buffer_lock:
            queue_size, pre_record_size = self.get_buffer_size()
            return (queue_size > 0 and 
                    pre_record_size >= self.min_size and 
                    not self.is_processing)
    
    def start_recording(self) -> None:
        """开始录制"""
        self.is_recording = True
        self.last_activity_time = time.time()
        logger.info("开始音频录制")
    
    def stop_recording(self) -> None:
        """停止录制"""
        self.is_recording = False
        logger.info("停止音频录制")
    
    def start_processing(self) -> None:
        """开始处理"""
        self.is_processing = True
    
    def stop_processing(self) -> None:
        """停止处理"""
        self.is_processing = False
    
    def check_timeout(self) -> bool:
        """
        检查是否超时
        
        Returns:
            bool: 是否超时
        """
        elapsed = time.time() - self.last_activity_time
        return elapsed > self.timeout
    
    def _check_buffer_status(self) -> None:
        """检查缓冲区状态并触发回调"""
        queue_size, pre_record_size = self.get_buffer_size()
        
        # 检查缓冲区是否已满
        if queue_size >= self.audio_queue.maxsize * 0.9:  # 90%满
            if self.on_buffer_full:
                self.on_buffer_full(queue_size, self.audio_queue.maxsize)
        
        # 检查是否准备好处理
        if self.is_buffer_ready() and self.on_audio_ready:
            self.on_audio_ready()
        
        # 检查超时
        if self.check_timeout() and self.on_timeout:
            self.on_timeout(elapsed=time.time() - self.last_activity_time)
    
    def get_statistics(self) -> dict:
        """
        获取缓冲区统计信息
        
        Returns:
            dict: 统计信息
        """
        with self.buffer_lock:
            queue_size, pre_record_size = self.get_buffer_size()
            return {
                "queue_size": queue_size,
                "pre_record_size": pre_record_size,
                "total_chunks": self.total_chunks,
                "total_duration": self.total_duration,
                "avg_chunk_duration": self.avg_chunk_duration,
                "is_recording": self.is_recording,
                "is_processing": self.is_processing,
                "last_activity": self.last_activity_time,
                "buffer_utilization": queue_size / self.audio_queue.maxsize if self.audio_queue.maxsize > 0 else 0
            }
    
    def combine_audio_chunks(self, chunks: List[AudioChunk]) -> np.ndarray:
        """
        合并多个音频块
        
        Args:
            chunks: 音频块列表
            
        Returns:
            np.ndarray: 合并后的音频数据
        """
        if not chunks:
            return np.array([], dtype=np.int16)
        
        # 确保所有块具有相同的采样率和声道数
        sample_rate = chunks[0].sample_rate
        channels = chunks[0].channels
        
        for chunk in chunks:
            if chunk.sample_rate != sample_rate or chunk.channels != channels:
                raise ValueError("音频块参数不匹配")
        
        # 合并音频数据
        combined_data = np.concatenate([chunk.data for chunk in chunks])
        return combined_data
    
    def resample_audio(self, data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """
        重采样音频数据
        
        Args:
            data: 原始音频数据
            original_rate: 原始采样率
            target_rate: 目标采样率
            
        Returns:
            np.ndarray: 重采样后的音频数据
        """
        if original_rate == target_rate:
            return data
        
        # 简单的重采样实现（实际项目中建议使用librosa或scipy）
        ratio = target_rate / original_rate
        new_length = int(len(data) * ratio)
        
        if new_length <= 0:
            return np.array([], dtype=data.dtype)
        
        # 线性插值重采样
        indices = np.linspace(0, len(data) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(data)), data)
        
        return resampled.astype(data.dtype)