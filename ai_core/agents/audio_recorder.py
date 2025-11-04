"""
音频录制模块

提供实时音频监听和录制功能
"""

import pyaudio
import numpy as np
import threading
import time
import wave
import io
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import logging

from .audio_buffer import AudioChunk, AudioBuffer

logger = logging.getLogger(__name__)

@dataclass
class RecordingConfig:
    """录制配置"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "int16"
    device_index: Optional[int] = None
    input: bool = True
    output: bool = False

class AudioRecorder:
    """音频录制器"""
    
    def __init__(self, 
                 config: RecordingConfig,
                 buffer_size: int = 4096,
                 callback: Optional[Callable[[AudioChunk], None]] = None):
        """
        初始化音频录制器
        
        Args:
            config: 录制配置
            buffer_size: 缓冲区大小
            callback: 音频数据回调函数
        """
        self.config = config
        self.callback = callback
        
        # PyAudio实例
        self.audio = pyaudio.PyAudio()
        
        # 音频缓冲区
        self.buffer = AudioBuffer(max_size=buffer_size)
        
        # 录制状态
        self.is_recording = False
        self.is_paused = False
        self.stream: Optional[pyaudio.Stream] = None
        
        # 线程控制
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 统计信息
        self.total_recorded_samples = 0
        self.recording_start_time = 0.0
        self.last_chunk_time = 0.0
        
        # 设备信息
        self.input_device_info = None
        self.available_devices = []
        
        logger.info(f"音频录制器初始化完成: {config}")
    
    def initialize(self) -> bool:
        """
        初始化录制器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 获取可用设备
            self._get_available_devices()
            
            # 检查设备
            if not self._check_input_device():
                logger.error("没有找到可用的输入设备")
                return False
            
            # 设置回调
            if self.callback:
                self.buffer.on_audio_ready = self._on_audio_ready
            
            logger.info("音频录制器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"音频录制器初始化失败: {e}")
            return False
    
    def start_recording(self) -> bool:
        """
        开始录制
        
        Returns:
            bool: 是否成功开始录制
        """
        if self.is_recording:
            logger.warning("录制已在进行中")
            return True
        
        try:
            # 创建音频流
            self.stream = self.audio.open(
                format=self._get_format(),
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=self.config.input,
                output=self.config.output,
                input_device_index=self.config.device_index,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback,
                start=False
            )
            
            # 重置状态
            self.stop_event.clear()
            self.is_recording = True
            self.is_paused = False
            self.recording_start_time = time.time()
            self.total_recorded_samples = 0
            
            # 启动录制线程
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.start()
            
            # 启动流
            self.stream.start_stream()
            
            logger.info("开始音频录制")
            return True
            
        except Exception as e:
            logger.error(f"开始录制失败: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> bool:
        """
        停止录制
        
        Returns:
            bool: 是否成功停止录制
        """
        if not self.is_recording:
            logger.warning("录制未在进行中")
            return True
        
        try:
            # 设置停止事件
            self.stop_event.set()
            self.is_recording = False
            
            # 停止流
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # 等待录制线程结束
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)
            
            # 清空缓冲区
            self.buffer.clear_buffer()
            
            logger.info("停止音频录制")
            return True
            
        except Exception as e:
            logger.error(f"停止录制失败: {e}")
            return False
    
    def pause_recording(self) -> None:
        """暂停录制"""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            if self.stream:
                self.stream.stop_stream()
            logger.info("暂停音频录制")
    
    def resume_recording(self) -> None:
        """恢复录制"""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            if self.stream:
                self.stream.start_stream()
            logger.info("恢复音频录制")
    
    def _audio_callback(self, in_data, frame_count, time_info, status) -> tuple:
        """
        音频数据回调函数
        
        Args:
            in_data: 输入音频数据
            frame_count: 帧数
            time_info: 时间信息
            status: 状态
            
        Returns:
            tuple: (输出数据, 状态)
        """
        if status:
            logger.warning(f"音频流状态: {status}")
        
        if in_data and not self.is_paused:
            try:
                # 转换音频数据
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                
                # 创建音频块
                chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels
                )
                
                # 添加到缓冲区
                self.buffer.add_audio_chunk(chunk)
                
                # 更新统计信息
                self.total_recorded_samples += len(audio_data)
                self.last_chunk_time = time.time()
                
                # 调用回调函数
                if self.callback:
                    self.callback(chunk)
                
            except Exception as e:
                logger.error(f"处理音频数据失败: {e}")
        
        return (None, pyaudio.paContinue)
    
    def _recording_loop(self) -> None:
        """录制循环"""
        logger.info("录制线程启动")
        
        while not self.stop_event.is_set():
            try:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # 检查缓冲区状态
                if self.buffer.check_timeout():
                    logger.warning("录制缓冲区超时")
                    break
                
                time.sleep(0.01)  # 10ms睡眠
                
            except Exception as e:
                logger.error(f"录制循环错误: {e}")
                break
        
        logger.info("录制线程结束")
    
    def _on_audio_ready(self) -> None:
        """音频数据就绪回调"""
        # 可以在这里处理音频数据
        pass
    
    def _get_format(self) -> int:
        """
        获取音频格式
        
        Returns:
            int: PyAudio格式常量
        """
        format_map = {
            "int16": pyaudio.paInt16,
            "int32": pyaudio.paInt32,
            "float32": pyaudio.paFloat32,
        }
        return format_map.get(self.config.format, pyaudio.paInt16)
    
    def _get_available_devices(self) -> None:
        """获取可用设备列表"""
        try:
            self.available_devices = []
            
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                
                if device_info['maxInputChannels'] > 0:
                    self.available_devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
            
            logger.info(f"找到 {len(self.available_devices)} 个输入设备")
            
        except Exception as e:
            logger.error(f"获取设备列表失败: {e}")
    
    def _check_input_device(self) -> bool:
        """
        检查输入设备
        
        Returns:
            bool: 是否有可用输入设备
        """
        try:
            if self.config.device_index is not None:
                # 检查指定设备
                device_info = self.audio.get_device_info_by_index(self.config.device_index)
                if device_info['maxInputChannels'] > 0:
                    self.input_device_info = device_info
                    return True
            else:
                # 使用默认输入设备
                default_device = self.audio.get_default_input_device_info()
                if default_device['maxInputChannels'] > 0:
                    self.input_device_info = default_device
                    self.config.device_index = default_device['index']
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查输入设备失败: {e}")
            return False
    
    def get_current_audio(self) -> Optional[np.ndarray]:
        """
        获取当前音频数据
        
        Returns:
            Optional[np.ndarray]: 当前音频数据
        """
        return self.buffer.get_pre_recorded_data()
    
    def save_recording(self, filename: str, duration: Optional[float] = None) -> bool:
        """
        保存录制到文件
        
        Args:
            filename: 文件名
            duration: 保存时长（秒），None表示保存所有数据
            
        Returns:
            bool: 是否成功保存
        """
        try:
            # 获取音频数据
            audio_data = self.get_audio_data(duration)
            if audio_data is None or len(audio_data) == 0:
                logger.warning("没有音频数据可保存")
                return False
            
            # 保存为WAV文件
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self._get_format()))
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            logger.info(f"录制已保存到: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"保存录制失败: {e}")
            return False
    
    def get_audio_data(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        获取音频数据
        
        Args:
            duration: 获取时长（秒）
            
        Returns:
            Optional[np.ndarray]: 音频数据
        """
        try:
            if duration is None:
                # 获取所有数据
                chunks = []
                while True:
                    chunk = self.buffer.get_audio_data(blocking=False)
                    if chunk is None:
                        break
                    chunks.append(chunk.data)
                
                if chunks:
                    return np.concatenate(chunks)
                return None
            else:
                # 获取指定时长的数据
                target_samples = int(duration * self.config.sample_rate)
                chunks = []
                total_samples = 0
                
                while total_samples < target_samples:
                    chunk = self.buffer.get_audio_data(blocking=False)
                    if chunk is None:
                        break
                    
                    remaining = target_samples - total_samples
                    if len(chunk.data) > remaining:
                        chunk_data = chunk.data[:remaining]
                    else:
                        chunk_data = chunk.data
                    
                    chunks.append(chunk_data)
                    total_samples += len(chunk_data)
                
                if chunks:
                    return np.concatenate(chunks)
                return None
                
        except Exception as e:
            logger.error(f"获取音频数据失败: {e}")
            return None
    
    def get_recording_statistics(self) -> Dict[str, Any]:
        """
        获取录制统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            current_time = time.time()
            recording_duration = current_time - self.recording_start_time if self.recording_start_time > 0 else 0
            
            buffer_stats = self.buffer.get_statistics()
            
            return {
                "is_recording": self.is_recording,
                "is_paused": self.is_paused,
                "recording_duration": recording_duration,
                "total_samples": self.total_recorded_samples,
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "chunk_size": self.config.chunk_size,
                "device_index": self.config.device_index,
                "buffer_statistics": buffer_stats,
                "available_devices": len(self.available_devices),
                "last_chunk_time": self.last_chunk_time
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def list_audio_devices(self) -> List[Dict[str, Any]]:
        """
        列出所有音频设备
        
        Returns:
            List[Dict[str, Any]]: 设备列表
        """
        return self.available_devices.copy()
    
    def set_device(self, device_index: int) -> bool:
        """
        设置音频设备
        
        Args:
            device_index: 设备索引
            
        Returns:
            bool: 是否成功设置
        """
        try:
            device_info = self.audio.get_device_info_by_index(device_index)
            if device_info['maxInputChannels'] > 0:
                self.config.device_index = device_index
                self.input_device_info = device_info
                logger.info(f"设置音频设备: {device_info['name']}")
                return True
            else:
                logger.error(f"设备 {device_index} 不支持输入")
                return False
                
        except Exception as e:
            logger.error(f"设置音频设备失败: {e}")
            return False
    
    def terminate(self) -> None:
        """终止录制器"""
        try:
            self.stop_recording()
            self.audio.terminate()
            logger.info("音频录制器已终止")
        except Exception as e:
            logger.error(f"终止录制器失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.terminate()