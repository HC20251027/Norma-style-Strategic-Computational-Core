"""
实时流式语音处理器
支持音频流的实时处理、转录和合成
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, AsyncGenerator, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import deque
import threading
import queue

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """流事件类型"""
    AUDIO_CHUNK = "audio_chunk"
    TRANSCRIPTION = "transcription"
    SYNTHESIS = "synthesis"
    EMOTION_CHANGE = "emotion_change"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class StreamEvent:
    """流事件"""
    type: StreamEventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class StreamConfig:
    """流配置"""
    chunk_size: int = 1024  # 音频块大小
    sample_rate: int = 16000  # 采样率
    channels: int = 1  # 声道数
    buffer_size: int = 30  # 缓冲区大小(秒)
    overlap_size: int = 512  # 重叠大小
    processing_interval: float = 0.1  # 处理间隔(秒)
    enable_vad: bool = True  # 启用语音活动检测
    vad_threshold: float = 0.5  # VAD阈值
    max_latency: float = 0.5  # 最大延迟(秒)


class VoiceActivityDetector:
    """语音活动检测器"""
    
    def __init__(self, threshold: float = 0.5, window_size: int = 1024):
        self.threshold = threshold
        self.window_size = window_size
        self.energy_history = deque(maxlen=10)
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_counter = 0
        
    def detect(self, audio_chunk: np.ndarray) -> bool:
        """检测语音活动"""
        # 计算音频能量
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        self.energy_history.append(energy)
        
        # 计算动态阈值
        if len(self.energy_history) >= 5:
            avg_energy = np.mean(list(self.energy_history)[:-1])
            dynamic_threshold = avg_energy * (1 + self.threshold)
        else:
            dynamic_threshold = self.threshold
        
        # 语音检测逻辑
        if energy > dynamic_threshold:
            self.speech_counter += 1
            self.silence_counter = 0
            if self.speech_counter >= 3:  # 连续3次检测到语音才认为是语音
                self.is_speaking = True
        else:
            self.silence_counter += 1
            self.speech_counter = 0
            if self.silence_counter >= 10:  # 连续10次检测到静音才认为语音结束
                self.is_speaking = False
        
        return self.is_speaking
    
    def reset(self):
        """重置检测器"""
        self.energy_history.clear()
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_counter = 0


class AudioBuffer:
    """音频缓冲区"""
    
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.buffer = deque()
        self.total_duration = 0.0
        self.lock = threading.Lock()
    
    def add_chunk(self, audio_chunk: np.ndarray, sample_rate: int):
        """添加音频块"""
        with self.lock:
            chunk_duration = len(audio_chunk) / sample_rate
            self.buffer.append(audio_chunk)
            self.total_duration += chunk_duration
            
            # 保持缓冲区大小
            while self.total_duration > self.max_size:
                removed_chunk = self.buffer.popleft()
                removed_duration = len(removed_chunk) / sample_rate
                self.total_duration -= removed_duration
    
    def get_buffered_audio(self) -> np.ndarray:
        """获取缓冲的音频"""
        with self.lock:
            if not self.buffer:
                return np.array([])
            return np.concatenate(list(self.buffer))
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
            self.total_duration = 0.0
    
    def get_duration(self) -> float:
        """获取缓冲区总时长"""
        return self.total_duration


class StreamingProcessor:
    """实时流式语音处理器"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.buffer = AudioBuffer(config.buffer_size)
        self.vad = VoiceActivityDetector(config.vad_threshold) if config.enable_vad else None
        self.is_running = False
        self.event_queue = queue.Queue()
        self.callbacks: Dict[StreamEventType, List[Callable]] = {}
        self.processing_task: Optional[asyncio.Task] = None
        self.last_process_time = 0.0
        
    def register_callback(self, event_type: StreamEventType, callback: Callable):
        """注册事件回调"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def _emit_event(self, event: StreamEvent):
        """发射事件"""
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("事件队列已满，丢弃事件")
        
        # 调用注册的回调
        if event.type in self.callbacks:
            for callback in self.callbacks[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"回调执行失败: {e}")
    
    async def process_audio_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        asr_callback: Optional[Callable] = None,
        tts_callback: Optional[Callable] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """处理音频流"""
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        try:
            async for audio_chunk in audio_stream:
                # 解析音频数据
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0  # 归一化
                
                # 添加到缓冲区
                self.buffer.add_chunk(audio_array, self.config.sample_rate)
                
                # VAD检测
                if self.vad:
                    is_speaking = self.vad.detect(audio_array)
                    if not is_speaking and self.vad.silence_counter == 10:
                        # 语音结束，发射事件
                        self._emit_event(StreamEvent(
                            type=StreamEventType.AUDIO_CHUNK,
                            data=audio_chunk,
                            metadata={"vad": "speech_end"}
                        ))
                
                # 发射音频块事件
                self._emit_event(StreamEvent(
                    type=StreamEventType.AUDIO_CHUNK,
                    data=audio_chunk,
                    metadata={
                        "buffer_duration": self.buffer.get_duration(),
                        "vad": self.vad.is_speaking if self.vad else None
                    }
                ))
                
                # 检查是否需要处理
                current_time = time.time()
                if current_time - self.last_process_time >= self.config.processing_interval:
                    await self._process_buffered_audio(asr_callback, tts_callback)
                    self.last_process_time = current_time
                
                # 处理事件队列
                while not self.event_queue.empty():
                    try:
                        event = self.event_queue.get_nowait()
                        yield event
                    except queue.Empty:
                        break
                        
        except Exception as e:
            logger.error(f"音频流处理错误: {e}")
            self._emit_event(StreamEvent(
                type=StreamEventType.ERROR,
                data=str(e)
            ))
        finally:
            self.is_running = False
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
    
    async def _processing_loop(self):
        """后台处理循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.processing_interval)
                if self.buffer.get_duration() >= 1.0:  # 至少1秒音频才处理
                    # 这里可以添加后台处理逻辑
                    pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"处理循环错误: {e}")
    
    async def _process_buffered_audio(
        self,
        asr_callback: Optional[Callable] = None,
        tts_callback: Optional[Callable] = None
    ):
        """处理缓冲的音频"""
        if self.buffer.get_duration() < 1.0:
            return
        
        try:
            buffered_audio = self.buffer.get_buffered_audio()
            
            # ASR处理
            if asr_callback:
                try:
                    result = await asr_callback(buffered_audio)
                    self._emit_event(StreamEvent(
                        type=StreamEventType.TRANSCRIPTION,
                        data=result
                    ))
                except Exception as e:
                    logger.error(f"ASR处理错误: {e}")
            
            # TTS处理
            if tts_callback:
                try:
                    result = await tts_callback(buffered_audio)
                    self._emit_event(StreamEvent(
                        type=StreamEventType.SYNTHESIS,
                        data=result
                    ))
                except Exception as e:
                    logger.error(f"TTS处理错误: {e}")
            
        except Exception as e:
            logger.error(f"音频处理错误: {e}")
    
    async def process_text_stream(
        self,
        text_stream: AsyncGenerator[str, None],
        tts_engine,
        voice_id: str,
        emotion_controller=None
    ) -> AsyncGenerator[StreamEvent, None]:
        """处理文本流"""
        try:
            async for text_chunk in text_stream:
                # 分析情感
                if emotion_controller:
                    emotion = emotion_controller.analyze_text_emotion(text_chunk)
                    self._emit_event(StreamEvent(
                        type=StreamEventType.EMOTION_CHANGE,
                        data=emotion
                    ))
                
                # TTS合成
                try:
                    audio_chunks = []
                    async for audio_chunk in tts_engine.synthesize_stream(
                        text_chunk, voice_id
                    ):
                        audio_chunks.append(audio_chunk)
                        self._emit_event(StreamEvent(
                            type=StreamEventType.SYNTHESIS,
                            data=audio_chunk,
                            metadata={"text": text_chunk}
                        ))
                    
                    # 合并音频块
                    if audio_chunks:
                        combined_audio = b''.join(audio_chunks)
                        self._emit_event(StreamEvent(
                            type=StreamEventType.COMPLETE,
                            data=combined_audio,
                            metadata={"text": text_chunk}
                        ))
                
                except Exception as e:
                    logger.error(f"TTS合成错误: {e}")
                    self._emit_event(StreamEvent(
                        type=StreamEventType.ERROR,
                        data=str(e)
                    ))
        
        except Exception as e:
            logger.error(f"文本流处理错误: {e}")
            self._emit_event(StreamEvent(
                type=StreamEventType.ERROR,
                data=str(e)
            ))
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """获取流统计信息"""
        return {
            "is_running": self.is_running,
            "buffer_duration": self.buffer.get_duration(),
            "buffer_size": len(self.buffer.buffer),
            "vad_speaking": self.vad.is_speaking if self.vad else None,
            "queue_size": self.event_queue.qsize(),
            "last_process_time": self.last_process_time
        }
    
    def reset(self):
        """重置处理器"""
        self.buffer.clear()
        if self.vad:
            self.vad.reset()
        self.last_process_time = 0.0
        
        # 清空事件队列
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break


class StreamManager:
    """流管理器"""
    
    def __init__(self):
        self.active_streams: Dict[str, StreamingProcessor] = {}
        self.stream_configs: Dict[str, StreamConfig] = {}
        
    def create_stream(
        self,
        stream_id: str,
        config: Optional[StreamConfig] = None
    ) -> StreamingProcessor:
        """创建新的流"""
        if stream_id in self.active_streams:
            raise ValueError(f"流 {stream_id} 已存在")
        
        if config is None:
            config = StreamConfig()
        
        processor = StreamingProcessor(config)
        self.active_streams[stream_id] = processor
        self.stream_configs[stream_id] = config
        
        logger.info(f"创建流 {stream_id}")
        return processor
    
    def get_stream(self, stream_id: str) -> Optional[StreamingProcessor]:
        """获取流"""
        return self.active_streams.get(stream_id)
    
    def close_stream(self, stream_id: str):
        """关闭流"""
        if stream_id in self.active_streams:
            processor = self.active_streams[stream_id]
            processor.is_running = False
            del self.active_streams[stream_id]
            
            if stream_id in self.stream_configs:
                del self.stream_configs[stream_id]
            
            logger.info(f"关闭流 {stream_id}")
    
    def get_all_streams(self) -> Dict[str, StreamingProcessor]:
        """获取所有流"""
        return self.active_streams.copy()
    
    def cleanup(self):
        """清理所有流"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)


# 实用工具函数
async def create_audio_stream_from_file(
    file_path: str,
    chunk_size: int = 1024
) -> AsyncGenerator[bytes, None]:
    """从文件创建音频流"""
    import wave
    
    try:
        with wave.open(file_path, 'rb') as wav_file:
            while True:
                chunk = wav_file.readframes(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        logger.error(f"从文件创建音频流失败: {e}")
        raise


async def create_audio_stream_from_microphone(
    device_index: Optional[int] = None,
    sample_rate: int = 16000,
    chunk_size: int = 1024
) -> AsyncGenerator[bytes, None]:
    """从麦克风创建音频流"""
    try:
        import pyaudio
        
        audio = pyaudio.PyAudio()
        
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )
        
        try:
            while True:
                data = stream.read(chunk_size, exception_on_overflow=False)
                yield data
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
    except ImportError:
        raise ImportError("请安装pyaudio: pip install pyaudio")
    except Exception as e:
        logger.error(f"从麦克风创建音频流失败: {e}")
        raise


def audio_to_stream_events(
    audio_data: bytes,
    sample_rate: int = 16000
) -> List[StreamEvent]:
    """将音频数据转换为流事件"""
    events = []
    
    # 分块处理
    chunk_size = 1024
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        events.append(StreamEvent(
            type=StreamEventType.AUDIO_CHUNK,
            data=chunk,
            metadata={
                "chunk_index": i // chunk_size,
                "sample_rate": sample_rate
            }
        ))
    
    return events