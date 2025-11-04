"""
语音网关主服务类

整合所有语音处理功能，提供完整的语音交互服务
"""

import asyncio
import websockets
import json
import logging
import time
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, asdict
from enum import Enum
import threading

from .config import VoiceGatewayConfig
from .audio_recorder import AudioRecorder, RecordingConfig
from .audio_processor import AudioProcessor
from .audio_buffer import AudioChunk, AudioBuffer
from .whisper_asr import WhisperASR, ASRResult
from .tts_synthesizer import TTSSynthesizer, TTSResult

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """连接状态"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class VoiceGatewayEvent:
    """语音网关事件"""
    type: str
    data: Dict[str, Any]
    timestamp: float
    client_id: Optional[str] = None

class VoiceGateway:
    """语音网关主服务"""
    
    def __init__(self, config: Optional[VoiceGatewayConfig] = None):
        """
        初始化语音网关
        
        Args:
            config: 配置对象
        """
        self.config = config or VoiceGatewayConfig()
        
        # 组件实例
        self.audio_recorder: Optional[AudioRecorder] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.audio_buffer: Optional[AudioBuffer] = None
        self.whisper_asr: Optional[WhisperASR] = None
        self.tts_synthesizer: Optional[TTSSynthesizer] = None
        
        # 服务状态
        self.is_running = False
        self.is_initialized = False
        
        # WebSocket服务
        self.websocket_server: Optional[websockets.WebSocketServer] = None
        self.clients: Dict[str, websockets.WebSocket] = {}
        self.client_states: Dict[str, ConnectionState] = {}
        
        # 事件处理
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue: List[VoiceGatewayEvent] = []
        self.event_lock = threading.Lock()
        
        # 统计信息
        self.start_time = 0
        self.total_connections = 0
        self.active_connections = 0
        self.total_requests = 0
        self.successful_requests = 0
        
        logger.info("语音网关初始化完成")
    
    async def initialize(self) -> bool:
        """
        初始化语音网关服务
        
        Returns:
            bool: 是否成功初始化
        """
        try:
            logger.info("开始初始化语音网关服务...")
            
            # 初始化音频处理器
            self.audio_processor = AudioProcessor(
                sample_rate=self.config.AUDIO_CONFIG["sample_rate"],
                noise_reduction=self.config.AUDIO_PROCESSING_CONFIG["noise_reduction"],
                noise_reduction_level=self.config.AUDIO_PROCESSING_CONFIG["noise_reduction_level"],
                normalize=self.config.AUDIO_PROCESSING_CONFIG["normalize"],
                high_pass_filter=self.config.AUDIO_PROCESSING_CONFIG["high_pass_filter"],
                vad_threshold=self.config.AUDIO_PROCESSING_CONFIG["vad_threshold"]
            )
            
            # 初始化音频缓冲区
            self.audio_buffer = AudioBuffer(
                max_size=self.config.BUFFER_CONFIG["max_buffer_size"],
                timeout=self.config.BUFFER_CONFIG["buffer_timeout"],
                min_size=self.config.BUFFER_CONFIG["min_buffer_size"],
                pre_recording_duration=self.config.BUFFER_CONFIG["pre_recording_duration"]
            )
            
            # 初始化Whisper ASR
            self.whisper_asr = WhisperASR(
                model_name=self.config.WHISPER_CONFIG["model_name"],
                language=self.config.WHISPER_CONFIG["language"],
                task=self.config.WHISPER_CONFIG["task"],
                device=self.config.WHISPER_CONFIG["device"],
                temperature=self.config.WHISPER_CONFIG["temperature"]
            )
            
            if not self.whisper_asr.load_model():
                logger.error("Whisper模型加载失败")
                return False
            
            # 初始化TTS合成器
            self.tts_synthesizer = TTSSynthesizer(
                engine=self.config.TTS_CONFIG["engine"],
                voice=self.config.TTS_CONFIG["voice"],
                rate=self.config.TTS_CONFIG["rate"],
                volume=self.config.TTS_CONFIG["volume"],
                pitch=self.config.TTS_CONFIG["pitch"],
                format=self.config.TTS_CONFIG["format"]
            )
            
            if not self.tts_synthesizer.initialize():
                logger.warning("TTS合成器初始化失败，某些功能可能不可用")
            
            # 初始化音频录制器
            recording_config = RecordingConfig(
                sample_rate=self.config.AUDIO_CONFIG["sample_rate"],
                channels=self.config.AUDIO_CONFIG["channels"],
                chunk_size=self.config.AUDIO_CONFIG["chunk_size"],
                format=self.config.AUDIO_CONFIG["format"],
                device_index=self.config.AUDIO_CONFIG["device_index"]
            )
            
            self.audio_recorder = AudioRecorder(
                config=recording_config,
                buffer_size=self.config.BUFFER_CONFIG["max_buffer_size"],
                callback=self._on_audio_chunk
            )
            
            if not self.audio_recorder.initialize():
                logger.error("音频录制器初始化失败")
                return False
            
            # 设置事件回调
            self.audio_buffer.on_audio_ready = self._on_audio_ready
            self.audio_buffer.on_buffer_full = self._on_buffer_full
            self.audio_buffer.on_timeout = self._on_buffer_timeout
            
            self.is_initialized = True
            logger.info("语音网关服务初始化完成")
            
            return True
            
        except Exception as e:
            logger.error(f"语音网关服务初始化失败: {e}")
            return False
    
    async def start(self) -> bool:
        """
        启动语音网关服务
        
        Returns:
            bool: 是否成功启动
        """
        if not self.is_initialized:
            logger.error("服务未初始化，请先调用initialize()")
            return False
        
        try:
            logger.info("启动语音网关服务...")
            
            self.is_running = True
            self.start_time = time.time()
            
            # 启动音频录制
            if not self.audio_recorder.start_recording():
                logger.error("音频录制启动失败")
                return False
            
            # 启动WebSocket服务器
            await self._start_websocket_server()
            
            # 启动事件处理循环
            asyncio.create_task(self._event_processing_loop())
            
            logger.info("语音网关服务启动完成")
            return True
            
        except Exception as e:
            logger.error(f"语音网关服务启动失败: {e}")
            self.is_running = False
            return False
    
    async def stop(self) -> None:
        """停止语音网关服务"""
        try:
            logger.info("停止语音网关服务...")
            
            self.is_running = False
            
            # 停止音频录制
            if self.audio_recorder:
                self.audio_recorder.stop_recording()
            
            # 关闭WebSocket服务器
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
            
            # 断开所有客户端连接
            for client_id, websocket in list(self.clients.items()):
                try:
                    await websocket.close()
                except Exception:
                    pass
            
            self.clients.clear()
            self.client_states.clear()
            
            logger.info("语音网关服务已停止")
            
        except Exception as e:
            logger.error(f"停止语音网关服务时出错: {e}")
    
    async def _start_websocket_server(self) -> None:
        """启动WebSocket服务器"""
        try:
            host = self.config.WEBSOCKET_CONFIG["host"]
            port = self.config.WEBSOCKET_CONFIG["port"]
            
            self.websocket_server = await websockets.serve(
                self._handle_client,
                host,
                port,
                max_connections=self.config.WEBSOCKET_CONFIG["max_connections"],
                ping_interval=self.config.WEBSOCKET_CONFIG["ping_interval"],
                ping_timeout=self.config.WEBSOCKET_CONFIG["ping_timeout"]
            )
            
            logger.info(f"WebSocket服务器启动: ws://{host}:{port}")
            
        except Exception as e:
            logger.error(f"WebSocket服务器启动失败: {e}")
            raise
    
    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """处理客户端连接"""
        client_id = f"client_{len(self.clients)}"
        
        try:
            logger.info(f"客户端连接: {client_id}")
            
            # 注册客户端
            self.clients[client_id] = websocket
            self.client_states[client_id] = ConnectionState.CONNECTED
            self.total_connections += 1
            self.active_connections += 1
            
            # 发送欢迎消息
            await self._send_to_client(client_id, {
                "type": "welcome",
                "data": {
                    "client_id": client_id,
                    "message": "欢迎连接语音网关服务",
                    "capabilities": ["asr", "tts", "audio_processing"]
                }
            })
            
            # 处理客户端消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_client_message(client_id, data)
                except json.JSONDecodeError:
                    await self._send_error_to_client(client_id, "无效的JSON格式")
                except Exception as e:
                    logger.error(f"处理客户端消息失败: {e}")
                    await self._send_error_to_client(client_id, str(e))
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端连接关闭: {client_id}")
        except Exception as e:
            logger.error(f"客户端处理错误: {e}")
            self.client_states[client_id] = ConnectionState.ERROR
        finally:
            # 清理客户端
            if client_id in self.clients:
                del self.clients[client_id]
            if client_id in self.client_states:
                del self.client_states[client_id]
            
            self.active_connections = max(0, self.active_connections - 1)
    
    async def _process_client_message(self, client_id: str, data: Dict[str, Any]) -> None:
        """处理客户端消息"""
        try:
            message_type = data.get("type")
            message_data = data.get("data", {})
            
            self.total_requests += 1
            
            if message_type == "start_recording":
                await self._handle_start_recording(client_id, message_data)
            elif message_type == "stop_recording":
                await self._handle_stop_recording(client_id, message_data)
            elif message_type == "transcribe":
                await self._handle_transcribe(client_id, message_data)
            elif message_type == "synthesize":
                await self._handle_synthesize(client_id, message_data)
            elif message_type == "get_status":
                await self._handle_get_status(client_id, message_data)
            elif message_type == "get_voices":
                await self._handle_get_voices(client_id, message_data)
            else:
                await self._send_error_to_client(client_id, f"未知的消息类型: {message_type}")
            
            self.successful_requests += 1
            
        except Exception as e:
            logger.error(f"处理客户端消息失败: {e}")
            await self._send_error_to_client(client_id, str(e))
    
    async def _handle_start_recording(self, client_id: str, data: Dict[str, Any]) -> None:
        """处理开始录制请求"""
        try:
            # 开始录制
            if self.audio_recorder.start_recording():
                await self._send_to_client(client_id, {
                    "type": "recording_started",
                    "data": {"message": "开始录制"}
                })
            else:
                await self._send_error_to_client(client_id, "录制启动失败")
        except Exception as e:
            await self._send_error_to_client(client_id, str(e))
    
    async def _handle_stop_recording(self, client_id: str, data: Dict[str, Any]) -> None:
        """处理停止录制请求"""
        try:
            # 停止录制
            if self.audio_recorder.stop_recording():
                await self._send_to_client(client_id, {
                    "type": "recording_stopped",
                    "data": {"message": "停止录制"}
                })
            else:
                await self._send_error_to_client(client_id, "录制停止失败")
        except Exception as e:
            await self._send_error_to_client(client_id, str(e))
    
    async def _handle_transcribe(self, client_id: str, data: Dict[str, Any]) -> None:
        """处理语音识别请求"""
        try:
            # 获取音频数据
            audio_data = self.audio_buffer.get_pre_recorded_data()
            if audio_data is None or len(audio_data) == 0:
                await self._send_error_to_client(client_id, "没有音频数据")
                return
            
            # 执行语音识别
            result = self.whisper_asr.transcribe(audio_data)
            
            if result:
                await self._send_to_client(client_id, {
                    "type": "transcription_result",
                    "data": {
                        "text": result.text,
                        "language": result.language,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time,
                        "segments": result.segments
                    }
                })
            else:
                await self._send_error_to_client(client_id, "语音识别失败")
                
        except Exception as e:
            await self._send_error_to_client(client_id, str(e))
    
    async def _handle_synthesize(self, client_id: str, data: Dict[str, Any]) -> None:
        """处理语音合成请求"""
        try:
            text = data.get("text", "")
            voice = data.get("voice")
            
            if not text:
                await self._send_error_to_client(client_id, "文本不能为空")
                return
            
            # 执行语音合成
            result = self.tts_synthesizer.synthesize(text, voice)
            
            if result:
                # 将音频数据转换为base64
                import base64
                audio_bytes = (result.audio_data * 32767).astype(np.int16).tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                await self._send_to_client(client_id, {
                    "type": "synthesis_result",
                    "data": {
                        "audio_data": audio_b64,
                        "sample_rate": result.sample_rate,
                        "duration": result.duration,
                        "text": result.text,
                        "voice": result.voice,
                        "processing_time": result.processing_time
                    }
                })
            else:
                await self._send_error_to_client(client_id, "语音合成失败")
                
        except Exception as e:
            await self._send_error_to_client(client_id, str(e))
    
    async def _handle_get_status(self, client_id: str, data: Dict[str, Any]) -> None:
        """处理获取状态请求"""
        try:
            status = self.get_service_status()
            await self._send_to_client(client_id, {
                "type": "status",
                "data": status
            })
        except Exception as e:
            await self._send_error_to_client(client_id, str(e))
    
    async def _handle_get_voices(self, client_id: str, data: Dict[str, Any]) -> None:
        """处理获取语音列表请求"""
        try:
            voices = self.tts_synthesizer.get_available_voices()
            await self._send_to_client(client_id, {
                "type": "voices",
                "data": {"voices": voices}
            })
        except Exception as e:
            await self._send_error_to_client(client_id, str(e))
    
    async def _send_to_client(self, client_id: str, message: Dict[str, Any]) -> None:
        """发送消息给客户端"""
        try:
            if client_id in self.clients:
                websocket = self.clients[client_id]
                await websocket.send(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送消息给客户端失败: {e}")
    
    async def _send_error_to_client(self, client_id: str, error_message: str) -> None:
        """发送错误消息给客户端"""
        await self._send_to_client(client_id, {
            "type": "error",
            "data": {"message": error_message}
        })
    
    def _on_audio_chunk(self, chunk: AudioChunk) -> None:
        """音频数据块回调"""
        try:
            # 处理音频数据
            processed_data = self.audio_processor.process_audio(chunk.data)
            
            # 语音活动检测
            is_speech, speech_prob = self.audio_processor.detect_voice_activity(processed_data)
            
            # 创建事件
            event = VoiceGatewayEvent(
                type="audio_chunk",
                data={
                    "chunk_id": len(self.event_queue),
                    "duration": chunk.duration,
                    "sample_rate": chunk.sample_rate,
                    "is_speech": is_speech,
                    "speech_probability": speech_prob,
                    "timestamp": chunk.timestamp
                },
                timestamp=time.time()
            )
            
            with self.event_lock:
                self.event_queue.append(event)
            
        except Exception as e:
            logger.error(f"处理音频块失败: {e}")
    
    def _on_audio_ready(self) -> None:
        """音频数据就绪回调"""
        # 可以在这里触发语音识别
        logger.debug("音频数据就绪")
    
    def _on_buffer_full(self, current_size: int, max_size: int) -> None:
        """缓冲区满回调"""
        logger.warning(f"音频缓冲区已满: {current_size}/{max_size}")
    
    def _on_buffer_timeout(self, elapsed: float) -> None:
        """缓冲区超时回调"""
        logger.warning(f"音频缓冲区超时: {elapsed:.2f}秒")
    
    async def _event_processing_loop(self) -> None:
        """事件处理循环"""
        while self.is_running:
            try:
                with self.event_lock:
                    if self.event_queue:
                        event = self.event_queue.pop(0)
                    else:
                        event = None
                
                if event:
                    await self._process_event(event)
                
                await asyncio.sleep(0.1)  # 100ms间隔
                
            except Exception as e:
                logger.error(f"事件处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, event: VoiceGatewayEvent) -> None:
        """处理事件"""
        try:
            # 调用事件处理器
            handlers = self.event_handlers.get(event.type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"事件处理器执行失败: {e}")
            
        except Exception as e:
            logger.error(f"处理事件失败: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        添加事件处理器
        
        Args:
            event_type: 事件类型
            handler: 处理器函数
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"添加事件处理器: {event_type}")
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        移除事件处理器
        
        Args:
            event_type: 事件类型
            handler: 处理器函数
        """
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                logger.info(f"移除事件处理器: {event_type}")
            except ValueError:
                pass
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        获取服务状态
        
        Returns:
            Dict[str, Any]: 服务状态信息
        """
        try:
            uptime = time.time() - self.start_time if self.start_time > 0 else 0
            
            # 获取组件状态
            recorder_stats = self.audio_recorder.get_recording_statistics() if self.audio_recorder else {}
            buffer_stats = self.audio_buffer.get_statistics() if self.audio_buffer else {}
            asr_stats = self.whisper_asr.get_model_info() if self.whisper_asr else {}
            tts_stats = self.tts_synthesizer.get_synthesis_statistics() if self.tts_synthesizer else {}
            
            return {
                "is_running": self.is_running,
                "is_initialized": self.is_initialized,
                "uptime": uptime,
                "active_connections": self.active_connections,
                "total_connections": self.total_connections,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "success_rate": self.successful_requests / max(self.total_requests, 1),
                "components": {
                    "audio_recorder": recorder_stats,
                    "audio_buffer": buffer_stats,
                    "whisper_asr": asr_stats,
                    "tts_synthesizer": tts_stats
                },
                "clients": {
                    "active": list(self.clients.keys()),
                    "states": {k: v.value for k, v in self.client_states.items()}
                }
            }
            
        except Exception as e:
            logger.error(f"获取服务状态失败: {e}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            Dict[str, Any]: 性能指标
        """
        try:
            uptime = time.time() - self.start_time if self.start_time > 0 else 1
            
            return {
                "requests_per_minute": (self.total_requests / uptime) * 60,
                "successful_requests_per_minute": (self.successful_requests / uptime) * 60,
                "average_uptime": uptime,
                "connection_rate": self.total_connections / uptime,
                "active_connection_ratio": self.active_connections / max(self.total_connections, 1),
                "buffer_utilization": self.audio_buffer.get_statistics().get("buffer_utilization", 0) if self.audio_buffer else 0,
                "asr_avg_processing_time": self.whisper_asr.avg_processing_time if self.whisper_asr else 0,
                "tts_avg_processing_time": self.tts_synthesizer.avg_processing_time if self.tts_synthesizer else 0
            }
            
        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return {"error": str(e)}
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop()