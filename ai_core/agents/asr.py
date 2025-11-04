"""
ASR (自动语音识别) 管理器
支持多种ASR引擎的集成和管理
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import numpy as np
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)


class ASREngine(Enum):
    """支持的ASR引擎类型"""
    WHISPER = "whisper"
    AZURE_SPEECH = "azure_speech"
    GOOGLE_SPEECH = "google_speech"
    OPENAI_WHISPER = "openai_whisper"


@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str
    confidence: float
    language: str
    duration: float
    engine: ASREngine
    timestamp: float
    segments: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


class BaseASREngine(ABC):
    """ASR引擎基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """初始化引擎"""
        pass
    
    @abstractmethod
    async def transcribe_audio(
        self, 
        audio_data: Union[bytes, np.ndarray, str],
        language: str = "zh-CN",
        **kwargs
    ) -> ASRResult:
        """转录音频数据"""
        pass
    
    @abstractmethod
    async def transcribe_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "zh-CN",
        **kwargs
    ) -> AsyncGenerator[ASRResult, None]:
        """流式转录音频"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源"""
        pass


class WhisperASR(BaseASREngine):
    """Whisper ASR引擎实现"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = None
        self.model_size = config.get("model_size", "base")
        self.device = config.get("device", "cpu")
        
    async def initialize(self) -> None:
        """初始化Whisper模型"""
        try:
            import whisper
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.is_initialized = True
            logger.info(f"Whisper ASR引擎初始化完成，模型大小: {self.model_size}")
        except ImportError:
            raise ImportError("请安装openai-whisper: pip install openai-whisper")
        except Exception as e:
            logger.error(f"Whisper初始化失败: {e}")
            raise
    
    async def transcribe_audio(
        self, 
        audio_data: Union[bytes, np.ndarray, str],
        language: str = "zh-CN",
        **kwargs
    ) -> ASRResult:
        """转录音频数据"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            import whisper
            import tempfile
            
            # 处理不同类型的输入
            if isinstance(audio_data, str):
                # 文件路径
                result = self.model.transcribe(audio_data, language=language, **kwargs)
            elif isinstance(audio_data, bytes):
                # 字节数据
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file.flush()
                    result = self.model.transcribe(tmp_file.name, language=language, **kwargs)
                    Path(tmp_file.name).unlink()
            else:
                # numpy数组
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_data, 16000)
                    result = self.model.transcribe(tmp_file.name, language=language, **kwargs)
                    Path(tmp_file.name).unlink()
            
            return ASRResult(
                text=result["text"].strip(),
                confidence=result.get("confidence", 0.0),
                language=result.get("language", language),
                duration=result.get("duration", 0.0),
                engine=ASREngine.WHISPER,
                timestamp=asyncio.get_event_loop().time(),
                segments=result.get("segments", []),
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Whisper转录失败: {e}")
            raise
    
    async def transcribe_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "zh-CN",
        **kwargs
    ) -> AsyncGenerator[ASRResult, None]:
        """流式转录音频"""
        # Whisper本身不直接支持流式，这里实现一个简单的分块处理
        buffer_size = int(kwargs.get("buffer_size", 30))  # 30秒缓冲
        buffer_duration = 0
        audio_buffer = []
        
        async for chunk in audio_stream:
            audio_buffer.append(chunk)
            buffer_duration += len(chunk) / 16000 / 2  # 假设16kHz 16bit
            
            if buffer_duration >= buffer_size:
                # 处理缓冲的音频
                combined_audio = b''.join(audio_buffer)
                try:
                    result = await self.transcribe_audio(combined_audio, language, **kwargs)
                    yield result
                except Exception as e:
                    logger.error(f"流式转录错误: {e}")
                
                # 清空缓冲区
                audio_buffer = []
                buffer_duration = 0
        
        # 处理剩余音频
        if audio_buffer:
            combined_audio = b''.join(audio_buffer)
            try:
                result = await self.transcribe_audio(combined_audio, language, **kwargs)
                yield result
            except Exception as e:
                logger.error(f"流式转录错误: {e}")
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.model = None
        self.is_initialized = False


class AzureASR(BaseASREngine):
    """Azure Speech ASR引擎实现"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.speech_config = None
        self.speech_recognizer = None
        
    async def initialize(self) -> None:
        """初始化Azure Speech SDK"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # 配置Azure Speech
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.config.get("subscription_key"),
                region=self.config.get("region")
            )
            self.speech_config.speech_recognition_language = self.config.get("language", "zh-CN")
            
            self.is_initialized = True
            logger.info("Azure Speech ASR引擎初始化完成")
            
        except ImportError:
            raise ImportError("请安装Azure Speech SDK: pip install azure-cognitiveservices-speech")
        except Exception as e:
            logger.error(f"Azure Speech初始化失败: {e}")
            raise
    
    async def transcribe_audio(
        self, 
        audio_data: Union[bytes, np.ndarray, str],
        language: str = "zh-CN",
        **kwargs
    ) -> ASRResult:
        """转录音频数据"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            import azure.cognitiveservices.speech as speechsdk
            import tempfile
            
            # 创建音频配置
            if isinstance(audio_data, str):
                # 文件路径
                audio_config = speechsdk.audio.AudioConfig(filename=audio_data)
            elif isinstance(audio_data, bytes):
                # 字节数据
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file.flush()
                    audio_config = speechsdk.audio.AudioConfig(filename=tmp_file.name)
                    Path(tmp_file.name).unlink()
            else:
                # numpy数组
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_data, 16000)
                    audio_config = speechsdk.audio.AudioConfig(filename=tmp_file.name)
                    Path(tmp_file.name).unlink()
            
            # 创建识别器
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # 执行识别
            result = speech_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return ASRResult(
                    text=result.text,
                    confidence=1.0,  # Azure不直接提供置信度
                    language=language,
                    duration=0.0,
                    engine=ASREngine.AZURE_SPEECH,
                    timestamp=asyncio.get_event_loop().time(),
                    metadata={"json": result.json}
                )
            else:
                raise Exception(f"识别失败: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure Speech转录失败: {e}")
            raise
    
    async def transcribe_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "zh-CN",
        **kwargs
    ) -> AsyncGenerator[ASRResult, None]:
        """流式转录音频"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # 创建流式音频配置
            push_stream = speechsdk.audio.PullAudioOutputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            
            # 创建识别器
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # 处理音频流
            async for chunk in audio_stream:
                # 将数据推送到Azure流
                chunk_array = np.frombuffer(chunk, dtype=np.int16)
                push_stream.write(chunk_array.tobytes())
            
            # 停止流
            push_stream.close()
            
            # 执行识别
            result = speech_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                yield ASRResult(
                    text=result.text,
                    confidence=1.0,
                    language=language,
                    duration=0.0,
                    engine=ASREngine.AZURE_SPEECH,
                    timestamp=asyncio.get_event_loop().time(),
                    metadata={"json": result.json}
                )
                
        except Exception as e:
            logger.error(f"Azure Speech流式转录失败: {e}")
            raise
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.speech_config = None
        self.speech_recognizer = None
        self.is_initialized = False


class ASRManager:
    """ASR管理器 - 统一管理多个ASR引擎"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.engines: Dict[ASREngine, BaseASREngine] = {}
        self.default_engine = ASREngine(config.get("default_engine", ASREngine.WHISPER))
        self._initialize_engines()
        
    def _initialize_engines(self) -> None:
        """初始化ASR引擎"""
        engine_configs = self.config.get("engines", {})
        
        # 初始化Whisper
        if ASREngine.WHISPER in engine_configs:
            self.engines[ASREngine.WHISPER] = WhisperASR(engine_configs[ASREngine.WHISPER])
        
        # 初始化Azure Speech
        if ASREngine.AZURE_SPEECH in engine_configs:
            self.engines[ASREngine.AZURE_SPEECH] = AzureASR(engine_configs[ASREngine.AZURE_SPEECH])
        
        logger.info(f"ASR管理器初始化完成，支持引擎: {list(self.engines.keys())}")
    
    async def transcribe(
        self,
        audio_data: Union[bytes, np.ndarray, str],
        engine: Optional[ASREngine] = None,
        language: str = "zh-CN",
        **kwargs
    ) -> ASRResult:
        """转录音频"""
        if engine is None:
            engine = self.default_engine
            
        if engine not in self.engines:
            raise ValueError(f"ASR引擎 {engine} 未配置")
        
        asr_engine = self.engines[engine]
        return await asr_engine.transcribe_audio(audio_data, language, **kwargs)
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        engine: Optional[ASREngine] = None,
        language: str = "zh-CN",
        **kwargs
    ) -> AsyncGenerator[ASRResult, None]:
        """流式转录音频"""
        if engine is None:
            engine = self.default_engine
            
        if engine not in self.engines:
            raise ValueError(f"ASR引擎 {engine} 未配置")
        
        asr_engine = self.engines[engine]
        async for result in asr_engine.transcribe_stream(audio_stream, language, **kwargs):
            yield result
    
    async def get_supported_languages(self, engine: ASREngine) -> List[str]:
        """获取支持的语言列表"""
        language_mappings = {
            ASREngine.WHISPER: [
                "zh-CN", "en-US", "ja-JP", "ko-KR", 
                "fr-FR", "de-DE", "es-ES", "it-IT"
            ],
            ASREngine.AZURE_SPEECH: [
                "zh-CN", "en-US", "ja-JP", "ko-KR",
                "fr-FR", "de-DE", "es-ES", "it-IT"
            ]
        }
        return language_mappings.get(engine, ["zh-CN", "en-US"])
    
    async def cleanup(self) -> None:
        """清理所有引擎资源"""
        for engine in self.engines.values():
            await engine.cleanup()
        self.engines.clear()