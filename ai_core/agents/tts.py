"""
TTS (文本转语音) 管理器
支持多种TTS引擎的集成和管理
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, AsyncGenerator, Iterator
from dataclasses import dataclass
from enum import Enum
import io
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    """支持的TTS引擎类型"""
    EDGE_TTS = "edge_tts"
    AZURE_SPEECH = "azure_speech"
    GOOGLE_TTS = "google_tts"
    PYTORCH_TTS = "pytorch_tts"


@dataclass
class TTSResult:
    """TTS合成结果"""
    audio_data: bytes
    format: str
    sample_rate: int
    duration: float
    engine: TTSEngine
    voice_id: str
    language: str
    timestamp: float
    metadata: Optional[Dict] = None


@dataclass
class VoiceInfo:
    """语音信息"""
    voice_id: str
    name: str
    language: str
    gender: str
    engine: TTSEngine
    description: Optional[str] = None


class BaseTTSEngine(ABC):
    """TTS引擎基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_initialized = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """初始化引擎"""
        pass
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "zh-CN",
        **kwargs
    ) -> TTSResult:
        """合成语音"""
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        language: str = "zh-CN",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        pass
    
    @abstractmethod
    async def get_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """获取可用语音列表"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源"""
        pass


class EdgeTTS(BaseTTSEngine):
    """Edge TTS引擎实现"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.communicator = None
        self.voices = {}
        
    async def initialize(self) -> None:
        """初始化Edge TTS"""
        try:
            import edge_tts
            
            # 获取可用语音列表
            self.voices = {
                voice["Name"]: voice 
                for voice in await edge_tts.list_voices()
            }
            
            self.is_initialized = True
            logger.info(f"Edge TTS引擎初始化完成，可用语音: {len(self.voices)}个")
            
        except ImportError:
            raise ImportError("请安装edge-tts: pip install edge-tts")
        except Exception as e:
            logger.error(f"Edge TTS初始化失败: {e}")
            raise
    
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "zh-CN",
        **kwargs
    ) -> TTSResult:
        """合成语音"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            import edge_tts
            import tempfile
            
            # 配置语音参数
            rate = kwargs.get("rate", "+0%")
            volume = kwargs.get("volume", "+0%")
            pitch = kwargs.get("pitch", "+0%")
            
            # 创建TTS通信器
            communicate = edge_tts.Communicate(text, voice_id)
            
            # 添加SSML样式
            if rate != "+0%" or volume != "+0%" or pitch != "+0%":
                ssml_text = f"""
                <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
                    <voice name="{voice_id}">
                        <prosody rate="{rate}" volume="{volume}" pitch="{pitch}">
                            {text}
                        </prosody>
                    </voice>
                </speak>
                """
                communicate = edge_tts.Communicate(ssml_text, voice_id)
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                await communicate.save(tmp_file.name)
                
                # 读取音频数据
                with open(tmp_file.name, "rb") as f:
                    audio_data = f.read()
                
                # 获取音频信息
                audio_array, sample_rate = sf.read(tmp_file.name)
                duration = len(audio_array) / sample_rate
                
                # 清理临时文件
                import os
                os.unlink(tmp_file.name)
            
            return TTSResult(
                audio_data=audio_data,
                format="wav",
                sample_rate=sample_rate,
                duration=duration,
                engine=TTSEngine.EDGE_TTS,
                voice_id=voice_id,
                language=language,
                timestamp=asyncio.get_event_loop().time(),
                metadata={
                    "rate": rate,
                    "volume": volume,
                    "pitch": pitch
                }
            )
            
        except Exception as e:
            logger.error(f"Edge TTS合成失败: {e}")
            raise
    
    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        language: str = "zh-CN",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        try:
            import edge_tts
            
            # 配置语音参数
            rate = kwargs.get("rate", "+0%")
            volume = kwargs.get("volume", "+0%")
            pitch = kwargs.get("pitch", "+0%")
            
            # 创建TTS通信器
            ssml_text = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
                <voice name="{voice_id}">
                    <prosody rate="{rate}" volume="{volume}" pitch="{pitch}">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            communicate = edge_tts.Communicate(ssml_text, voice_id)
            
            # 流式生成音频
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                    
        except Exception as e:
            logger.error(f"Edge TTS流式合成失败: {e}")
            raise
    
    async def get_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """获取可用语音列表"""
        if not self.is_initialized:
            await self.initialize()
            
        voices = []
        for voice_name, voice_info in self.voices.items():
            if language and not voice_info["Locale"].startswith(language.split("-")[0]):
                continue
                
            voices.append(VoiceInfo(
                voice_id=voice_name,
                name=voice_info["FriendlyName"],
                language=voice_info["Locale"],
                gender=voice_info["Gender"],
                engine=TTSEngine.EDGE_TTS,
                description=voice_info.get("Description", "")
            ))
        
        return voices
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.communicator = None
        self.voices = {}
        self.is_initialized = False


class AzureTTS(BaseTTSEngine):
    """Azure Speech TTS引擎实现"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.speech_config = None
        self.synthesizer = None
        
    async def initialize(self) -> None:
        """初始化Azure Speech SDK"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # 配置Azure Speech
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.config.get("subscription_key"),
                region=self.config.get("region")
            )
            
            self.is_initialized = True
            logger.info("Azure Speech TTS引擎初始化完成")
            
        except ImportError:
            raise ImportError("请安装Azure Speech SDK: pip install azure-cognitiveservices-speech")
        except Exception as e:
            logger.error(f"Azure Speech TTS初始化失败: {e}")
            raise
    
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "zh-CN",
        **kwargs
    ) -> TTSResult:
        """合成语音"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # 配置语音参数
            rate = kwargs.get("rate", "0%")
            volume = kwargs.get("volume", "0%")
            pitch = kwargs.get("pitch", "0%")
            
            # 设置语音
            self.speech_config.speech_synthesis_voice_name = voice_id
            
            # 创建SSML
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
                <voice name="{voice_id}">
                    <prosody rate="{rate}" volume="{volume}" pitch="{pitch}">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            # 创建合成器
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            # 执行合成
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                
                # 计算音频信息
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                sample_rate = 16000  # Azure默认采样率
                duration = len(audio_array) / sample_rate
                
                return TTSResult(
                    audio_data=audio_data,
                    format="wav",
                    sample_rate=sample_rate,
                    duration=duration,
                    engine=TTSEngine.AZURE_SPEECH,
                    voice_id=voice_id,
                    language=language,
                    timestamp=asyncio.get_event_loop().time(),
                    metadata={
                        "rate": rate,
                        "volume": volume,
                        "pitch": pitch
                    }
                )
            else:
                raise Exception(f"合成失败: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure Speech TTS合成失败: {e}")
            raise
    
    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        language: str = "zh-CN",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # 设置语音
            self.speech_config.speech_synthesis_voice_name = voice_id
            
            # 配置语音参数
            rate = kwargs.get("rate", "0%")
            volume = kwargs.get("volume", "0%")
            pitch = kwargs.get("pitch", "0%")
            
            # 创建SSML
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
                <voice name="{voice_id}">
                    <prosody rate="{rate}" volume="{volume}" pitch="{pitch}">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            # 创建流式合成器
            stream = speechsdk.audio.PullAudioOutputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=stream)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # 执行流式合成
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # 流式返回音频数据
                chunk_size = 4096
                audio_data = result.audio_data
                
                for i in range(0, len(audio_data), chunk_size):
                    yield audio_data[i:i + chunk_size]
            else:
                raise Exception(f"流式合成失败: {result.reason}")
                
        except Exception as e:
            logger.error(f"Azure Speech TTS流式合成失败: {e}")
            raise
    
    async def get_voices(self, language: Optional[str] = None) -> List[VoiceInfo]:
        """获取可用语音列表"""
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # 获取语音列表
            voices = speechsdk.SpeechSynthesisVoiceList.from_speech_config(self.speech_config)
            voice_list = voices.get_voices()
            
            result_voices = []
            for voice in voice_list:
                if language and not voice.locale.startswith(language.split("-")[0]):
                    continue
                    
                result_voices.append(VoiceInfo(
                    voice_id=voice.short_name,
                    name=voice.friendly_name,
                    language=voice.locale,
                    gender="Female" if voice.gender == "Female" else "Male",
                    engine=TTSEngine.AZURE_SPEECH,
                    description=voice.description
                ))
            
            return result_voices
            
        except Exception as e:
            logger.error(f"获取Azure TTS语音列表失败: {e}")
            return []
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.speech_config = None
        self.synthesizer = None
        self.is_initialized = False


class TTSManager:
    """TTS管理器 - 统一管理多个TTS引擎"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.engines: Dict[TTSEngine, BaseTTSEngine] = {}
        self.default_engine = TTSEngine(config.get("default_engine", TTSEngine.EDGE_TTS))
        self._initialize_engines()
        
    def _initialize_engines(self) -> None:
        """初始化TTS引擎"""
        engine_configs = self.config.get("engines", {})
        
        # 初始化Edge TTS
        if TTSEngine.EDGE_TTS in engine_configs:
            self.engines[TTSEngine.EDGE_TTS] = EdgeTTS(engine_configs[TTSEngine.EDGE_TTS])
        
        # 初始化Azure Speech TTS
        if TTSEngine.AZURE_SPEECH in engine_configs:
            self.engines[TTSEngine.AZURE_SPEECH] = AzureTTS(engine_configs[TTSEngine.AZURE_SPEECH])
        
        logger.info(f"TTS管理器初始化完成，支持引擎: {list(self.engines.keys())}")
    
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        engine: Optional[TTSEngine] = None,
        language: str = "zh-CN",
        **kwargs
    ) -> TTSResult:
        """合成语音"""
        if engine is None:
            engine = self.default_engine
            
        if engine not in self.engines:
            raise ValueError(f"TTS引擎 {engine} 未配置")
        
        tts_engine = self.engines[engine]
        return await tts_engine.synthesize(text, voice_id, language, **kwargs)
    
    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        engine: Optional[TTSEngine] = None,
        language: str = "zh-CN",
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        if engine is None:
            engine = self.default_engine
            
        if engine not in self.engines:
            raise ValueError(f"TTS引擎 {engine} 未配置")
        
        tts_engine = self.engines[engine]
        async for chunk in tts_engine.synthesize_stream(text, voice_id, language, **kwargs):
            yield chunk
    
    async def get_voices(
        self,
        engine: Optional[TTSEngine] = None,
        language: Optional[str] = None
    ) -> List[VoiceInfo]:
        """获取可用语音列表"""
        if engine is None:
            engine = self.default_engine
            
        if engine not in self.engines:
            raise ValueError(f"TTS引擎 {engine} 未配置")
        
        tts_engine = self.engines[engine]
        return await tts_engine.get_voices(language)
    
    async def get_all_voices(
        self,
        language: Optional[str] = None
    ) -> Dict[TTSEngine, List[VoiceInfo]]:
        """获取所有引擎的语音列表"""
        all_voices = {}
        for engine, tts_engine in self.engines.items():
            try:
                voices = await tts_engine.get_voices(language)
                all_voices[engine] = voices
            except Exception as e:
                logger.error(f"获取{engine}语音列表失败: {e}")
                all_voices[engine] = []
        return all_voices
    
    async def cleanup(self) -> None:
        """清理所有引擎资源"""
        for engine in self.engines.values():
            await engine.cleanup()
        self.engines.clear()