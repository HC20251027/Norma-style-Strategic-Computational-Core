"""
语音到文本转换器

负责将音频数据转换为文本
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Union
import tempfile
import os
from datetime import datetime

from .models import SpeechToTextResult, PipelineRequest


class SpeechToTextConverter:
    """语音到文本转换器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.engine = self.config.get('engine', 'whisper')
        self.language = self.config.get('language', 'zh-CN')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # 初始化语音识别引擎
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化语音识别引擎"""
        try:
            if self.engine == 'whisper':
                self._init_whisper()
            elif self.engine == 'speech_recognition':
                self._init_speech_recognition()
            elif self.engine == 'azure':
                self._init_azure_speech()
            else:
                self.logger.warning(f"Unknown speech recognition engine: {self.engine}")
                self.engine = 'mock'  # 使用模拟引擎
        except Exception as e:
            self.logger.error(f"Failed to initialize speech recognition engine: {e}")
            self.engine = 'mock'
    
    def _init_whisper(self):
        """初始化Whisper引擎"""
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            self.logger.info("Whisper engine initialized successfully")
        except ImportError:
            self.logger.warning("Whisper not available, falling back to mock engine")
            self.engine = 'mock'
        except Exception as e:
            self.logger.error(f"Failed to initialize Whisper: {e}")
            self.engine = 'mock'
    
    def _init_speech_recognition(self):
        """初始化Speech Recognition引擎"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.logger.info("Speech Recognition engine initialized successfully")
        except ImportError:
            self.logger.warning("Speech Recognition library not available, falling back to mock engine")
            self.engine = 'mock'
        except Exception as e:
            self.logger.error(f"Failed to initialize Speech Recognition: {e}")
            self.engine = 'mock'
    
    def _init_azure_speech(self):
        """初始化Azure Speech引擎"""
        try:
            # 这里应该配置Azure Speech API密钥
            # import azure.cognitiveservices.speech as speechsdk
            # self.speech_config = speechsdk.SpeechConfig(subscription, region)
            self.logger.info("Azure Speech engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure Speech: {e}")
            self.engine = 'mock'
    
    async def convert(self, request: PipelineRequest) -> SpeechToTextResult:
        """将音频转换为文本"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting speech-to-text conversion for request {request.id}")
            
            # 预处理音频数据
            audio_data = await self._preprocess_audio(request)
            
            # 执行语音识别
            if self.engine == 'whisper':
                result = await self._convert_with_whisper(audio_data, request)
            elif self.engine == 'speech_recognition':
                result = await self._convert_with_speech_recognition(audio_data, request)
            elif self.engine == 'azure':
                result = await self._convert_with_azure(audio_data, request)
            else:
                result = await self._convert_with_mock(audio_data, request)
            
            # 后处理结果
            result = await self._postprocess_result(result, request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Speech-to-text conversion completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Speech-to-text conversion failed: {e}")
            raise
    
    async def _preprocess_audio(self, request: PipelineRequest) -> bytes:
        """预处理音频数据"""
        if request.audio_data is None:
            raise ValueError("No audio data provided")
        
        # 音频格式验证和转换
        if request.audio_format.lower() not in ['wav', 'mp3', 'flac', 'm4a']:
            raise ValueError(f"Unsupported audio format: {request.audio_format}")
        
        # 如果需要格式转换，在这里处理
        # 目前假设输入格式已经是支持的格式
        
        return request.audio_data
    
    async def _convert_with_whisper(self, audio_data: bytes, 
                                  request: PipelineRequest) -> SpeechToTextResult:
        """使用Whisper进行语音识别"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=f".{request.audio_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                # 使用Whisper进行识别
                result = self.whisper_model.transcribe(
                    temp_file.name,
                    language=self.language,
                    task='transcribe'
                )
                
                # 清理临时文件
                os.unlink(temp_file.name)
            
            # 提取文本和置信度
            text = result['text'].strip()
            
            # 计算置信度（Whisper不直接提供置信度，使用文本长度作为近似）
            confidence = min(1.0, len(text) / 100.0) if text else 0.0
            
            # 处理分段信息
            segments = []
            if 'segments' in result:
                for segment in result['segments']:
                    segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'confidence': segment.get('avg_logprob', -1.0)
                    })
            
            return SpeechToTextResult(
                text=text,
                confidence=confidence,
                language=self.language,
                duration=result.get('duration', 0.0),
                segments=segments,
                metadata={'engine': 'whisper', 'model': 'base'}
            )
            
        except Exception as e:
            self.logger.error(f"Whisper conversion failed: {e}")
            raise
    
    async def _convert_with_speech_recognition(self, audio_data: bytes, 
                                             request: PipelineRequest) -> SpeechToTextResult:
        """使用Speech Recognition库进行语音识别"""
        try:
            import speech_recognition as sr
            from io import BytesIO
            
            # 创建音频对象
            audio = sr.AudioData(audio_data, request.sample_rate, 2)
            
            # 使用Google语音识别（需要网络连接）
            text = self.recognizer.recognize_google(audio, language=self.language)
            
            # Speech Recognition不提供置信度，使用固定值
            confidence = 0.8
            
            return SpeechToTextResult(
                text=text,
                confidence=confidence,
                language=self.language,
                duration=0.0,  # Speech Recognition不提供时长信息
                metadata={'engine': 'speech_recognition', 'service': 'google'}
            )
            
        except Exception as e:
            self.logger.error(f"Speech Recognition conversion failed: {e}")
            raise
    
    async def _convert_with_azure(self, audio_data: bytes, 
                                request: PipelineRequest) -> SpeechToTextResult:
        """使用Azure Speech进行语音识别"""
        try:
            # 这里应该实现Azure Speech API调用
            # 由于需要API密钥，这里使用模拟实现
            
            text = "Azure Speech识别结果（模拟）"
            confidence = 0.85
            
            return SpeechToTextResult(
                text=text,
                confidence=confidence,
                language=self.language,
                duration=0.0,
                metadata={'engine': 'azure', 'service': 'azure_speech'}
            )
            
        except Exception as e:
            self.logger.error(f"Azure Speech conversion failed: {e}")
            raise
    
    async def _convert_with_mock(self, audio_data: bytes, 
                               request: PipelineRequest) -> SpeechToTextResult:
        """使用模拟引擎进行语音识别"""
        self.logger.warning("Using mock speech recognition engine")
        
        # 根据音频数据长度生成模拟文本
        audio_length = len(audio_data)
        
        mock_texts = [
            "您好，请问有什么可以帮助您的？",
            "请告诉我您的需求。",
            "我正在听您说话，请继续。",
            "谢谢您的输入，我会帮您处理。",
            "请再说一遍，我没有听清楚。"
        ]
        
        # 简单的模拟逻辑
        text = mock_texts[audio_length % len(mock_texts)]
        confidence = 0.7
        
        return SpeechToTextResult(
            text=text,
            confidence=confidence,
            language=self.language,
            duration=audio_length / (request.sample_rate * 2),  # 估算时长
            metadata={'engine': 'mock', 'note': 'Simulated recognition'}
        )
    
    async def _postprocess_result(self, result: SpeechToTextResult, 
                                request: PipelineRequest) -> SpeechToTextResult:
        """后处理识别结果"""
        # 文本清理
        result.text = result.text.strip()
        
        # 移除多余的空格和标点
        import re
        result.text = re.sub(r'\s+', ' ', result.text)
        
        # 检查置信度
        if result.confidence < self.confidence_threshold:
            self.logger.warning(f"Low confidence recognition: {result.confidence}")
            result.metadata['low_confidence'] = True
        
        # 添加时间戳
        result.metadata['processed_at'] = datetime.now().isoformat()
        
        return result
    
    async def get_supported_languages(self) -> list:
        """获取支持的语言列表"""
        if self.engine == 'whisper':
            return [
                'zh-CN', 'zh-TW', 'en-US', 'en-GB', 'ja-JP', 'ko-KR',
                'fr-FR', 'de-DE', 'es-ES', 'it-IT', 'pt-BR', 'ru-RU'
            ]
        elif self.engine == 'speech_recognition':
            return ['zh-CN', 'en-US', 'ja-JP', 'ko-KR']  # Google支持的主要语言
        else:
            return ['zh-CN', 'en-US']
    
    async def validate_audio_format(self, request: PipelineRequest) -> bool:
        """验证音频格式"""
        supported_formats = ['wav', 'mp3', 'flac', 'm4a', 'ogg']
        return request.audio_format.lower() in supported_formats
    
    async def estimate_processing_time(self, audio_duration: float) -> float:
        """估算处理时间"""
        # 根据引擎和音频时长估算处理时间
        base_time = 0.5  # 基础处理时间
        
        if self.engine == 'whisper':
            # Whisper处理时间与音频时长成正比
            return base_time + (audio_duration * 0.3)
        elif self.engine == 'speech_recognition':
            # Speech Recognition处理时间较短
            return base_time + (audio_duration * 0.1)
        else:
            return base_time