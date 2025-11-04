"""
文本到语音转换器

负责将文本转换为语音音频
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime
import tempfile
import os
import io

from .models import TextToSpeechResult, PipelineRequest


class TextToSpeechConverter:
    """文本到语音转换器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.engine = self.config.get('engine', 'pyttsx3')
        self.voice = self.config.get('voice', 'zh')
        self.rate = self.config.get('rate', 150)
        self.volume = self.config.get('volume', 0.8)
        
        # 初始化语音合成引擎
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化语音合成引擎"""
        try:
            if self.engine == 'pyttsx3':
                self._init_pyttsx3()
            elif self.engine == 'gTTS':
                self._init_gtts()
            elif self.engine == 'azure':
                self._init_azure_speech()
            elif self.engine == 'espeak':
                self._init_espeak()
            else:
                self.logger.warning(f"Unknown TTS engine: {self.engine}")
                self.engine = 'mock'  # 使用模拟引擎
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = 'mock'
    
    def _init_pyttsx3(self):
        """初始化pyttsx3引擎"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            
            # 设置语音参数
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # 选择中文语音
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.setProperty('rate', self.rate)
            self.tts_engine.setProperty('volume', self.volume)
            
            self.logger.info("pyttsx3 engine initialized successfully")
        except ImportError:
            self.logger.warning("pyttsx3 not available, falling back to mock engine")
            self.engine = 'mock'
        except Exception as e:
            self.logger.error(f"Failed to initialize pyttsx3: {e}")
            self.engine = 'mock'
    
    def _init_gtts(self):
        """初始化Google Text-to-Speech引擎"""
        try:
            from gtts import gTTS
            self.gtts = gTTS
            self.logger.info("gTTS engine initialized successfully")
        except ImportError:
            self.logger.warning("gTTS not available, falling back to mock engine")
            self.engine = 'mock'
        except Exception as e:
            self.logger.error(f"Failed to initialize gTTS: {e}")
            self.engine = 'mock'
    
    def _init_azure_speech(self):
        """初始化Azure Speech引擎"""
        try:
            # 这里应该配置Azure Speech API密钥
            # import azure.cognitiveservices.speech as speechsdk
            # self.speech_config = speechsdk.SpeechConfig(subscription, region)
            self.logger.info("Azure Speech TTS engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure Speech TTS: {e}")
            self.engine = 'mock'
    
    def _init_espeak(self):
        """初始化espeak引擎"""
        try:
            import subprocess
            # 检查espeak是否可用
            subprocess.run(['espeak', '--version'], capture_output=True, check=True)
            self.logger.info("espeak engine initialized successfully")
        except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("espeak not available, falling back to mock engine")
            self.engine = 'mock'
    
    async def synthesize(self, text: str, request: PipelineRequest) -> TextToSpeechResult:
        """将文本转换为语音"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting text-to-speech synthesis for request {request.id}")
            
            if not text.strip():
                self.logger.warning("Empty text provided for synthesis")
                return await self._create_empty_result()
            
            # 文本预处理
            processed_text = await self._preprocess_text(text)
            
            # 执行语音合成
            if self.engine == 'pyttsx3':
                audio_data = await self._synthesize_with_pyttsx3(processed_text, request)
            elif self.engine == 'gTTS':
                audio_data = await self._synthesize_with_gtts(processed_text, request)
            elif self.engine == 'azure':
                audio_data = await self._synthesize_with_azure(processed_text, request)
            elif self.engine == 'espeak':
                audio_data = await self._synthesize_with_espeak(processed_text, request)
            else:
                audio_data = await self._synthesize_with_mock(processed_text, request)
            
            # 后处理音频数据
            audio_data = await self._postprocess_audio(audio_data, request)
            
            # 计算时长
            duration = await self._estimate_duration(processed_text, audio_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Text-to-speech synthesis completed in {processing_time:.2f}s")
            
            return TextToSpeechResult(
                audio_data=audio_data,
                format=request.output_config.get('format', 'wav'),
                duration=duration,
                metadata={
                    'engine': self.engine,
                    'processed_text': processed_text,
                    'processing_time': processing_time
                }
            )
            
        except Exception as e:
            self.logger.error(f"Text-to-speech synthesis failed: {e}")
            raise
    
    async def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 移除多余的空格
        text = ' '.join(text.split())
        
        # 替换特殊符号
        replacements = {
            '%': '百分之',
            '&': '和',
            '@': '在',
            '#': '井号',
            '*': '星号',
            '+': '加号',
            '=': '等于'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 处理数字
        text = await self._process_numbers(text)
        
        # 处理缩写
        abbreviations = {
            'AI': '人工智能',
            'API': '应用程序接口',
            'CPU': '中央处理器',
            'GPU': '图形处理器',
            'RAM': '内存',
            'USB': '通用串行总线',
            'WiFi': '无线网络',
            'OK': '好的'
        }
        
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        return text
    
    async def _process_numbers(self, text: str) -> str:
        """处理文本中的数字"""
        import re
        
        # 替换数字为中文读法（简化版本）
        number_words = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
            '10': '十', '100': '一百', '1000': '一千', '10000': '一万'
        }
        
        for num, word in number_words.items():
            text = text.replace(num, word)
        
        return text
    
    async def _synthesize_with_pyttsx3(self, text: str, request: PipelineRequest) -> bytes:
        """使用pyttsx3进行语音合成"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 保存到文件
            self.tts_engine.save_to_file(text, temp_path)
            self.tts_engine.runAndWait()
            
            # 读取生成的音频文件
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # 清理临时文件
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"pyttsx3 synthesis failed: {e}")
            raise
    
    async def _synthesize_with_gtts(self, text: str, request: PipelineRequest) -> bytes:
        """使用Google TTS进行语音合成"""
        try:
            # 确定语言
            lang = request.language if request.language else 'zh-cn'
            
            # 创建TTS对象
            tts = self.gtts(text=text, lang=lang, slow=False)
            
            # 保存到内存中的字节流
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.read()
            
        except Exception as e:
            self.logger.error(f"gTTS synthesis failed: {e}")
            raise
    
    async def _synthesize_with_azure(self, text: str, request: PipelineRequest) -> bytes:
        """使用Azure Speech进行语音合成"""
        try:
            # 这里应该实现Azure Speech TTS API调用
            # 由于需要API密钥，这里使用模拟实现
            
            # 创建模拟音频数据（简单的音频头）
            audio_data = self._create_mock_audio_data()
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Azure Speech TTS synthesis failed: {e}")
            raise
    
    async def _synthesize_with_espeak(self, text: str, request: PipelineRequest) -> bytes:
        """使用espeak进行语音合成"""
        try:
            import subprocess
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 使用espeak命令
            cmd = [
                'espeak',
                '-v', 'zh',  # 中文语音
                '-s', str(self.rate),  # 语速
                '-a', str(int(self.volume * 100)),  # 音量
                '-w', temp_path,  # 输出文件
                text
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 读取生成的音频文件
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            
            # 清理临时文件
            os.unlink(temp_path)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"espeak synthesis failed: {e}")
            raise
    
    async def _synthesize_with_mock(self, text: str, request: PipelineRequest) -> bytes:
        """使用模拟引擎进行语音合成"""
        self.logger.warning("Using mock TTS engine")
        
        # 创建模拟音频数据
        audio_data = self._create_mock_audio_data()
        return audio_data
    
    def _create_mock_audio_data(self) -> bytes:
        """创建模拟音频数据"""
        # 创建一个简单的WAV文件头（44字节）
        import struct
        
        # WAV文件头
        riff_header = b'RIFF'
        file_size = struct.pack('<I', 44)  # 文件大小
        wave_header = b'WAVE'
        fmt_header = b'fmt '
        fmt_chunk_size = struct.pack('<I', 16)  # fmt块大小
        audio_format = struct.pack('<H', 1)  # PCM格式
        num_channels = struct.pack('<H', 1)  # 单声道
        sample_rate = struct.pack('<I', 16000)  # 采样率
        byte_rate = struct.pack('<I', 32000)  # 字节率
        block_align = struct.pack('<H', 2)  # 块对齐
        bits_per_sample = struct.pack('<H', 16)  # 位深度
        data_header = b'data'
        data_size = struct.pack('<I', 0)  # 数据大小
        
        wav_header = (
            riff_header + file_size + wave_header +
            fmt_header + fmt_chunk_size + audio_format +
            num_channels + sample_rate + byte_rate +
            block_align + bits_per_sample + data_header + data_size
        )
        
        return wav_header
    
    async def _postprocess_audio(self, audio_data: bytes, request: PipelineRequest) -> bytes:
        """后处理音频数据"""
        # 格式转换（如果需要）
        output_format = request.output_config.get('format', 'wav')
        
        if output_format.lower() == 'mp3':
            # 这里可以添加MP3转换逻辑
            # 目前简单返回原始数据
            pass
        
        return audio_data
    
    async def _estimate_duration(self, text: str, audio_data: bytes) -> float:
        """估算音频时长"""
        # 简单的时长估算：基于文本长度和平均语速
        words_per_second = self.rate / 60.0  # 转换为每秒字数
        estimated_duration = len(text) / words_per_second if words_per_second > 0 else 3.0
        
        # 限制在合理范围内
        return max(0.5, min(30.0, estimated_duration))
    
    async def _create_empty_result(self) -> TextToSpeechResult:
        """创建空结果"""
        return TextToSpeechResult(
            audio_data=b'',
            format='wav',
            duration=0.0,
            metadata={'note': 'Empty text provided'}
        )
    
    async def get_available_voices(self) -> List[Dict[str, str]]:
        """获取可用语音列表"""
        voices = []
        
        if self.engine == 'pyttsx3':
            try:
                voices_list = self.tts_engine.getProperty('voices')
                for voice in voices_list:
                    voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'languages': getattr(voice, 'languages', []),
                        'engine': 'pyttsx3'
                    })
            except Exception as e:
                self.logger.error(f"Failed to get voices from pyttsx3: {e}")
        
        elif self.engine == 'gTTS':
            # gTTS支持的语言
            gtts_languages = [
                {'code': 'zh-cn', 'name': '中文（简体）', 'engine': 'gTTS'},
                {'code': 'zh-tw', 'name': '中文（繁体）', 'engine': 'gTTS'},
                {'code': 'en', 'name': '英语', 'engine': 'gTTS'},
                {'code': 'ja', 'name': '日语', 'engine': 'gTTS'},
                {'code': 'ko', 'name': '韩语', 'engine': 'gTTS'},
                {'code': 'fr', 'name': '法语', 'engine': 'gTTS'},
                {'code': 'de', 'name': '德语', 'engine': 'gTTS'},
                {'code': 'es', 'name': '西班牙语', 'engine': 'gTTS'}
            ]
            voices.extend(gtts_languages)
        
        return voices
    
    async def test_voice(self, text: str = "测试语音") -> bool:
        """测试语音合成"""
        try:
            # 创建测试请求
            test_request = PipelineRequest(
                audio_data=None,
                text=text,
                language='zh-CN'
            )
            
            result = await self.synthesize(text, test_request)
            return len(result.audio_data) > 0
            
        except Exception as e:
            self.logger.error(f"Voice test failed: {e}")
            return False
    
    async def validate_text(self, text: str) -> Dict[str, Any]:
        """验证文本"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # 检查文本长度
        if len(text) == 0:
            validation_result['errors'].append("文本不能为空")
            validation_result['valid'] = False
        elif len(text) > 1000:
            validation_result['warnings'].append("文本较长，可能影响合成效果")
            validation_result['suggestions'].append("建议将长文本分段处理")
        
        # 检查特殊字符
        special_chars = ['@', '#', '$', '%', '^', '&', '*']
        found_special = [char for char in special_chars if char in text]
        if found_special:
            validation_result['warnings'].append(f"包含特殊字符：{', '.join(found_special)}")
            validation_result['suggestions'].append("特殊字符可能影响语音合成效果")
        
        # 检查数字
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            validation_result['suggestions'].append("数字将被转换为中文读法")
        
        return validation_result