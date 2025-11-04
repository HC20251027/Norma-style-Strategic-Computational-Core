"""
TTS语音合成模块

提供多种TTS引擎的语音合成功能
"""

import io
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Any, List, Union
import logging
import time
from dataclasses import dataclass
import tempfile
import os

# TTS引擎导入
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TTSResult:
    """TTS合成结果"""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    text: str
    voice: str
    engine: str
    processing_time: float

class TTSSynthesizer:
    """TTS语音合成器"""
    
    def __init__(self,
                 engine: str = "edge",
                 voice: str = "zh-CN-XiaoxiaoNeural",
                 rate: str = "+0%",
                 volume: str = "+0%",
                 pitch: str = "+0Hz",
                 format: str = "wav"):
        """
        初始化TTS合成器
        
        Args:
            engine: TTS引擎 (edge, coqui, pyttsx3)
            voice: 语音名称
            rate: 语速 (+/-百分比)
            volume: 音量 (+/-百分比)
            pitch: 音调 (+/-Hz)
            format: 输出格式
        """
        self.engine = engine
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.format = format
        
        # 引擎实例
        self.tts_engine = None
        self.coqui_model = None
        
        # 统计信息
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        
        # 支持的引擎
        self.supported_engines = []
        if EDGE_TTS_AVAILABLE:
            self.supported_engines.append("edge")
        if COQUI_TTS_AVAILABLE:
            self.supported_engines.append("coqui")
        if PYTTSX3_AVAILABLE:
            self.supported_engines.append("pyttsx3")
        
        logger.info(f"TTS合成器初始化: engine={engine}, voice={voice}")
        logger.info(f"支持的引擎: {self.supported_engines}")
    
    def initialize(self) -> bool:
        """
        初始化TTS引擎
        
        Returns:
            bool: 是否成功初始化
        """
        try:
            if self.engine not in self.supported_engines:
                logger.error(f"不支持的TTS引擎: {self.engine}")
                return False
            
            if self.engine == "edge":
                return self._initialize_edge_tts()
            elif self.engine == "coqui":
                return self._initialize_coqui_tts()
            elif self.engine == "pyttsx3":
                return self._initialize_pyttsx3()
            else:
                logger.error(f"未实现的TTS引擎: {self.engine}")
                return False
                
        except Exception as e:
            logger.error(f"初始化TTS引擎失败: {e}")
            return False
    
    def synthesize(self, 
                   text: str,
                   voice: Optional[str] = None,
                   **kwargs) -> Optional[TTSResult]:
        """
        语音合成
        
        Args:
            text: 要合成的文本
            voice: 语音名称，None则使用初始化时的语音
            **kwargs: 其他参数
            
        Returns:
            Optional[TTSResult]: 合成结果
        """
        if self.tts_engine is None and self.coqui_model is None:
            logger.error("TTS引擎未初始化")
            return None
        
        try:
            start_time = time.time()
            
            # 使用指定语音或默认语音
            synth_voice = voice or self.voice
            
            logger.info(f"开始语音合成: '{text}', 语音={synth_voice}")
            
            if self.engine == "edge":
                result = self._synthesize_edge(text, synth_voice, **kwargs)
            elif self.engine == "coqui":
                result = self._synthesize_coqui(text, synth_voice, **kwargs)
            elif self.engine == "pyttsx3":
                result = self._synthesize_pyttsx3(text, synth_voice, **kwargs)
            else:
                logger.error(f"未实现的TTS引擎: {self.engine}")
                return None
            
            if result:
                processing_time = time.time() - start_time
                
                # 更新统计信息
                self.total_requests += 1
                self.total_processing_time += processing_time
                self.avg_processing_time = self.total_processing_time / self.total_requests
                
                logger.info(f"语音合成完成，耗时: {processing_time:.2f}秒，音频时长: {result.duration:.2f}秒")
                
                return result
            else:
                logger.error("语音合成失败")
                return None
                
        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            return None
    
    def _initialize_edge_tts(self) -> bool:
        """初始化Edge TTS"""
        try:
            # Edge TTS不需要预初始化
            logger.info("Edge TTS初始化成功")
            return True
        except Exception as e:
            logger.error(f"Edge TTS初始化失败: {e}")
            return False
    
    def _synthesize_edge(self, text: str, voice: str, **kwargs) -> Optional[TTSResult]:
        """使用Edge TTS合成"""
        try:
            import asyncio
            import edge_tts
            
            # 创建通信器
            communicate = edge_tts.Communicate(text, voice)
            
            # 合成音频
            audio_data = b""
            for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if not audio_data:
                return None
            
            # 转换为numpy数组
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            
            return TTSResult(
                audio_data=audio_array,
                sample_rate=sample_rate,
                duration=len(audio_array) / sample_rate,
                text=text,
                voice=voice,
                engine=self.engine,
                processing_time=0.0  # 将在外层计算
            )
            
        except Exception as e:
            logger.error(f"Edge TTS合成失败: {e}")
            return None
    
    def _initialize_coqui_tts(self) -> bool:
        """初始化Coqui TTS"""
        try:
            # 使用预训练的中文TTS模型
            model_name = "tts_models/multilingual/multi-dataset/your_tts"
            self.coqui_model = CoquiTTS(model_name=model_name)
            logger.info("Coqui TTS初始化成功")
            return True
        except Exception as e:
            logger.error(f"Coqui TTS初始化失败: {e}")
            return False
    
    def _synthesize_coqui(self, text: str, voice: str, **kwargs) -> Optional[TTSResult]:
        """使用Coqui TTS合成"""
        try:
            # 合成音频
            wav = self.coqui_model.tts(text=text, speaker=voice)
            
            # 转换为numpy数组
            audio_array = np.array(wav)
            sample_rate = 22050  # Coqui TTS默认采样率
            
            return TTSResult(
                audio_data=audio_array,
                sample_rate=sample_rate,
                duration=len(audio_array) / sample_rate,
                text=text,
                voice=voice,
                engine=self.engine,
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Coqui TTS合成失败: {e}")
            return None
    
    def _initialize_pyttsx3(self) -> bool:
        """初始化pyttsx3"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # 设置语音参数
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # 尝试找到中文语音
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            logger.info("pyttsx3初始化成功")
            return True
        except Exception as e:
            logger.error(f"pyttsx3初始化失败: {e}")
            return False
    
    def _synthesize_pyttsx3(self, text: str, voice: str, **kwargs) -> Optional[TTSResult]:
        """使用pyttsx3合成"""
        try:
            # 设置语音参数
            if voice:
                self.tts_engine.setProperty('voice', voice)
            
            # 设置语速和音量
            if self.rate != "+0%":
                rate_value = int(self.rate.replace('%', ''))
                if rate_value != 0:
                    current_rate = self.tts_engine.getProperty('rate')
                    self.tts_engine.setProperty('rate', current_rate + rate_value)
            
            if self.volume != "+0%":
                volume_value = int(self.volume.replace('%', ''))
                if volume_value != 0:
                    self.tts_engine.setProperty('volume', volume_value / 100.0)
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            self.tts_engine.save_to_file(text, tmp_path)
            self.tts_engine.runAndWait()
            
            # 读取生成的音频文件
            audio_array, sample_rate = sf.read(tmp_path)
            
            # 删除临时文件
            os.unlink(tmp_path)
            
            return TTSResult(
                audio_data=audio_array,
                sample_rate=sample_rate,
                duration=len(audio_array) / sample_rate,
                text=text,
                voice=voice,
                engine=self.engine,
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"pyttsx3合成失败: {e}")
            return None
    
    def get_available_voices(self, engine: Optional[str] = None) -> List[Dict[str, str]]:
        """
        获取可用语音列表
        
        Args:
            engine: 指定引擎，None则使用当前引擎
            
        Returns:
            List[Dict[str, str]]: 语音列表
        """
        target_engine = engine or self.engine
        
        try:
            if target_engine == "edge":
                return self._get_edge_voices()
            elif target_engine == "coqui":
                return self._get_coqui_voices()
            elif target_engine == "pyttsx3":
                return self._get_pyttsx3_voices()
            else:
                return []
        except Exception as e:
            logger.error(f"获取语音列表失败: {e}")
            return []
    
    def _get_edge_voices(self) -> List[Dict[str, str]]:
        """获取Edge TTS语音列表"""
        try:
            import asyncio
            import edge_tts
            
            voices = asyncio.run(edge_tts.list_voices())
            voice_list = []
            
            for voice in voices:
                voice_list.append({
                    "name": voice["Name"],
                    "short_name": voice["ShortName"],
                    "gender": voice["Gender"],
                    "locale": voice["Locale"]
                })
            
            return voice_list
        except Exception as e:
            logger.error(f"获取Edge TTS语音列表失败: {e}")
            return []
    
    def _get_coqui_voices(self) -> List[Dict[str, str]]:
        """获取Coqui TTS语音列表"""
        try:
            # Coqui TTS的语音列表获取比较复杂，这里返回一些预设的语音
            return [
                {"name": "default", "description": "默认语音"},
                {"name": "female_1", "description": "女声1"},
                {"name": "male_1", "description": "男声1"},
            ]
        except Exception as e:
            logger.error(f"获取Coqui TTS语音列表失败: {e}")
            return []
    
    def _get_pyttsx3_voices(self) -> List[Dict[str, str]]:
        """获取pyttsx3语音列表"""
        try:
            if self.tts_engine is None:
                return []
            
            voices = self.tts_engine.getProperty('voices')
            voice_list = []
            
            for voice in voices:
                voice_list.append({
                    "name": voice.name,
                    "id": voice.id,
                    "languages": getattr(voice, 'languages', [])
                })
            
            return voice_list
        except Exception as e:
            logger.error(f"获取pyttsx3语音列表失败: {e}")
            return []
    
    def set_voice_parameters(self,
                           voice: Optional[str] = None,
                           rate: Optional[str] = None,
                           volume: Optional[str] = None,
                           pitch: Optional[str] = None) -> None:
        """
        设置语音参数
        
        Args:
            voice: 语音名称
            rate: 语速
            volume: 音量
            pitch: 音调
        """
        if voice is not None:
            self.voice = voice
        
        if rate is not None:
            self.rate = rate
        
        if volume is not None:
            self.volume = volume
        
        if pitch is not None:
            self.pitch = pitch
        
        logger.info(f"语音参数已更新: voice={self.voice}, rate={self.rate}, "
                   f"volume={self.volume}, pitch={self.pitch}")
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """
        获取合成统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "engine": self.engine,
            "voice": self.voice,
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": self.avg_processing_time,
            "supported_engines": self.supported_engines,
            "current_engine_available": self.engine in self.supported_engines
        }
    
    def test_voice(self, test_text: str = "你好，这是语音合成测试") -> bool:
        """
        测试语音合成
        
        Args:
            test_text: 测试文本
            
        Returns:
            bool: 测试是否成功
        """
        try:
            result = self.synthesize(test_text)
            return result is not None
        except Exception as e:
            logger.error(f"语音测试失败: {e}")
            return False
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        logger.info("TTS统计信息已重置")
    
    def __del__(self):
        """析构函数"""
        try:
            if self.tts_engine is not None:
                # pyttsx3引擎清理
                pass
        except Exception:
            pass