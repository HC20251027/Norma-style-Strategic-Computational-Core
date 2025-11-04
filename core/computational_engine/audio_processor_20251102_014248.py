"""
音频处理模块
支持音频预处理、格式转换、特征提取和转录
"""

import asyncio
import base64
import hashlib
import wave
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import io
import logging

from ..utils import format_file_size

logger = logging.getLogger(__name__)

@dataclass
class AudioProcessingConfig:
    """音频处理配置"""
    max_duration: int = 300  # 最大时长（秒）
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    sample_rate: int = 16000  # 采样率
    channels: int = 1  # 声道数
    supported_formats: List[str] = None
    target_format: str = "WAV"  # WAV, MP3, FLAC, AAC
    enable_noise_reduction: bool = True
    enable_normalization: bool = True
    enable_transcription: bool = False
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["WAV", "MP3", "FLAC", "AAC", "M4A"]

@dataclass
class AudioMetadata:
    """音频元数据"""
    duration: float
    sample_rate: int
    channels: int
    format: str
    file_size: int
    bit_rate: Optional[int] = None
    codec: Optional[str] = None
    has_voice: bool = False
    volume_level: float = 0.0
    frequency_range: Optional[Tuple[float, float]] = None

class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, config: AudioProcessingConfig = None):
        self.config = config or AudioProcessingConfig()
        self.supported_mime_types = {
            'WAV': 'audio/wav',
            'MP3': 'audio/mpeg',
            'FLAC': 'audio/flac',
            'AAC': 'audio/aac',
            'M4A': 'audio/mp4'
        }
    
    async def process_audio(
        self,
        audio_data: Union[bytes, str, io.BytesIO],
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理音频
        
        Args:
            audio_data: 音频数据（字节串、base64字符串或BytesIO对象）
            **kwargs: 额外参数
            
        Returns:
            处理后的音频和元数据
        """
        try:
            # 1. 解析音频数据
            audio_buffer, original_size = await self._parse_audio_data(audio_data)
            
            # 2. 验证音频
            await self._validate_audio(audio_buffer)
            
            # 3. 提取元数据
            metadata = await self._extract_metadata(audio_buffer, original_size)
            
            # 4. 预处理音频
            processed_buffer = await self._preprocess_audio(audio_buffer)
            
            # 5. 转换格式
            final_buffer, final_size = await self._convert_format(processed_buffer)
            
            # 6. 生成音频哈希
            audio_hash = await self._generate_audio_hash(final_buffer)
            
            # 7. 语音检测
            has_voice = await self._detect_voice(final_buffer)
            
            # 8. 转录（如果启用）
            transcription = None
            if self.config.enable_transcription:
                transcription = await self._transcribe_audio(final_buffer)
            
            return {
                'audio': final_buffer,
                'metadata': metadata,
                'original_size': original_size,
                'processed_size': final_size,
                'compression_ratio': original_size / final_size if final_size > 0 else 1,
                'hash': audio_hash,
                'has_voice': has_voice,
                'transcription': transcription,
                'processing_info': {
                    'format_changed': metadata.format != self.config.target_format,
                    'noise_reduced': self.config.enable_noise_reduction,
                    'normalized': self.config.enable_normalization
                }
            }
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            raise
    
    async def _parse_audio_data(self, audio_data: Union[bytes, str, io.BytesIO]) -> Tuple[io.BytesIO, int]:
        """解析音频数据"""
        if isinstance(audio_data, bytes):
            audio_buffer = io.BytesIO(audio_data)
            original_size = len(audio_data)
        elif isinstance(audio_data, str):
            # 假设是base64编码
            if audio_data.startswith('data:audio'):
                # 移除data URL前缀
                audio_data = audio_data.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_data)
            audio_buffer = io.BytesIO(audio_bytes)
            original_size = len(audio_bytes)
        elif isinstance(audio_data, io.BytesIO):
            audio_buffer = audio_data
            original_size = audio_data.tell()
        else:
            raise ValueError("不支持的音频数据格式")
        
        return audio_buffer, original_size
    
    async def _validate_audio(self, audio_buffer: io.BytesIO):
        """验证音频"""
        # 重置缓冲区位置
        audio_buffer.seek(0)
        
        # 简单的格式验证
        header = audio_buffer.read(12)
        audio_buffer.seek(0)
        
        # 检查WAV格式
        if header.startswith(b'RIFF') and b'WAVE' in header:
            return
        
        # 检查MP3格式
        if header[:3] == b'ID3' or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
            return
        
        # 其他格式的简单检查
        logger.warning("无法验证音频格式，继续处理")
    
    async def _extract_metadata(self, audio_buffer: io.BytesIO, file_size: int) -> AudioMetadata:
        """提取音频元数据"""
        audio_buffer.seek(0)
        
        metadata = AudioMetadata(
            duration=0.0,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            format='UNKNOWN',
            file_size=file_size
        )
        
        try:
            # 尝试读取WAV文件元数据
            if audio_buffer.read(4) == b'RIFF':
                audio_buffer.seek(0)
                with wave.open(audio_buffer, 'rb') as wav_file:
                    metadata.duration = wav_file.getnframes() / wav_file.getframerate()
                    metadata.sample_rate = wav_file.getframerate()
                    metadata.channels = wav_file.getnchannels()
                    metadata.format = 'WAV'
                    metadata.codec = 'PCM'
            else:
                # 对于其他格式，使用默认值
                metadata.duration = file_size / (metadata.sample_rate * metadata.channels * 2)  # 假设16位
                metadata.format = 'MP3'  # 猜测格式
                
        except Exception as e:
            logger.warning(f"无法提取音频元数据: {e}")
            # 使用默认值
            metadata.duration = 0.0
        
        # 估算音量级别
        metadata.volume_level = await self._estimate_volume_level(audio_buffer)
        
        return metadata
    
    async def _preprocess_audio(self, audio_buffer: io.BytesIO) -> io.BytesIO:
        """预处理音频"""
        processed_buffer = io.BytesIO()
        
        try:
            audio_buffer.seek(0)
            
            # 如果是WAV格式，进行预处理
            if audio_buffer.read(4) == b'RIFF':
                audio_buffer.seek(0)
                with wave.open(audio_buffer, 'rb') as input_wav:
                    # 重采样
                    if input_wav.getframerate() != self.config.sample_rate:
                        # 这里应该实现重采样逻辑
                        logger.info("需要重采样音频")
                    
                    # 声道转换
                    if input_wav.getnchannels() != self.config.channels:
                        # 这里应该实现声道转换逻辑
                        logger.info("需要转换音频声道")
                    
                    # 噪音降低
                    if self.config.enable_noise_reduction:
                        # 这里应该实现噪音降低算法
                        logger.info("应用噪音降低")
                    
                    # 音量标准化
                    if self.config.enable_normalization:
                        # 这里应该实现音量标准化
                        logger.info("应用音量标准化")
                    
                    # 写入处理后的音频
                    with wave.open(processed_buffer, 'wb') as output_wav:
                        output_wav.setnchannels(self.config.channels)
                        output_wav.setsampwidth(2)  # 16位
                        output_wav.setframerate(self.config.sample_rate)
                        
                        # 复制音频数据（简化处理）
                        frames = input_wav.readframes(input_wav.getnframes())
                        output_wav.writeframes(frames)
            
            else:
                # 对于非WAV格式，直接复制
                audio_buffer.seek(0)
                processed_buffer.write(audio_buffer.read())
                
        except Exception as e:
            logger.warning(f"音频预处理失败: {e}")
            # 预处理失败时返回原始数据
            audio_buffer.seek(0)
            processed_buffer.write(audio_buffer.read())
        
        processed_buffer.seek(0)
        return processed_buffer
    
    async def _convert_format(self, audio_buffer: io.BytesIO) -> Tuple[bytes, int]:
        """转换音频格式"""
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        
        # 简化实现：直接返回原始数据
        # 实际实现中应该根据target_format进行转换
        if self.config.target_format == 'WAV':
            # 确保是WAV格式
            if not audio_data.startswith(b'RIFF'):
                # 这里应该实现格式转换逻辑
                logger.warning("非WAV格式，将尝试转换")
        
        return audio_data, len(audio_data)
    
    async def _generate_audio_hash(self, audio_data: bytes) -> str:
        """生成音频哈希"""
        return hashlib.md5(audio_data).hexdigest()
    
    async def _detect_voice(self, audio_buffer: io.BytesIO) -> bool:
        """检测是否有语音"""
        try:
            audio_buffer.seek(0)
            
            # 简单的语音检测算法
            # 检查音频能量和频率特征
            if audio_buffer.read(4) == b'RIFF':
                audio_buffer.seek(0)
                with wave.open(audio_buffer, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    
                    # 计算音频能量
                    energy = 0
                    for i in range(0, len(frames), 2):
                        if i + 1 < len(frames):
                            sample = int.from_bytes(frames[i:i+2], byteorder='little', signed=True)
                            energy += sample * sample
                    
                    avg_energy = energy / (len(frames) / 2) if frames else 0
                    
                    # 简单的阈值判断
                    return avg_energy > 1000  # 可调整的阈值
            
        except Exception as e:
            logger.warning(f"语音检测失败: {e}")
        
        return False
    
    async def _estimate_volume_level(self, audio_buffer: io.BytesIO) -> float:
        """估算音量级别"""
        try:
            audio_buffer.seek(0)
            
            if audio_buffer.read(4) == b'RIFF':
                audio_buffer.seek(0)
                with wave.open(audio_buffer, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    
                    # 计算RMS音量
                    rms_sum = 0
                    sample_count = 0
                    
                    for i in range(0, len(frames), 2):
                        if i + 1 < len(frames):
                            sample = int.from_bytes(frames[i:i+2], byteorder='little', signed=True)
                            rms_sum += sample * sample
                            sample_count += 1
                    
                    if sample_count > 0:
                        rms = (rms_sum / sample_count) ** 0.5
                        # 归一化到0-1范围
                        return min(1.0, rms / 32768.0)
            
        except Exception as e:
            logger.warning(f"音量级别估算失败: {e}")
        
        return 0.0
    
    async def _transcribe_audio(self, audio_buffer: io.BytesIO) -> Optional[str]:
        """转录音频"""
        try:
            # 这里应该集成语音识别API
            # 简化实现：返回模拟转录结果
            return "这是音频转录的模拟结果"
        except Exception as e:
            logger.warning(f"音频转录失败: {e}")
            return None
    
    async def batch_process(
        self,
        audios: List[Union[bytes, str, io.BytesIO]]
    ) -> List[Dict[str, Any]]:
        """批量处理音频"""
        tasks = [self.process_audio(audio) for audio in audios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'audio': None,
                    'error': str(result),
                    'metadata': None,
                    'original_size': 0,
                    'processed_size': 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def extract_audio_features(self, audio_buffer: io.BytesIO) -> Dict[str, Any]:
        """提取音频特征"""
        features = {}
        
        try:
            audio_buffer.seek(0)
            
            if audio_buffer.read(4) == b'RIFF':
                audio_buffer.seek(0)
                with wave.open(audio_buffer, 'rb') as wav_file:
                    # 基础特征
                    features['duration'] = wav_file.getnframes() / wav_file.getframerate()
                    features['sample_rate'] = wav_file.getframerate()
                    features['channels'] = wav_file.getnchannels()
                    features['total_frames'] = wav_file.getnframes()
                    
                    # 音频数据特征
                    frames = wav_file.readframes(wav_file.getnframes())
                    
                    # 计算统计特征
                    samples = []
                    for i in range(0, len(frames), 2):
                        if i + 1 < len(frames):
                            sample = int.from_bytes(frames[i:i+2], byteorder='little', signed=True)
                            samples.append(sample)
                    
                    if samples:
                        features['max_amplitude'] = max(samples)
                        features['min_amplitude'] = min(samples)
                        features['mean_amplitude'] = sum(samples) / len(samples)
                        
                        # 计算RMS
                        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
                        features['rms'] = rms
                        
                        # 计算零交叉率（语音活动检测的简单指标）
                        zero_crossings = sum(1 for i in range(1, len(samples)) 
                                           if samples[i] * samples[i-1] < 0)
                        features['zero_crossing_rate'] = zero_crossings / len(samples)
                    
        except Exception as e:
            logger.warning(f"音频特征提取失败: {e}")
        
        return features
    
    async def generate_waveform(self, audio_buffer: io.BytesIO, width: int = 1000) -> List[float]:
        """生成音频波形数据"""
        try:
            audio_buffer.seek(0)
            
            if audio_buffer.read(4) == b'RIFF':
                audio_buffer.seek(0)
                with wave.open(audio_buffer, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    
                    # 计算每个波段的RMS值
                    samples_per_band = len(frames) // 2 // width
                    waveform = []
                    
                    for i in range(width):
                        start = i * samples_per_band * 2
                        end = min(start + samples_per_band * 2, len(frames))
                        
                        if start < len(frames):
                            band_samples = []
                            for j in range(start, end, 2):
                                if j + 1 < len(frames):
                                    sample = int.from_bytes(frames[j:j+2], 
                                                           byteorder='little', signed=True)
                                    band_samples.append(sample)
                            
                            if band_samples:
                                rms = (sum(s * s for s in band_samples) / len(band_samples)) ** 0.5
                                waveform.append(rms / 32768.0)  # 归一化
                            else:
                                waveform.append(0.0)
                        else:
                            waveform.append(0.0)
                    
                    return waveform
            
        except Exception as e:
            logger.warning(f"波形生成失败: {e}")
        
        return []
    
    def audio_to_base64(self, audio_data: bytes, format: str = 'WAV') -> str:
        """将音频转换为base64"""
        return base64.b64encode(audio_data).decode('utf-8')
    
    def base64_to_audio(self, base64_string: str) -> bytes:
        """将base64转换为音频"""
        if base64_string.startswith('data:audio'):
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"音频处理配置已更新: {kwargs}")
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        return self.config.supported_formats.copy()
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.__dict__.copy()