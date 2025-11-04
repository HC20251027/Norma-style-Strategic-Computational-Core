"""
音频处理模块

提供语音质量检测、降噪、标准化等音频处理功能
"""

import numpy as np
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """音频处理器"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 noise_reduction: bool = True,
                 noise_reduction_level: float = 0.7,
                 normalize: bool = True,
                 high_pass_filter: bool = True,
                 vad_threshold: float = 0.5):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率
            noise_reduction: 是否启用降噪
            noise_reduction_level: 降噪强度 (0.0-1.0)
            normalize: 是否标准化音频
            high_pass_filter: 是否启用高通滤波
            vad_threshold: 语音活动检测阈值
        """
        self.sample_rate = sample_rate
        self.noise_reduction = noise_reduction
        self.noise_reduction_level = noise_reduction_level
        self.normalize = normalize
        self.high_pass_filter = high_pass_filter
        self.vad_threshold = vad_threshold
        
        # 噪声配置文件
        self.noise_profile = None
        self.noise_samples = []
        
        logger.info(f"音频处理器初始化完成: sample_rate={sample_rate}, "
                   f"noise_reduction={noise_reduction}")
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        处理音频数据
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            processed = audio_data.copy()
            
            # 1. 高通滤波（去除低频噪声）
            if self.high_pass_filter:
                processed = self._apply_high_pass_filter(processed)
            
            # 2. 降噪处理
            if self.noise_reduction:
                processed = self._apply_noise_reduction(processed)
            
            # 3. 音频标准化
            if self.normalize:
                processed = self._normalize_audio(processed)
            
            # 4. 动态范围压缩
            processed = self._apply_dynamic_range_compression(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            return audio_data
    
    def _apply_high_pass_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        应用高通滤波器
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            np.ndarray: 滤波后的音频数据
        """
        try:
            # 设计高通滤波器（截止频率80Hz）
            nyquist = self.sample_rate / 2
            cutoff = 80 / nyquist
            
            # 使用Butterworth滤波器
            b, a = signal.butter(4, cutoff, btype='high')
            filtered = signal.filtfilt(b, a, audio_data)
            
            return filtered.astype(audio_data.dtype)
            
        except Exception as e:
            logger.warning(f"高通滤波失败: {e}")
            return audio_data
    
    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        应用降噪算法
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            np.ndarray: 降噪后的音频数据
        """
        try:
            # 频域降噪
            return self._spectral_subtraction(audio_data)
            
        except Exception as e:
            logger.warning(f"降噪处理失败: {e}")
            return audio_data
    
    def _spectral_subtraction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        频谱减法降噪
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            np.ndarray: 降噪后的音频数据
        """
        # 转换为浮点数进行处理
        audio_float = audio_data.astype(np.float32)
        
        # 短时傅里叶变换
        window_length = int(0.025 * self.sample_rate)  # 25ms窗口
        hop_length = int(0.010 * self.sample_rate)     # 10ms跳跃
        
        # 分帧处理
        frames = self._frame_signal(audio_float, window_length, hop_length)
        
        # 对每一帧进行频谱减法
        denoised_frames = []
        for frame in frames:
            denoised_frame = self._spectral_subtraction_frame(frame)
            denoised_frames.append(denoised_frame)
        
        # 重构信号
        denoised_audio = self._overlap_add(denoised_frames, hop_length)
        
        return denoised_audio.astype(audio_data.dtype)
    
    def _spectral_subtraction_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧进行频谱减法
        
        Args:
            frame: 单帧音频数据
            
        Returns:
            np.ndarray: 降噪后的帧数据
        """
        # 加窗
        windowed = frame * signal.windows.hann(len(frame))
        
        # FFT
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # 估计噪声频谱
        if self.noise_profile is None:
            # 使用前几帧估计噪声
            if len(self.noise_samples) < 5:
                self.noise_samples.append(magnitude)
                return frame
            else:
                self.noise_profile = np.mean(self.noise_samples, axis=0)
        
        # 频谱减法
        alpha = 2.0 * self.noise_reduction_level  # 过减法因子
        clean_magnitude = magnitude - alpha * self.noise_profile
        
        # 确保幅度非负
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # 重构频谱
        clean_spectrum = clean_magnitude * np.exp(1j * phase)
        
        # IFFT
        clean_frame = np.fft.irfft(clean_spectrum)
        
        return clean_frame[:len(frame)]
    
    def _frame_signal(self, signal: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
        """
        将信号分帧
        
        Args:
            signal: 输入信号
            window_length: 窗口长度
            hop_length: 跳跃长度
            
        Returns:
            np.ndarray: 分帧后的信号
        """
        frames = []
        for i in range(0, len(signal) - window_length + 1, hop_length):
            frames.append(signal[i:i + window_length])
        return np.array(frames)
    
    def _overlap_add(self, frames: list, hop_length: int) -> np.ndarray:
        """
        重叠相加重构信号
        
        Args:
            frames: 分帧后的信号列表
            hop_length: 跳跃长度
            
        Returns:
            np.ndarray: 重构的信号
        """
        if not frames:
            return np.array([])
        
        frame_length = len(frames[0])
        num_frames = len(frames)
        
        # 计算输出信号长度
        output_length = frame_length + (num_frames - 1) * hop_length
        output = np.zeros(output_length)
        
        # 重叠相加
        for i, frame in enumerate(frames):
            start = i * hop_length
            end = start + frame_length
            output[start:end] += frame
        
        return output
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        音频标准化
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            np.ndarray: 标准化后的音频数据
        """
        # 计算RMS值
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        if rms > 0:
            # 目标RMS值（-20dB）
            target_rms = 0.1
            gain = target_rms / rms
            normalized = audio_data * gain
            
            # 限制幅度
            max_val = np.iinfo(audio_data.dtype).max
            normalized = np.clip(normalized, -max_val, max_val)
            
            return normalized.astype(audio_data.dtype)
        
        return audio_data
    
    def _apply_dynamic_range_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """
        应用动态范围压缩
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            np.ndarray: 压缩后的音频数据
        """
        # 简单的动态范围压缩
        threshold = 0.7
        ratio = 4.0
        
        # 转换为浮点数
        audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        
        # 应用压缩
        compressed = np.where(
            np.abs(audio_float) > threshold,
            np.sign(audio_float) * (threshold + (np.abs(audio_float) - threshold) / ratio),
            audio_float
        )
        
        # 转换回原始格式
        return (compressed * np.iinfo(audio_data.dtype).max).astype(audio_data.dtype)
    
    def detect_voice_activity(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        语音活动检测（VAD）
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            Tuple[bool, float]: (是否为语音, 语音概率)
        """
        try:
            # 计算短时能量
            frame_length = int(0.025 * self.sample_rate)  # 25ms帧
            hop_length = int(0.010 * self.sample_rate)    # 10ms跳跃
            
            frames = self._frame_signal(audio_data, frame_length, hop_length)
            energies = [np.sum(frame ** 2) for frame in frames]
            
            if not energies:
                return False, 0.0
            
            # 计算零交叉率
            zcrs = []
            for frame in frames:
                zero_crossings = np.sum(np.diff(np.sign(frame)) != 0)
                zcr = zero_crossings / len(frame)
                zcrs.append(zcr)
            
            # 组合特征判断语音
            avg_energy = np.mean(energies)
            avg_zcr = np.mean(zcrs)
            
            # 简单的VAD规则
            energy_threshold = 0.01
            zcr_threshold = 0.3
            
            is_speech = (avg_energy > energy_threshold and 
                        avg_zcr < zcr_threshold)
            
            # 计算语音概率
            speech_prob = min(avg_energy / energy_threshold, 1.0)
            
            return is_speech, speech_prob
            
        except Exception as e:
            logger.error(f"VAD检测失败: {e}")
            return False, 0.0
    
    def analyze_audio_quality(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        分析音频质量
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            Dict[str, Any]: 音频质量分析结果
        """
        try:
            # 基本统计信息
            rms = np.sqrt(np.mean(audio_data ** 2))
            peak = np.max(np.abs(audio_data))
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            
            # 频域分析
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            dominant_freq = np.argmax(magnitude) * self.sample_rate / (2 * len(magnitude))
            
            # 信噪比估计
            snr_estimate = self._estimate_snr(audio_data)
            
            # 语音活动检测
            is_speech, speech_prob = self.detect_voice_activity(audio_data)
            
            return {
                "rms": float(rms),
                "peak": float(peak),
                "dynamic_range": float(dynamic_range),
                "dominant_frequency": float(dominant_freq),
                "snr_estimate": float(snr_estimate),
                "is_speech": bool(is_speech),
                "speech_probability": float(speech_prob),
                "duration": len(audio_data) / self.sample_rate,
                "sample_rate": self.sample_rate,
                "quality_score": self._calculate_quality_score(rms, peak, snr_estimate)
            }
            
        except Exception as e:
            logger.error(f"音频质量分析失败: {e}")
            return {"error": str(e)}
    
    def _estimate_snr(self, audio_data: np.ndarray) -> float:
        """
        估计信噪比
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            float: 估计的信噪比（dB）
        """
        try:
            # 使用频谱分析估计SNR
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft) ** 2
            
            # 假设前10%为噪声
            noise_floor = np.mean(magnitude[:len(magnitude)//10])
            signal_power = np.mean(magnitude)
            
            if noise_floor > 0:
                snr = 10 * np.log10(signal_power / noise_floor)
                return max(snr, 0)  # 确保非负
            return 0
            
        except Exception:
            return 0
    
    def _calculate_quality_score(self, rms: float, peak: float, snr: float) -> float:
        """
        计算音频质量评分
        
        Args:
            rms: RMS值
            peak: 峰值
            snr: 信噪比
            
        Returns:
            float: 质量评分 (0-1)
        """
        try:
            # RMS评分 (0-0.3为正常范围)
            rms_score = 1.0 if 0.01 <= rms <= 0.3 else max(0, 1 - abs(rms - 0.15) / 0.15)
            
            # 峰值评分 (避免削波)
            peak_score = 1.0 if peak < 0.95 else max(0, 1 - (peak - 0.95) / 0.05)
            
            # SNR评分
            snr_score = min(snr / 30.0, 1.0)  # 30dB以上为满分
            
            # 综合评分
            quality_score = (rms_score * 0.3 + peak_score * 0.3 + snr_score * 0.4)
            
            return max(0, min(1, quality_score))
            
        except Exception:
            return 0.5  # 默认中等质量
    
    def reset_noise_profile(self) -> None:
        """重置噪声配置文件"""
        self.noise_profile = None
        self.noise_samples = []
        logger.info("噪声配置文件已重置")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "sample_rate": self.sample_rate,
            "noise_reduction_enabled": self.noise_reduction,
            "normalization_enabled": self.normalize,
            "high_pass_filter_enabled": self.high_pass_filter,
            "noise_reduction_level": self.noise_reduction_level,
            "vad_threshold": self.vad_threshold,
            "noise_profile_available": self.noise_profile is not None,
            "noise_samples_count": len(self.noise_samples)
        }