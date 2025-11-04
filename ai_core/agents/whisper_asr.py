"""
Whisper ASR语音识别模块

集成OpenAI Whisper模型进行语音识别
"""

import whisper
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]
    processing_time: float
    model_used: str

class WhisperASR:
    """Whisper语音识别器"""
    
    def __init__(self,
                 model_name: str = "base",
                 language: str = "zh",
                 task: str = "transcribe",
                 device: str = "cpu",
                 temperature: float = 0.0):
        """
        初始化Whisper ASR
        
        Args:
            model_name: 模型名称 (tiny, base, small, medium, large)
            language: 识别语言
            task: 任务类型 (transcribe, translate)
            device: 运行设备 (cpu, cuda)
            temperature: 生成温度
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        self.device = device
        self.temperature = temperature
        
        # 模型实例
        self.model = None
        self.model_info = {}
        
        # 统计信息
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        
        logger.info(f"Whisper ASR初始化: model={model_name}, language={language}, device={device}")
    
    def load_model(self) -> bool:
        """
        加载Whisper模型
        
        Returns:
            bool: 是否成功加载
        """
        try:
            start_time = time.time()
            
            # 检查设备可用性
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA不可用，回退到CPU")
                self.device = "cpu"
            
            # 加载模型
            logger.info(f"正在加载Whisper模型: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            
            # 获取模型信息
            self.model_info = {
                "name": self.model_name,
                "device": self.device,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "multilingual": self.model.is_multilingual,
            }
            
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            logger.info(f"模型参数数量: {self.model_info['parameters']:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载Whisper模型失败: {e}")
            self.model = None
            return False
    
    def transcribe(self, 
                   audio_data: Union[np.ndarray, str], 
                   language: Optional[str] = None,
                   task: Optional[str] = None,
                   temperature: Optional[float] = None,
                   **kwargs) -> Optional[ASRResult]:
        """
        语音识别
        
        Args:
            audio_data: 音频数据或音频文件路径
            language: 识别语言，None则使用初始化时的语言
            task: 任务类型，None则使用初始化时的任务
            temperature: 生成温度，None则使用初始化时的温度
            **kwargs: 其他Whisper参数
            
        Returns:
            Optional[ASRResult]: 识别结果
        """
        if self.model is None:
            logger.error("模型未加载")
            return None
        
        try:
            start_time = time.time()
            
            # 使用默认参数或传入参数
            lang = language or self.language
            task_type = task or self.task
            temp = temperature if temperature is not None else self.temperature
            
            # 准备选项
            options = {
                "language": lang,
                "task": task_type,
                "temperature": temp,
                "verbose": False,
                **kwargs
            }
            
            # 执行识别
            logger.info(f"开始语音识别: 语言={lang}, 任务={task_type}")
            result = self.model.transcribe(audio_data, **options)
            
            processing_time = time.time() - start_time
            
            # 计算置信度（基于logprob）
            confidence = self._calculate_confidence(result)
            
            # 创建结果对象
            asr_result = ASRResult(
                text=result["text"].strip(),
                language=result.get("language", lang),
                confidence=confidence,
                segments=result.get("segments", []),
                processing_time=processing_time,
                model_used=self.model_name
            )
            
            # 更新统计信息
            self.total_requests += 1
            self.total_processing_time += processing_time
            self.avg_processing_time = self.total_processing_time / self.total_requests
            
            logger.info(f"语音识别完成: '{asr_result.text}', 耗时: {processing_time:.2f}秒, 置信度: {confidence:.2f}")
            
            return asr_result
            
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return None
    
    def transcribe_streaming(self, 
                           audio_chunks: List[np.ndarray],
                           language: Optional[str] = None,
                           **kwargs) -> Optional[ASRResult]:
        """
        流式语音识别
        
        Args:
            audio_chunks: 音频块列表
            language: 识别语言
            **kwargs: 其他参数
            
        Returns:
            Optional[ASRResult]: 识别结果
        """
        try:
            # 合并音频块
            if not audio_chunks:
                return None
            
            combined_audio = np.concatenate(audio_chunks)
            
            # 执行识别
            return self.transcribe(combined_audio, language, **kwargs)
            
        except Exception as e:
            logger.error(f"流式语音识别失败: {e}")
            return None
    
    def _calculate_confidence(self, whisper_result: Dict[str, Any]) -> float:
        """
        计算识别置信度
        
        Args:
            whisper_result: Whisper识别结果
            
        Returns:
            float: 置信度 (0-1)
        """
        try:
            segments = whisper_result.get("segments", [])
            
            if not segments:
                return 0.5  # 默认中等置信度
            
            # 计算所有段的平均logprob
            logprobs = []
            for segment in segments:
                avg_logprob = segment.get("avg_logprob", -1.0)
                if avg_logprob is not None:
                    logprobs.append(avg_logprob)
            
            if not logprobs:
                return 0.5
            
            # 将logprob转换为置信度
            avg_logprob = np.mean(logprobs)
            # Whisper的logprob通常在-1到0之间，转换为0-1的置信度
            confidence = max(0, min(1, (avg_logprob + 1.0)))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"计算置信度失败: {e}")
            return 0.5
    
    def detect_language(self, audio_data: Union[np.ndarray, str]) -> Optional[str]:
        """
        检测音频语言
        
        Args:
            audio_data: 音频数据或文件路径
            
        Returns:
            Optional[str]: 检测到的语言
        """
        if self.model is None:
            logger.error("模型未加载")
            return None
        
        try:
            # 使用Whisper的语言检测
            audio = whisper.load_audio(audio_data) if isinstance(audio_data, str) else audio_data
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # 获取语言概率
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            
            logger.info(f"检测到语言: {detected_language}, 概率: {probs[detected_language]:.2f}")
            
            return detected_language
            
        except Exception as e:
            logger.error(f"语言检测失败: {e}")
            return None
    
    def get_supported_languages(self) -> List[str]:
        """
        获取支持的语言列表
        
        Returns:
            List[str]: 支持的语言列表
        """
        if self.model is None:
            return []
        
        try:
            return whisper.tokenizer.TO_LANGUAGE_CODE.keys()
        except Exception:
            # 如果无法获取完整列表，返回常见语言
            return ["zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru"]
    
    def benchmark_model(self, test_audio: Union[np.ndarray, str], num_runs: int = 3) -> Dict[str, float]:
        """
        模型性能基准测试
        
        Args:
            test_audio: 测试音频
            num_runs: 运行次数
            
        Returns:
            Dict[str, float]: 性能统计
        """
        if self.model is None:
            logger.error("模型未加载")
            return {}
        
        try:
            logger.info(f"开始模型基准测试，运行 {num_runs} 次")
            
            times = []
            for i in range(num_runs):
                start_time = time.time()
                result = self.transcribe(test_audio)
                end_time = time.time()
                
                if result:
                    times.append(end_time - start_time)
                    logger.info(f"第 {i+1} 次运行: {times[-1]:.2f}秒")
                else:
                    logger.warning(f"第 {i+1} 次运行失败")
            
            if not times:
                return {}
            
            return {
                "avg_time": np.mean(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "std_time": np.std(times),
                "runs_successful": len(times),
                "runs_total": num_runs
            }
            
        except Exception as e:
            logger.error(f"基准测试失败: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "language": self.language,
            "task": self.task,
            "temperature": self.temperature,
            "model_loaded": self.model is not None,
            "statistics": {
                "total_requests": self.total_requests,
                "total_processing_time": self.total_processing_time,
                "avg_processing_time": self.avg_processing_time,
            }
        }
        
        if self.model_info:
            info.update(self.model_info)
        
        return info
    
    def set_parameters(self, 
                      model_name: Optional[str] = None,
                      language: Optional[str] = None,
                      task: Optional[str] = None,
                      temperature: Optional[float] = None) -> None:
        """
        设置模型参数
        
        Args:
            model_name: 模型名称
            language: 识别语言
            task: 任务类型
            temperature: 生成温度
        """
        if model_name is not None:
            self.model_name = model_name
        
        if language is not None:
            self.language = language
        
        if task is not None:
            self.task = task
        
        if temperature is not None:
            self.temperature = temperature
        
        logger.info(f"参数已更新: model={self.model_name}, language={self.language}, "
                   f"task={self.task}, temperature={self.temperature}")
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        logger.info("统计信息已重置")
    
    def __del__(self):
        """析构函数"""
        try:
            if self.model is not None and hasattr(self.model, 'cpu'):
                # 移动模型到CPU以释放GPU内存
                self.model.cpu()
        except Exception:
            pass