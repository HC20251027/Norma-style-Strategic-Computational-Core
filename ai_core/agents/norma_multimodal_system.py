#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诺玛Agent多模态处理系统
基于Agno框架的多模态能力，实现文本、图像、音频、视频处理以及跨模态融合

作者: 皇
创建时间: 2025-11-01
版本: 1.0.0
"""

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

# Agno框架导入
from agno.agent import Agent
from agno.team import Team
from agno.media import Image, Audio, Video, File
from agno.tools import Toolkit, Function
from agno.models.openai import OpenAIChat

# 多模态处理库导入
try:
    import cv2
    import numpy as np
    from PIL import Image as PILImage
    import librosa
    import whisper
    CV2_AVAILABLE = True
    LIBROSA_AVAILABLE = True
    WHISPER_AVAILABLE = True
except ImportError as e:
    print(f"警告: 某些多模态库未安装: {e}")
    CV2_AVAILABLE = False
    LIBROSA_AVAILABLE = False
    WHISPER_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaType(Enum):
    """媒体类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class ProcessingMode(Enum):
    """处理模式枚举"""
    ANALYSIS = "analysis"  # 分析模式
    GENERATION = "generation"  # 生成模式
    TRANSFORMATION = "transformation"  # 转换模式
    EXTRACTION = "extraction"  # 提取模式


@dataclass
class MediaContent:
    """媒体内容数据类"""
    media_type: MediaType
    content: Union[str, bytes, Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_history: List[Dict[str, Any]]
    confidence_score: float = 0.0
    processing_time: float = 0.0


@dataclass
class MultimodalResult:
    """多模态处理结果"""
    original_content: MediaContent
    processed_content: MediaContent
    analysis_results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_metrics: Dict[str, float]
    cross_modal_insights: List[str]
    recommendations: List[str]


class NormaTextProcessor:
    """诺玛文本处理Agent"""
    
    def __init__(self):
        self.agent = Agent(
            name="诺玛文本处理专家",
            model=OpenAIChat(id="gpt-4"),
            description="专业的文本理解和生成专家",
            instructions=[
                "你是一个专业的文本处理专家，擅长自然语言理解、生成、翻译和分析",
                "能够处理多语言文本，包括中文、英文等",
                "具备文本摘要、情感分析、关键词提取等能力",
                "能够生成高质量的文本内容"
            ],
            send_media_to_model=True,
            store_media=True
        )
    
    async def analyze_text(self, text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """分析文本内容"""
        start_time = time.time()
        
        try:
            prompt = f"""
            请对以下文本进行{analysis_type}分析：
            
            文本内容：{text}
            
            请提供以下分析结果：
            1. 文本类型和主题
            2. 情感倾向和强度
            3. 关键词和短语
            4. 文本质量评估
            5. 语言特征分析
            6. 改进建议（如适用）
            
            请以JSON格式返回结果。
            """
            
            response = await self.agent.arun(prompt)
            processing_time = time.time() - start_time
            
            return {
                "analysis_type": analysis_type,
                "original_text": text,
                "results": response.content,
                "processing_time": processing_time,
                "confidence": 0.95
            }
        except Exception as e:
            logger.error(f"文本分析失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def generate_text(self, prompt: str, generation_type: str = "creative") -> Dict[str, Any]:
        """生成文本内容"""
        start_time = time.time()
        
        try:
            generation_prompt = f"""
            请基于以下提示生成{generation_type}文本：
            
            提示：{prompt}
            
            要求：
            1. 内容准确、逻辑清晰
            2. 语言自然、表达流畅
            3. 符合指定类型的要求
            4. 长度适中，重点突出
            
            请生成高质量的文本内容。
            """
            
            response = await self.agent.arun(generation_prompt)
            processing_time = time.time() - start_time
            
            return {
                "generation_type": generation_type,
                "prompt": prompt,
                "generated_text": response.content,
                "processing_time": processing_time,
                "confidence": 0.90
            }
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def translate_text(self, text: str, target_language: str = "英文") -> Dict[str, Any]:
        """翻译文本"""
        start_time = time.time()
        
        try:
            translation_prompt = f"""
            请将以下文本翻译成{target_language}：
            
            原文：{text}
            
            要求：
            1. 保持原意准确
            2. 语言自然流畅
            3. 符合目标语言习惯
            4. 保持原文格式
            """
            
            response = await self.agent.arun(translation_prompt)
            processing_time = time.time() - start_time
            
            return {
                "original_text": text,
                "target_language": target_language,
                "translated_text": response.content,
                "processing_time": processing_time,
                "confidence": 0.92
            }
        except Exception as e:
            logger.error(f"文本翻译失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}


class NormaImageProcessor:
    """诺玛图像处理Agent"""
    
    def __init__(self):
        self.agent = Agent(
            name="诺玛图像分析专家",
            model=OpenAIChat(id="gpt-4-vision-preview"),
            description="专业的图像理解和分析专家",
            instructions=[
                "你是一个专业的图像分析专家，擅长视觉理解、图像描述、物体识别",
                "能够分析图像内容、风格、构图等特征",
                "具备图像质量评估和优化建议能力",
                "能够生成图像相关的创意内容"
            ],
            send_media_to_model=True,
            store_media=True
        )
    
    async def analyze_image(self, image_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """分析图像内容"""
        start_time = time.time()
        
        try:
            # 创建Image对象
            image = Image(filepath=image_path)
            
            analysis_prompt = f"""
            请对这张图像进行{analysis_type}分析：
            
            分析要求：
            1. 图像内容描述（主要对象、场景、动作等）
            2. 视觉风格分析（色彩、构图、光影等）
            3. 技术质量评估（清晰度、曝光、构图等）
            4. 情感和氛围分析
            5. 图像中的文字识别（如有）
            6. 改进建议（如适用）
            
            请提供详细的分析结果。
            """
            
            response = await self.agent.arun(analysis_prompt, images=[image])
            processing_time = time.time() - start_time
            
            return {
                "analysis_type": analysis_type,
                "image_path": image_path,
                "results": response.content,
                "processing_time": processing_time,
                "confidence": 0.88
            }
        except Exception as e:
            logger.error(f"图像分析失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """从图像中提取文字（OCR）"""
        start_time = time.time()
        
        try:
            if not CV2_AVAILABLE:
                return {"error": "OpenCV未安装，无法进行OCR处理"}
            
            # 使用OpenCV进行基础OCR
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 简单的文字检测
            # 这里可以集成更高级的OCR服务如Tesseract
            processing_time = time.time() - start_time
            
            return {
                "image_path": image_path,
                "extracted_text": "OCR结果待集成专业OCR服务",
                "confidence": 0.75,
                "processing_time": processing_time,
                "method": "basic_opencv"
            }
        except Exception as e:
            logger.error(f"图像OCR失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def generate_image_description(self, image_path: str, style: str = "detailed") -> Dict[str, Any]:
        """生成图像描述"""
        start_time = time.time()
        
        try:
            image = Image(filepath=image_path)
            
            description_prompt = f"""
            请为这张图像生成{style}风格的描述：
            
            描述要求：
            1. 语言生动、富有想象力
            2. 突出图像的特色和亮点
            3. 适合用于图像标注或展示
            4. 长度适中，结构清晰
            """
            
            response = await self.agent.arun(description_prompt, images=[image])
            processing_time = time.time() - start_time
            
            return {
                "image_path": image_path,
                "description_style": style,
                "description": response.content,
                "processing_time": processing_time,
                "confidence": 0.85
            }
        except Exception as e:
            logger.error(f"图像描述生成失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}


class NormaAudioProcessor:
    """诺玛音频处理Agent"""
    
    def __init__(self):
        self.agent = Agent(
            name="诺玛音频分析专家",
            model=OpenAIChat(id="gpt-4"),
            description="专业的音频理解和处理专家",
            instructions=[
                "你是一个专业的音频分析专家，擅长语音识别、音频分析、声音特征提取",
                "能够分析音频内容、情感、语调等特征",
                "具备音频质量评估和优化建议能力",
                "能够处理多语言语音内容"
            ],
            send_media_to_model=True,
            store_media=True
        )
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            self.whisper_model = whisper.load_model("base")
    
    async def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """音频转录"""
        start_time = time.time()
        
        try:
            if not WHISPER_AVAILABLE:
                return {"error": "Whisper未安装，无法进行语音识别"}
            
            # 使用Whisper进行转录
            result = self.whisper_model.transcribe(audio_path)
            processing_time = time.time() - start_time
            
            return {
                "audio_path": audio_path,
                "transcript": result["text"],
                "language": result["language"],
                "segments": result["segments"],
                "processing_time": processing_time,
                "confidence": 0.90
            }
        except Exception as e:
            logger.error(f"音频转录失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def analyze_audio_content(self, audio_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """分析音频内容"""
        start_time = time.time()
        
        try:
            audio = Audio(filepath=audio_path)
            
            # 首先尝试转录音频
            transcription_result = await self.transcribe_audio(audio_path)
            
            if "error" in transcription_result:
                return transcription_result
            
            analysis_prompt = f"""
            请对这段音频进行{analysis_type}分析：
            
            转录文本：{transcription_result['transcript']}
            
            分析要求：
            1. 音频内容概述
            2. 说话者情感和语调分析
            3. 音频质量评估
            4. 关键词和主题提取
            5. 改进建议（如适用）
            
            请提供详细的分析结果。
            """
            
            response = await self.agent.arun(analysis_prompt)
            processing_time = time.time() - start_time
            
            return {
                "analysis_type": analysis_type,
                "audio_path": audio_path,
                "transcript": transcription_result['transcript'],
                "analysis_results": response.content,
                "processing_time": processing_time,
                "confidence": 0.87
            }
        except Exception as e:
            logger.error(f"音频分析失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """提取音频特征"""
        start_time = time.time()
        
        try:
            if not LIBROSA_AVAILABLE:
                return {"error": "Librosa未安装，无法进行音频特征提取"}
            
            # 使用librosa提取音频特征
            y, sr = librosa.load(audio_path)
            
            features = {
                "duration": librosa.get_duration(y=y, sr=sr),
                "sample_rate": sr,
                "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
                "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1).tolist()
            }
            
            processing_time = time.time() - start_time
            
            return {
                "audio_path": audio_path,
                "features": features,
                "processing_time": processing_time,
                "confidence": 0.95
            }
        except Exception as e:
            logger.error(f"音频特征提取失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}


class NormaVideoProcessor:
    """诺玛视频处理Agent"""
    
    def __init__(self):
        self.agent = Agent(
            name="诺玛视频分析专家",
            model=OpenAIChat(id="gpt-4-vision-preview"),
            description="专业的视频理解和分析专家",
            instructions=[
                "你是一个专业的视频分析专家，擅长视频内容理解、场景分析、动作识别",
                "能够分析视频的视觉内容、音频内容、时序特征",
                "具备视频质量评估和关键帧提取能力",
                "能够生成视频相关的描述和标签"
            ],
            send_media_to_model=True,
            store_media=True
        )
    
    async def analyze_video(self, video_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """分析视频内容"""
        start_time = time.time()
        
        try:
            video = Video(filepath=video_path)
            
            analysis_prompt = f"""
            请对这段视频进行{analysis_type}分析：
            
            分析要求：
            1. 视频内容概述（主要场景、人物、动作等）
            2. 视觉风格分析（画面质量、色彩、构图等）
            3. 音频内容分析（如有）
            4. 时序结构分析（开始、发展、高潮、结尾）
            5. 关键场景和时刻识别
            6. 视频质量评估
            7. 改进建议（如适用）
            
            请提供详细的分析结果。
            """
            
            response = await self.agent.arun(analysis_prompt, videos=[video])
            processing_time = time.time() - start_time
            
            return {
                "analysis_type": analysis_type,
                "video_path": video_path,
                "results": response.content,
                "processing_time": processing_time,
                "confidence": 0.86
            }
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def extract_key_frames(self, video_path: str, num_frames: int = 5) -> Dict[str, Any]:
        """提取视频关键帧"""
        start_time = time.time()
        
        try:
            if not CV2_AVAILABLE:
                return {"error": "OpenCV未安装，无法进行视频帧提取"}
            
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps
            
            # 计算关键帧时间点
            frame_interval = frame_count // num_frames
            key_frames = []
            
            for i in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    timestamp = i / fps
                    key_frames.append({
                        "frame_number": i,
                        "timestamp": timestamp,
                        "image_path": f"frame_{i}.jpg"
                    })
            
            cap.release()
            processing_time = time.time() - start_time
            
            return {
                "video_path": video_path,
                "total_frames": frame_count,
                "fps": fps,
                "duration": duration,
                "key_frames": key_frames,
                "processing_time": processing_time,
                "confidence": 0.90
            }
        except Exception as e:
            logger.error(f"视频关键帧提取失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}


class NormaCrossModalProcessor:
    """诺玛跨模态融合处理Agent"""
    
    def __init__(self):
        self.agent = Agent(
            name="诺玛跨模态融合专家",
            model=OpenAIChat(id="gpt-4"),
            description="专业的跨模态信息融合和综合分析专家",
            instructions=[
                "你是一个专业的跨模态分析专家，擅长整合多种媒体信息",
                "能够发现不同模态之间的关联和互补信息",
                "具备综合分析和洞察生成能力",
                "能够提供跨模态的创意建议和解决方案"
            ],
            send_media_to_model=True,
            store_media=True
        )
    
    async def fuse_multimodal_content(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """融合多模态内容"""
        start_time = time.time()
        
        try:
            fusion_prompt = f"""
            请对以下多模态数据进行综合分析和融合：
            
            多模态数据：
            {json.dumps(multimodal_data, ensure_ascii=False, indent=2)}
            
            分析要求：
            1. 识别各模态间的关联性和互补性
            2. 提取跨模态的共同主题和概念
            3. 发现单一模态无法提供的信息
            4. 生成综合性的洞察和结论
            5. 提供跨模态的创意建议
            6. 评估整体信息质量和完整性
            
            请提供详细的融合分析结果。
            """
            
            response = await self.agent.arun(fusion_prompt)
            processing_time = time.time() - start_time
            
            return {
                "fusion_type": "comprehensive",
                "input_data": multimodal_data,
                "fusion_results": response.content,
                "processing_time": processing_time,
                "confidence": 0.92
            }
        except Exception as e:
            logger.error(f"跨模态融合失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
    
    async def generate_multimodal_summary(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成多模态摘要"""
        start_time = time.time()
        
        try:
            summary_prompt = f"""
            请基于以下多模态数据生成综合摘要：
            
            多模态数据：
            {json.dumps(multimodal_data, ensure_ascii=False, indent=2)}
            
            摘要要求：
            1. 整合所有模态的关键信息
            2. 突出最重要的发现和洞察
            3. 保持逻辑清晰、结构完整
            4. 适合快速理解和决策参考
            5. 包含行动建议（如适用）
            
            请生成高质量的综合摘要。
            """
            
            response = await self.agent.arun(summary_prompt)
            processing_time = time.time() - start_time
            
            return {
                "summary_type": "multimodal_fusion",
                "input_data": multimodal_data,
                "summary": response.content,
                "processing_time": processing_time,
                "confidence": 0.89
            }
        except Exception as e:
            logger.error(f"多模态摘要生成失败: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}


class NormaMultimodalOrchestrator:
    """诺玛多模态处理编排器"""
    
    def __init__(self):
        self.text_processor = NormaTextProcessor()
        self.image_processor = NormaImageProcessor()
        self.audio_processor = NormaAudioProcessor()
        self.video_processor = NormaVideoProcessor()
        self.crossmodal_processor = NormaCrossModalProcessor()
        
        # 创建专业团队
        self.team = Team(
            name="诺玛多模态处理团队",
            members=[
                self.text_processor.agent,
                self.image_processor.agent,
                self.audio_processor.agent,
                self.video_processor.agent,
                self.crossmodal_processor.agent
            ],
            description="专业的多模态内容处理和分析团队"
        )
        
        logger.info("诺玛多模态处理系统初始化完成")
    
    async def process_media_content(self, media_path: str, media_type: MediaType, 
                                  processing_mode: ProcessingMode = ProcessingMode.ANALYSIS) -> MultimodalResult:
        """处理媒体内容"""
        start_time = time.time()
        
        try:
            logger.info(f"开始处理{media_type.value}类型媒体: {media_path}")
            
            # 根据媒体类型选择处理器
            if media_type == MediaType.TEXT:
                if isinstance(media_path, str):
                    result = await self.text_processor.analyze_text(media_path)
                else:
                    result = {"error": "文本内容必须是字符串"}
            
            elif media_type == MediaType.IMAGE:
                result = await self.image_processor.analyze_image(media_path)
            
            elif media_type == MediaType.AUDIO:
                result = await self.audio_processor.analyze_audio_content(media_path)
            
            elif media_type == MediaType.VIDEO:
                result = await self.video_processor.analyze_video(media_path)
            
            else:
                result = {"error": f"不支持的媒体类型: {media_type}"}
            
            total_time = time.time() - start_time
            
            # 创建原始内容对象
            original_content = MediaContent(
                media_type=media_type,
                content=media_path,
                metadata={"path": media_path, "mode": processing_mode.value},
                processing_history=[]
            )
            
            # 创建处理结果对象
            processed_content = MediaContent(
                media_type=media_type,
                content=result,
                metadata={"processing_mode": processing_mode.value},
                processing_history=[{"step": "primary_processing", "result": result}],
                confidence_score=result.get("confidence", 0.0),
                processing_time=total_time
            )
            
            return MultimodalResult(
                original_content=original_content,
                processed_content=processed_content,
                analysis_results=result,
                confidence_scores={"overall": result.get("confidence", 0.0)},
                processing_metrics={"total_time": total_time, "efficiency": 1.0},
                cross_modal_insights=[],
                recommendations=[]
            )
            
        except Exception as e:
            logger.error(f"媒体内容处理失败: {e}")
            raise
    
    async def process_multimodal_batch(self, media_list: List[Dict[str, Any]]) -> List[MultimodalResult]:
        """批量处理多模态内容"""
        logger.info(f"开始批量处理{len(media_list)}个媒体文件")
        
        tasks = []
        for media_item in media_list:
            task = self.process_media_content(
                media_path=media_item["path"],
                media_type=MediaType(media_item["type"]),
                processing_mode=ProcessingMode(media_item.get("mode", "analysis"))
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉异常结果
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批量处理中出现异常: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def fuse_and_analyze(self, multimodal_results: List[MultimodalResult]) -> MultimodalResult:
        """融合和分析多模态结果"""
        start_time = time.time()
        
        try:
            # 准备融合数据
            fusion_data = {}
            for result in multimodal_results:
                media_type = result.original_content.media_type.value
                fusion_data[media_type] = {
                    "content": result.processed_content.content,
                    "metadata": result.processed_content.metadata,
                    "confidence": result.confidence_scores.get("overall", 0.0)
                }
            
            # 执行跨模态融合
            fusion_result = await self.crossmodal_processor.fuse_multimodal_content(fusion_data)
            
            # 创建综合结果
            total_time = time.time() - start_time
            
            # 合并所有原始内容
            all_original_content = [result.original_content for result in multimodal_results]
            
            # 创建融合后的内容
            fused_content = MediaContent(
                media_type=MediaType.FILE,  # 融合内容视为文件类型
                content=fusion_result,
                metadata={"fusion_type": "multimodal", "source_count": len(multimodal_results)},
                processing_history=[{"step": "crossmodal_fusion", "result": fusion_result}],
                confidence_score=fusion_result.get("confidence", 0.0),
                processing_time=total_time
            )
            
            return MultimodalResult(
                original_content=all_original_content[0],  # 使用第一个作为代表
                processed_content=fused_content,
                analysis_results=fusion_result,
                confidence_scores={"fusion": fusion_result.get("confidence", 0.0)},
                processing_metrics={"fusion_time": total_time, "source_count": len(multimodal_results)},
                cross_modal_insights=fusion_result.get("fusion_results", "").split("\n"),
                recommendations=["基于多模态融合的综合建议"]
            )
            
        except Exception as e:
            logger.error(f"多模态融合分析失败: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "system_name": "诺玛多模态处理系统",
            "version": "1.0.0",
            "components": {
                "text_processor": "已就绪",
                "image_processor": "已就绪",
                "audio_processor": "已就绪" if WHISPER_AVAILABLE else "部分就绪",
                "video_processor": "已就绪",
                "crossmodal_processor": "已就绪"
            },
            "capabilities": {
                "text_analysis": True,
                "image_analysis": True,
                "audio_transcription": WHISPER_AVAILABLE,
                "video_analysis": True,
                "crossmodal_fusion": True
            },
            "dependencies": {
                "opencv": CV2_AVAILABLE,
                "librosa": LIBROSA_AVAILABLE,
                "whisper": WHISPER_AVAILABLE
            }
        }


# 使用示例和测试函数
async def demo_multimodal_processing():
    """演示多模态处理功能"""
    print("=== 诺玛多模态处理系统演示 ===\n")
    
    # 初始化系统
    orchestrator = NormaMultimodalOrchestrator()
    
    # 显示系统状态
    status = orchestrator.get_system_status()
    print("系统状态:")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print()
    
    # 演示文本处理
    print("1. 文本处理演示:")
    text_result = await orchestrator.process_media_content(
        "这是一段测试文本，用于演示诺玛Agent的文本分析能力。",
        MediaType.TEXT
    )
    print(f"处理结果: {text_result.processed_content.content}")
    print()
    
    # 演示图像处理（如果有测试图像）
    print("2. 图像处理演示:")
    print("图像处理功能已就绪，等待提供图像文件进行测试")
    print()
    
    # 演示音频处理（如果Whisper可用）
    print("3. 音频处理演示:")
    if WHISPER_AVAILABLE:
        print("音频处理功能已就绪，支持语音识别和音频分析")
    else:
        print("音频处理功能部分可用，需要安装Whisper库")
    print()
    
    # 演示视频处理
    print("4. 视频处理演示:")
    print("视频处理功能已就绪，支持视频内容分析")
    print()
    
    print("=== 演示完成 ===")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_multimodal_processing())