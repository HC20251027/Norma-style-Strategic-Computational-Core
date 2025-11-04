"""
统一LLM接口
提供标准化的LLM调用接口
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..core.orchestrator import LLMOrchestrator, LLMRequest, LLMResponse, RequestPriority
from ..core.router import ModelRouter, RoutingStrategy
from ..multimodal.text_processor import TextProcessor, TextProcessingConfig
from ..multimodal.image_processor import ImageProcessor, ImageProcessingConfig
from ..multimodal.audio_processor import AudioProcessor, AudioProcessingConfig
from ..multimodal.video_processor import VideoProcessor, VideoProcessingConfig
from ..streaming.stream_manager import StreamManager, StreamConfig, StreamEventType
from ..config import ModelType, config

logger = logging.getLogger(__name__)

class LLMRequestType(Enum):
    """LLM请求类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"

@dataclass
class LLMInterfaceConfig:
    """LLM接口配置"""
    default_model: str = "gpt-3.5-turbo"
    default_timeout: float = 30.0
    max_retries: int = 3
    enable_streaming: bool = True
    enable_caching: bool = True
    enable_preprocessing: bool = True
    auto_model_selection: bool = True
    fallback_enabled: bool = True

class LLMInterface:
    """统一LLM接口"""
    
    def __init__(self, config: LLMInterfaceConfig = None):
        self.config = config or LLMInterfaceConfig()
        
        # 初始化核心组件
        self.orchestrator = LLMOrchestrator()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.stream_manager = StreamManager()
        
        logger.info("LLM接口初始化完成")
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成文本
        
        Args:
            prompt: 提示词
            model: 模型名称
            parameters: 模型参数
            **kwargs: 额外参数
            
        Returns:
            LLM响应
        """
        try:
            # 预处理文本
            if self.config.enable_preprocessing:
                processed_text = await self.text_processor.process_text(prompt)
                prompt = processed_text['processed_text']
            
            # 路由到合适的模型
            if not model and self.config.auto_model_selection:
                model = self.orchestrator.router.get_best_model_for_type(ModelType.TEXT)
            
            model = model or self.config.default_model
            
            # 创建请求
            request = LLMRequest(
                id=f"text_{int(asyncio.get_event_loop().time() * 1000)}",
                prompt=prompt,
                model_type=ModelType.TEXT,
                preferred_models=[model] if model else None,
                parameters=parameters or {},
                metadata=kwargs
            )
            
            # 处理请求
            response = await self.orchestrator.process_request(request)
            
            logger.info(f"文本生成完成: 模型={response.model_used}, 长度={len(str(response.content))}")
            return response
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            raise
    
    async def generate_image_description(
        self,
        image_data: Union[bytes, str],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        生成图像描述
        
        Args:
            image_data: 图像数据
            prompt: 提示词
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            LLM响应
        """
        try:
            # 处理图像
            processed_image = await self.image_processor.process_image(image_data)
            
            # 构建多模态请求
            media_data = {
                'image': processed_image['image'],
                'metadata': processed_image['metadata']
            }
            
            # 路由到多模态模型
            if not model and self.config.auto_model_selection:
                model = self.orchestrator.router.get_best_model_for_type(ModelType.MULTIMODAL)
            
            model = model or self.config.default_model
            
            # 创建请求
            request = LLMRequest(
                id=f"image_desc_{int(asyncio.get_event_loop().time() * 1000)}",
                prompt=prompt or "描述这张图片",
                model_type=ModelType.MULTIMODAL,
                media_data=media_data,
                preferred_models=[model] if model else None,
                parameters=kwargs
            )
            
            # 处理请求
            response = await self.orchestrator.process_request(request)
            
            logger.info(f"图像描述生成完成: 模型={response.model_used}")
            return response
            
        except Exception as e:
            logger.error(f"图像描述生成失败: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, str],
        language: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        转录音频
        
        Args:
            audio_data: 音频数据
            language: 语言
            **kwargs: 额外参数
            
        Returns:
            LLM响应
        """
        try:
            # 处理音频
            processed_audio = await self.audio_processor.process_audio(audio_data)
            
            # 提取转录文本
            transcription = processed_audio.get('transcription')
            if not transcription:
                # 如果音频处理器没有转录，使用LLM进行转录
                transcription = await self._transcribe_with_llm(processed_audio['audio'], language)
            
            response = LLMResponse(
                id=f"audio_transcript_{int(asyncio.get_event_loop().time() * 1000)}",
                content=transcription,
                model_used="audio_processor",
                metadata={
                    'audio_metadata': processed_audio['metadata'],
                    'language': language
                }
            )
            
            logger.info(f"音频转录完成: 时长={processed_audio['metadata'].duration}s")
            return response
            
        except Exception as e:
            logger.error(f"音频转录失败: {e}")
            raise
    
    async def analyze_video(
        self,
        video_data: Union[bytes, str],
        prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        分析视频
        
        Args:
            video_data: 视频数据
            prompt: 分析提示词
            **kwargs: 额外参数
            
        Returns:
            LLM响应
        """
        try:
            # 处理视频
            processed_video = await self.video_processor.process_video(video_data)
            
            # 构建多模态请求
            media_data = {
                'video': processed_video['video'],
                'metadata': processed_video['metadata'],
                'thumbnail': processed_video.get('thumbnail'),
                'key_frames': processed_video.get('key_frames', [])
            }
            
            # 路由到多模态模型
            model = self.orchestrator.router.get_best_model_for_type(ModelType.MULTIMODAL)
            
            # 创建请求
            request = LLMRequest(
                id=f"video_analysis_{int(asyncio.get_event_loop().time() * 1000)}",
                prompt=prompt or "分析这个视频的内容",
                model_type=ModelType.MULTIMODAL,
                media_data=media_data,
                preferred_models=[model] if model else None,
                parameters=kwargs
            )
            
            # 处理请求
            response = await self.orchestrator.process_request(request)
            
            logger.info(f"视频分析完成: 模型={response.model_used}, 时长={processed_video['metadata'].duration}s")
            return response
            
        except Exception as e:
            logger.error(f"视频分析失败: {e}")
            raise
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        聊天完成
        
        Args:
            messages: 消息列表
            model: 模型名称
            parameters: 模型参数
            **kwargs: 额外参数
            
        Returns:
            LLM响应
        """
        try:
            # 将消息转换为提示词
            prompt = self._format_chat_messages(messages)
            
            # 生成文本
            response = await self.generate_text(
                prompt=prompt,
                model=model,
                parameters=parameters,
                **kwargs
            )
            
            logger.info(f"聊天完成: 消息数={len(messages)}, 模型={response.model_used}")
            return response
            
        except Exception as e:
            logger.error(f"聊天完成失败: {e}")
            raise
    
    async def stream_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        流式生成文本
        
        Args:
            prompt: 提示词
            model: 模型名称
            parameters: 模型参数
            **kwargs: 额外参数
            
        Returns:
            流式文本响应
        """
        try:
            # 创建流连接
            request_id = f"stream_{int(asyncio.get_event_loop().time() * 1000)}"
            connection_id = await self.stream_manager.create_connection(request_id)
            
            # 预处理文本
            if self.config.enable_preprocessing:
                processed_text = await self.text_processor.process_text(prompt)
                prompt = processed_text['processed_text']
            
            # 路由模型
            if not model and self.config.auto_model_selection:
                model = self.orchestrator.router.get_best_model_for_type(ModelType.TEXT)
            
            model = model or self.config.default_model
            
            # 创建请求
            request = LLMRequest(
                id=request_id,
                prompt=prompt,
                model_type=ModelType.TEXT,
                preferred_models=[model] if model else None,
                parameters=parameters or {},
                metadata=kwargs
            )
            
            # 模拟流式响应
            async def text_generator():
                # 这里应该调用实际的LLM流式API
                # 简化实现：分词发送
                words = prompt.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.1)
            
            # 开始流式响应
            await self.stream_manager.stream_response(connection_id, text_generator(), request_id)
            
            return connection_id
            
        except Exception as e:
            logger.error(f"流式文本生成失败: {e}")
            raise
    
    async def batch_process(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[LLMResponse]:
        """
        批量处理请求
        
        Args:
            requests: 请求列表
            max_concurrent: 最大并发数
            
        Returns:
            响应列表
        """
        try:
            # 转换请求格式
            llm_requests = []
            for req_data in requests:
                request_type = req_data.get('type', 'text')
                
                if request_type == 'text':
                    llm_request = LLMRequest(
                        id=req_data.get('id', f"batch_text_{len(llm_requests)}"),
                        prompt=req_data['prompt'],
                        model_type=ModelType.TEXT,
                        parameters=req_data.get('parameters', {}),
                        metadata=req_data.get('metadata', {})
                    )
                elif request_type == 'multimodal':
                    llm_request = LLMRequest(
                        id=req_data.get('id', f"batch_multi_{len(llm_requests)}"),
                        prompt=req_data['prompt'],
                        model_type=ModelType.MULTIMODAL,
                        media_data=req_data.get('media_data'),
                        parameters=req_data.get('parameters', {}),
                        metadata=req_data.get('metadata', {})
                    )
                else:
                    raise ValueError(f"不支持的请求类型: {request_type}")
                
                llm_requests.append(llm_request)
            
            # 批量处理
            responses = await self.orchestrator.batch_process_requests(llm_requests, max_concurrent)
            
            logger.info(f"批量处理完成: {len(requests)}个请求")
            return responses
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            raise
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化聊天消息"""
        formatted_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_parts.append(f"系统: {content}")
            elif role == 'user':
                formatted_parts.append(f"用户: {content}")
            elif role == 'assistant':
                formatted_parts.append(f"助手: {content}")
        
        return "\n".join(formatted_parts)
    
    async def _transcribe_with_llm(self, audio_data: bytes, language: Optional[str] = None) -> str:
        """使用LLM转录音频"""
        # 这里应该实现音频到文本的转换
        # 简化实现：返回模拟转录
        return "这是音频转录的模拟结果"
    
    def get_model_list(self, model_type: Optional[ModelType] = None) -> List[str]:
        """获取模型列表"""
        return config.get_available_models(model_type)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'interface_config': asdict(self.config),
            'orchestrator_status': self.orchestrator.get_system_status(),
            'stream_manager_stats': self.stream_manager.get_stats(),
            'available_models': self.get_model_list(),
            'processors': {
                'text_processor': self.text_processor.get_config(),
                'image_processor': self.image_processor.get_config(),
                'audio_processor': self.audio_processor.get_config(),
                'video_processor': self.video_processor.get_config()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        orchestrator_health = await self.orchestrator.health_check()
        stream_health = await self.stream_manager.health_check()
        
        return {
            'interface_healthy': True,
            'orchestrator_health': orchestrator_health,
            'stream_health': stream_health,
            'timestamp': asyncio.get_event_loop().time()
        }
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"LLM接口配置已更新: {kwargs}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return asdict(self.config)
    
    async def shutdown(self):
        """关闭接口"""
        logger.info("开始关闭LLM接口")
        
        await self.orchestrator.shutdown()
        await self.stream_manager.shutdown()
        
        logger.info("LLM接口已关闭")