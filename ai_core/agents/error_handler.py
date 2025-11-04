"""
流水线错误处理器

负责处理流水线执行过程中的各种错误情况
"""

import asyncio
import logging
import traceback
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import json

from .models import ErrorType, ProcessingStage, PipelineStatus


class PipelineErrorHandler:
    """流水线错误处理器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_counts = {}
        self.fallback_handlers = {}
        self.retry_strategies = {}
        
        # 注册默认错误处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认错误处理器"""
        self.register_fallback_handler(ErrorType.SPEECH_RECOGNITION_ERROR, self._handle_speech_error)
        self.register_fallback_handler(ErrorType.TEXT_PARSING_ERROR, self._handle_text_error)
        self.register_fallback_handler(ErrorType.TOOL_MAPPING_ERROR, self._handle_tool_mapping_error)
        self.register_fallback_handler(ErrorType.TOOL_EXECUTION_ERROR, self._handle_tool_execution_error)
        self.register_fallback_handler(ErrorType.TEXT_TO_SPEECH_ERROR, self._handle_tts_error)
        self.register_fallback_handler(ErrorType.PIPELINE_ERROR, self._handle_pipeline_error)
        self.register_fallback_handler(ErrorType.SYSTEM_ERROR, self._handle_system_error)
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理错误的主入口"""
        error_type = self._classify_error(error)
        request_id = context.get('request_id', 'unknown')
        
        self.logger.error(f"Handling error for request {request_id}: {error_type.value}")
        self.logger.error(f"Error details: {str(error)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # 更新错误计数
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # 尝试重试
        retry_count = context.get('retry_count', 0)
        if retry_count < self.max_retries:
            return await self._attempt_retry(error, error_type, context)
        
        # 重试失败，使用fallback处理
        return await self._handle_with_fallback(error, error_type, context)
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """分类错误类型"""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # 语音识别错误
        if any(keyword in error_message for keyword in ['audio', 'speech', 'recognition', 'whisper']):
            return ErrorType.SPEECH_RECOGNITION_ERROR
        
        # 文本解析错误
        elif any(keyword in error_message for keyword in ['parse', 'text', 'nlp', 'intent']):
            return ErrorType.TEXT_PARSING_ERROR
        
        # 工具映射错误
        elif any(keyword in error_message for keyword in ['tool', 'mapping', 'function']):
            return ErrorType.TOOL_MAPPING_ERROR
        
        # 工具执行错误
        elif any(keyword in error_message for keyword in ['execute', 'run', 'call']):
            return ErrorType.TOOL_EXECUTION_ERROR
        
        # 文本转语音错误
        elif any(keyword in error_message for keyword in ['tts', 'speech', 'audio', 'voice']):
            return ErrorType.TEXT_TO_SPEECH_ERROR
        
        # 流水线错误
        elif any(keyword in error_message for keyword in ['pipeline', 'stage', 'flow']):
            return ErrorType.PIPELINE_ERROR
        
        # 系统错误
        else:
            return ErrorType.SYSTEM_ERROR
    
    async def _attempt_retry(self, error: Exception, error_type: ErrorType, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """尝试重试"""
        retry_count = context.get('retry_count', 0)
        request_id = context.get('request_id', 'unknown')
        
        self.logger.info(f"Retrying request {request_id} (attempt {retry_count + 1}/{self.max_retries})")
        
        # 等待重试延迟
        await asyncio.sleep(self.retry_delay * (2 ** retry_count))  # 指数退避
        
        # 更新重试计数
        context['retry_count'] = retry_count + 1
        
        return {
            'action': 'retry',
            'error_type': error_type,
            'retry_count': retry_count + 1,
            'context': context
        }
    
    async def _handle_with_fallback(self, error: Exception, error_type: ErrorType, 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """使用fallback处理错误"""
        request_id = context.get('request_id', 'unknown')
        
        self.logger.warning(f"Using fallback for request {request_id} with error type {error_type.value}")
        
        # 调用注册的fallback处理器
        fallback_handler = self.fallback_handlers.get(error_type)
        if fallback_handler:
            try:
                result = await fallback_handler(error, context)
                return result
            except Exception as fallback_error:
                self.logger.error(f"Fallback handler failed: {fallback_error}")
        
        # 默认fallback处理
        return await self._default_fallback(error, error_type, context)
    
    async def _default_fallback(self, error: Exception, error_type: ErrorType, 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """默认fallback处理"""
        request_id = context.get('request_id', 'unknown')
        stage = context.get('current_stage', ProcessingStage.RECEIVED)
        
        # 根据错误类型和阶段生成合适的响应
        fallback_messages = {
            ErrorType.SPEECH_RECOGNITION_ERROR: "抱歉，我没有听清您的话，请再说一遍。",
            ErrorType.TEXT_PARSING_ERROR: "抱歉，我没有理解您的意思，请重新表达。",
            ErrorType.TOOL_MAPPING_ERROR: "抱歉，暂时无法处理您的请求，请稍后重试。",
            ErrorType.TOOL_EXECUTION_ERROR: "抱歉，执行过程中出现了问题，请稍后重试。",
            ErrorType.TEXT_TO_SPEECH_ERROR: "抱歉，语音合成出现问题，请稍后重试。",
            ErrorType.PIPELINE_ERROR: "抱歉，系统暂时不可用，请稍后重试。",
            ErrorType.SYSTEM_ERROR: "抱歉，系统遇到了问题，请稍后重试。"
        }
        
        fallback_message = fallback_messages.get(error_type, "抱歉，发生了未知错误，请稍后重试。")
        
        return {
            'action': 'fallback',
            'error_type': error_type,
            'fallback_message': fallback_message,
            'stage': stage,
            'request_id': request_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def register_fallback_handler(self, error_type: ErrorType, handler: Callable):
        """注册fallback处理器"""
        self.fallback_handlers[error_type] = handler
        self.logger.info(f"Registered fallback handler for {error_type.value}")
    
    def register_retry_strategy(self, error_type: ErrorType, strategy: Callable):
        """注册重试策略"""
        self.retry_strategies[error_type] = strategy
        self.logger.info(f"Registered retry strategy for {error_type.value}")
    
    # 默认fallback处理器实现
    async def _handle_speech_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """语音识别错误fallback"""
        return {
            'action': 'fallback',
            'error_type': ErrorType.SPEECH_RECOGNITION_ERROR,
            'fallback_message': "抱歉，我没有听清您的话。请检查麦克风设置，或者大声一点说话。",
            'suggestions': [
                "请检查麦克风是否正常工作",
                "请确保说话声音清晰",
                "请尝试重新录制"
            ],
            'context': context
        }
    
    async def _handle_text_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """文本解析错误fallback"""
        return {
            'action': 'fallback',
            'error_type': ErrorType.TEXT_PARSING_ERROR,
            'fallback_message': "抱歉，我没有理解您的意思。请用更简单的语言重新表达。",
            'suggestions': [
                "请使用更简单的词汇",
                "请明确说明您的需求",
                "请一次只说一个指令"
            ],
            'context': context
        }
    
    async def _handle_tool_mapping_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """工具映射错误fallback"""
        return {
            'action': 'fallback',
            'error_type': ErrorType.TOOL_MAPPING_ERROR,
            'fallback_message': "抱歉，我暂时无法处理这个请求。请尝试其他指令。",
            'suggestions': [
                "请尝试使用其他指令",
                "请稍后重试",
                "请查看可用功能列表"
            ],
            'context': context
        }
    
    async def _handle_tool_execution_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """工具执行错误fallback"""
        return {
            'action': 'fallback',
            'error_type': ErrorType.TOOL_EXECUTION_ERROR,
            'fallback_message': "抱歉，执行过程中出现了问题。请检查参数或稍后重试。",
            'suggestions': [
                "请检查输入参数是否正确",
                "请稍后重试",
                "请联系技术支持"
            ],
            'context': context
        }
    
    async def _handle_tts_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """文本转语音错误fallback"""
        return {
            'action': 'fallback',
            'error_type': ErrorType.TEXT_TO_SPEECH_ERROR,
            'fallback_message': "抱歉，语音合成出现问题。我将以文本形式回复您。",
            'return_text_only': True,
            'context': context
        }
    
    async def _handle_pipeline_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """流水线错误fallback"""
        return {
            'action': 'fallback',
            'error_type': ErrorType.PIPELINE_ERROR,
            'fallback_message': "抱歉，系统暂时不可用。请稍后重试。",
            'retry_suggested': True,
            'context': context
        }
    
    async def _handle_system_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """系统错误fallback"""
        return {
            'action': 'fallback',
            'error_type': ErrorType.SYSTEM_ERROR,
            'fallback_message': "抱歉，系统遇到了问题。请联系技术支持。",
            'error_id': f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'context': context
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_breakdown': {error_type.value: count for error_type, count in self.error_counts.items()},
            'error_rate': total_errors / max(1, sum(1 for _ in [1])),  # 这里应该使用实际的总请求数
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_error_counts(self):
        """重置错误计数"""
        self.error_counts.clear()
        self.logger.info("Error counts reset")
    
    async def create_error_report(self, request_id: str, error: Exception, 
                                context: Dict[str, Any]) -> str:
        """创建错误报告"""
        report = {
            'request_id': request_id,
            'error_type': self._classify_error(error).value,
            'error_message': str(error),
            'error_traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'error_counts': self.get_error_statistics()
        }
        
        return json.dumps(report, indent=2, ensure_ascii=False)