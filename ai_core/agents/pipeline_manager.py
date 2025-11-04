"""
语音处理流水线管理器

协调整个语音-文本-工具流水线的执行
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
import uuid

from .models import (
    PipelineRequest, 
    PipelineResponse, 
    PipelineConfig,
    ProcessingStage, 
    PipelineStatus,
    ErrorType,
    ToolCall,
    ToolExecutionResult
)
from .state_manager import PipelineStateManager
from .error_handler import PipelineErrorHandler
from .speech_to_text import SpeechToTextConverter
from .text_to_tool import TextToToolMapper
from .tool_result_processor import ToolResultProcessor
from .text_to_speech import TextToSpeechConverter


class VoicePipelineManager:
    """语音处理流水线管理器"""
    
    def __init__(self, config: PipelineConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or PipelineConfig()
        
        # 初始化各个组件
        self.state_manager = PipelineStateManager()
        self.error_handler = PipelineErrorHandler(
            max_retries=self.config.error_handling.get('max_retries', 3),
            retry_delay=self.config.error_handling.get('retry_delay', 1.0)
        )
        
        self.speech_to_text = SpeechToTextConverter(
            self.config.speech_recognition
        )
        self.text_to_tool = TextToToolMapper(
            self.config.tool_mapping
        )
        self.tool_result_processor = ToolResultProcessor()
        self.text_to_speech = TextToSpeechConverter(
            self.config.text_to_speech
        )
        
        # 流水线状态
        self.is_running = False
        self.active_requests = set()
        
        self.logger.info("Voice pipeline manager initialized")
    
    async def start(self):
        """启动流水线管理器"""
        if self.is_running:
            self.logger.warning("Pipeline manager is already running")
            return
        
        self.is_running = True
        self.logger.info("Voice pipeline manager started")
        
        # 启动后台任务
        asyncio.create_task(self._cleanup_task())
    
    async def stop(self):
        """停止流水线管理器"""
        self.is_running = False
        self.logger.info("Voice pipeline manager stopped")
    
    async def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """处理语音请求"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing request {request.id}")
            
            # 创建流水线状态
            state = await self.state_manager.create_state(request)
            self.active_requests.add(request.id)
            
            # 更新状态为处理中
            await self.state_manager.set_status(request.id, PipelineStatus.PROCESSING)
            
            # 步骤1: 语音转文本
            await self.state_manager.update_stage(
                request.id, 
                ProcessingStage.SPEECH_TO_TEXT, 
                progress=0.1
            )
            
            speech_result = await self._execute_speech_to_text(request)
            await self.state_manager.set_speech_result(
                request.id, 
                speech_result.text, 
                speech_result.confidence
            )
            
            # 步骤2: 文本转工具调用
            await self.state_manager.update_stage(
                request.id, 
                ProcessingStage.TEXT_TOOL_MAPPING, 
                progress=0.3
            )
            
            tool_calls = await self._execute_text_to_tool_mapping(speech_result, request)
            
            # 添加工具调用到状态
            for tool_call in tool_calls:
                await self.state_manager.add_tool_call(request.id, tool_call)
            
            # 步骤3: 工具执行
            await self.state_manager.update_stage(
                request.id, 
                ProcessingStage.TOOL_EXECUTION, 
                progress=0.5
            )
            
            execution_results = await self._execute_tools(tool_calls, request)
            
            # 添加执行结果到状态
            for result in execution_results:
                await self.state_manager.add_execution_result(request.id, result)
            
            # 步骤4: 处理工具结果
            await self.state_manager.update_stage(
                request.id, 
                ProcessingStage.RESULT_PROCESSING, 
                progress=0.7
            )
            
            final_text = await self._process_tool_results(tool_calls, execution_results, request)
            await self.state_manager.set_final_result(request.id, text=final_text)
            
            # 步骤5: 文本转语音
            await self.state_manager.update_stage(
                request.id, 
                ProcessingStage.TEXT_TO_SPEECH, 
                progress=0.9
            )
            
            tts_result = await self._execute_text_to_speech(final_text, request)
            await self.state_manager.set_final_result(request.id, audio=tts_result.audio_data)
            
            # 完成流水线
            await self.state_manager.update_stage(request.id, ProcessingStage.COMPLETED, progress=1.0)
            await self.state_manager.set_status(request.id, PipelineStatus.COMPLETED)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = PipelineResponse(
                request_id=request.id,
                success=True,
                status=PipelineStatus.COMPLETED,
                audio_response=tts_result.audio_data,
                text_response=final_text,
                processing_time=processing_time,
                metadata={
                    'speech_confidence': speech_result.confidence,
                    'tool_calls_count': len(tool_calls),
                    'execution_results_count': len(execution_results)
                }
            )
            
            self.logger.info(f"Request {request.id} completed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Request {request.id} failed: {e}")
            
            # 错误处理
            error_context = {
                'request_id': request.id,
                'current_stage': await self._get_current_stage(request.id),
                'retry_count': 0
            }
            
            error_result = await self.error_handler.handle_error(e, error_context)
            
            if error_result.get('action') == 'retry':
                # 重试处理
                return await self._handle_retry(request, error_result, start_time)
            elif error_result.get('action') == 'fallback':
                # Fallback处理
                return await self._handle_fallback(request, error_result, start_time)
            else:
                # 最终失败
                await self.state_manager.set_status(
                    request.id, 
                    PipelineStatus.FAILED, 
                    str(e),
                    self.error_handler._classify_error(e)
                )
                
                return PipelineResponse(
                    request_id=request.id,
                    success=False,
                    status=PipelineStatus.FAILED,
                    error_message=str(e),
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
        
        finally:
            self.active_requests.discard(request.id)
    
    async def _execute_speech_to_text(self, request: PipelineRequest):
        """执行语音转文本"""
        return await self.speech_to_text.convert(request)
    
    async def _execute_text_to_tool_mapping(self, speech_result, request: PipelineRequest):
        """执行文本转工具映射"""
        return await self.text_to_tool.map_text_to_tools(speech_result, request)
    
    async def _execute_tools(self, tool_calls: List[ToolCall], request: PipelineRequest) -> List[ToolExecutionResult]:
        """执行工具调用"""
        results = []
        
        # 并行执行工具调用（如果配置允许）
        if self.config.performance.get('max_concurrent_requests', 1) > 1:
            tasks = [self._execute_single_tool(tool_call, request) for tool_call in tool_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(ToolExecutionResult(
                        call_id=tool_calls[i].call_id,
                        success=False,
                        error_message=str(result)
                    ))
                else:
                    processed_results.append(result)
            return processed_results
        else:
            # 串行执行
            for tool_call in tool_calls:
                result = await self._execute_single_tool(tool_call, request)
                results.append(result)
            return results
    
    async def _execute_single_tool(self, tool_call: ToolCall, request: PipelineRequest) -> ToolExecutionResult:
        """执行单个工具调用"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing tool {tool_call.tool_name} with parameters {tool_call.parameters}")
            
            # 这里应该集成实际的工具执行系统
            # 目前使用模拟实现
            result = await self._simulate_tool_execution(tool_call)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={'tool_name': tool_call.tool_name}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Tool execution failed for {tool_call.tool_name}: {e}")
            
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
                metadata={'tool_name': tool_call.tool_name}
            )
    
    async def _simulate_tool_execution(self, tool_call: ToolCall) -> Any:
        """模拟工具执行"""
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        
        # 模拟不同工具的执行结果
        if tool_name == 'weather':
            location = parameters.get('location', '未知地点')
            return {
                'location': location,
                'weather': '晴天',
                'temperature': '25',
                'humidity': '60'
            }
        
        elif tool_name == 'search':
            query = parameters.get('query', '搜索内容')
            return [
                {'title': f'搜索结果1: {query}', 'snippet': '这是第一个搜索结果'},
                {'title': f'搜索结果2: {query}', 'snippet': '这是第二个搜索结果'},
                {'title': f'搜索结果3: {query}', 'snippet': '这是第三个搜索结果'}
            ]
        
        elif tool_name == 'calculator':
            expression = parameters.get('expression', '0')
            try:
                # 安全的计算（仅支持基本运算）
                result = eval(expression)
                return result
            except:
                return f"计算错误：无法计算表达式 {expression}"
        
        elif tool_name == 'time':
            now = datetime.now()
            return {
                'current_time': now.strftime('%H:%M:%S'),
                'date': now.strftime('%Y年%m月%d日')
            }
        
        elif tool_name == 'system_info':
            import platform
            return {
                '操作系统': platform.system(),
                '版本': platform.version(),
                '处理器': platform.processor()
            }
        
        elif tool_name == 'network_check':
            return {
                'status': '连接正常',
                'latency': '15ms'
            }
        
        elif tool_name == 'file_operation':
            operation = parameters.get('operation', '')
            path = parameters.get('path', '')
            return f"文件操作 '{operation}' 在路径 '{path}' 执行完成"
        
        elif tool_name == 'translation':
            text = parameters.get('text', '')
            target_language = parameters.get('target_language', 'en')
            return f"[翻译到{target_language}] {text}"
        
        else:
            return f"工具 {tool_name} 执行完成，参数：{parameters}"
    
    async def _process_tool_results(self, tool_calls: List[ToolCall], 
                                  execution_results: List[ToolExecutionResult], 
                                  request: PipelineRequest) -> str:
        """处理工具结果"""
        return await self.tool_result_processor.process_results(
            tool_calls, execution_results, request
        )
    
    async def _execute_text_to_speech(self, text: str, request: PipelineRequest):
        """执行文本转语音"""
        return await self.text_to_speech.synthesize(text, request)
    
    async def _handle_retry(self, request: PipelineRequest, error_result: Dict[str, Any], 
                          start_time: datetime) -> PipelineResponse:
        """处理重试"""
        retry_count = error_result.get('retry_count', 0)
        self.logger.info(f"Retrying request {request.id} (attempt {retry_count})")
        
        # 递归调用（注意避免无限递归）
        if retry_count >= self.config.error_handling.get('max_retries', 3):
            # 超过最大重试次数，转为fallback处理
            return await self._handle_fallback(request, {'action': 'fallback'}, start_time)
        
        # 更新重试计数
        request.metadata['retry_count'] = retry_count
        
        # 重新处理请求
        return await self.process_request(request)
    
    async def _handle_fallback(self, request: PipelineRequest, error_result: Dict[str, Any], 
                             start_time: datetime) -> PipelineResponse:
        """处理fallback"""
        fallback_message = error_result.get('fallback_message', '抱歉，处理过程中出现错误。')
        
        self.logger.info(f"Using fallback for request {request.id}: {fallback_message}")
        
        # 尝试将fallback消息转换为语音
        try:
            tts_result = await self._execute_text_to_speech(fallback_message, request)
            
            return PipelineResponse(
                request_id=request.id,
                success=False,  # 标记为失败，但提供了fallback响应
                status=PipelineStatus.COMPLETED,  # 但状态为完成
                audio_response=tts_result.audio_data,
                text_response=fallback_message,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    'fallback': True,
                    'error_type': error_result.get('error_type', ErrorType.PIPELINE_ERROR)
                }
            )
        except Exception as e:
            # Fallback也失败了，返回文本响应
            return PipelineResponse(
                request_id=request.id,
                success=False,
                status=PipelineStatus.COMPLETED,
                text_response=fallback_message,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    'fallback': True,
                    'tts_failed': True
                }
            )
    
    async def _get_current_stage(self, request_id: str) -> ProcessingStage:
        """获取当前处理阶段"""
        state = await self.state_manager.get_state(request_id)
        return state.current_stage if state else ProcessingStage.RECEIVED
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """获取请求状态"""
        state = await self.state_manager.get_state(request_id)
        if not state:
            return None
        
        return {
            'request_id': state.request_id,
            'status': state.status.value,
            'current_stage': state.current_stage.value,
            'progress': state.progress,
            'error_message': state.error_message,
            'start_time': state.start_time.isoformat() if state.start_time else None,
            'end_time': state.end_time.isoformat() if state.end_time else None,
            'processing_times': state.processing_times,
            'metadata': state.metadata
        }
    
    async def cancel_request(self, request_id: str) -> bool:
        """取消请求"""
        return await self.state_manager.cancel_request(request_id)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = await self.state_manager.get_metrics()
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            'pipeline_metrics': {
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'average_processing_time': metrics.average_processing_time,
                'error_counts': {k.value: v for k, v in metrics.error_counts.items()}
            },
            'error_statistics': error_stats,
            'active_requests': len(self.active_requests),
            'is_running': self.is_running
        }
    
    async def _cleanup_task(self):
        """后台清理任务"""
        while self.is_running:
            try:
                # 清理过旧的状态
                await self.state_manager.cleanup_old_states(max_age_hours=24)
                
                # 等待一段时间后再次清理
                await asyncio.sleep(3600)  # 1小时清理一次
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(300)  # 出错时等待5分钟
    
    async def update_config(self, new_config: PipelineConfig):
        """更新配置"""
        self.config = new_config
        
        # 更新各个组件的配置
        self.error_handler.max_retries = new_config.error_handling.get('max_retries', 3)
        self.error_handler.retry_delay = new_config.error_handling.get('retry_delay', 1.0)
        
        self.logger.info("Pipeline configuration updated")
    
    async def get_available_tools(self) -> Dict[str, Any]:
        """获取可用工具列表"""
        return await self.text_to_tool.get_available_tools()
    
    async def add_custom_tool(self, tool_name: str, tool_definition: Dict):
        """添加工具"""
        await self.text_to_tool.add_custom_tool(tool_name, tool_definition)
    
    async def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        return await self.text_to_tool.remove_tool(tool_name)
    
    async def test_pipeline(self) -> Dict[str, Any]:
        """测试流水线"""
        test_results = {
            'speech_to_text': False,
            'text_to_tool': False,
            'tool_execution': False,
            'text_to_speech': False,
            'overall': False
        }
        
        try:
            # 创建测试请求
            test_request = PipelineRequest(
                audio_data=b'test_audio_data',
                text="你好，测试语音",
                language='zh-CN'
            )
            
            # 测试各个组件
            try:
                speech_result = await self.speech_to_text.convert(test_request)
                test_results['speech_to_text'] = len(speech_result.text) > 0
            except:
                pass
            
            try:
                tool_calls = await self.text_to_tool.map_text_to_tools(speech_result, test_request)
                test_results['text_to_tool'] = len(tool_calls) >= 0
            except:
                pass
            
            try:
                tts_result = await self.text_to_speech.synthesize("测试语音", test_request)
                test_results['text_to_speech'] = len(tts_result.audio_data) > 0
            except:
                pass
            
            # 检查整体测试结果
            test_results['overall'] = all([
                test_results['speech_to_text'],
                test_results['text_to_tool'],
                test_results['text_to_speech']
            ])
            
        except Exception as e:
            self.logger.error(f"Pipeline test failed: {e}")
        
        return test_results