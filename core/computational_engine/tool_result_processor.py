"""
工具执行结果处理器

负责处理工具执行的结果，将其转换为自然语言响应
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import re

from .models import ToolExecutionResult, ToolCall, PipelineRequest


class ToolResultProcessor:
    """工具执行结果处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.response_templates = self._load_response_templates()
        self.formatters = self._load_formatters()
    
    def _load_response_templates(self) -> Dict[str, Dict]:
        """加载响应模板"""
        return {
            'weather': {
                'success': '今天{location}的天气是{weather}，气温{temperature}度。',
                'error': '抱歉，无法获取{location}的天气信息。',
                'partial': '获取到{location}的部分天气信息：{info}'
            },
            'search': {
                'success': '为您找到了以下关于"{query}"的信息：\n{results}',
                'error': '抱歉，没有找到关于"{query}"的信息。',
                'partial': '找到了一些相关信息：\n{results}'
            },
            'calculator': {
                'success': '计算结果是：{expression} = {result}',
                'error': '抱歉，计算"{expression}"时出现错误。',
                'partial': '部分计算结果：{result}'
            },
            'time': {
                'success': '当前时间是：{current_time}',
                'error': '抱歉，无法获取当前时间。',
                'partial': '时间信息：{time_info}'
            },
            'system_info': {
                'success': '系统信息：\n{system_info}',
                'error': '抱歉，无法获取系统信息。',
                'partial': '部分系统信息：{info}'
            },
            'network_check': {
                'success': '网络连接状态：{status}',
                'error': '网络检查失败：{error}',
                'partial': '网络状态：{status}'
            },
            'file_operation': {
                'success': '文件操作"{operation}"已完成：{result}',
                'error': '文件操作失败：{error}',
                'partial': '文件操作部分完成：{result}'
            },
            'translation': {
                'success': '翻译结果：{translated_text}',
                'error': '翻译失败：{error}',
                'partial': '部分翻译：{partial_result}'
            }
        }
    
    def _load_formatters(self) -> Dict[str, callable]:
        """加载格式化函数"""
        return {
            'weather': self._format_weather_result,
            'search': self._format_search_result,
            'calculator': self._format_calculator_result,
            'time': self._format_time_result,
            'system_info': self._format_system_info_result,
            'network_check': self._format_network_result,
            'file_operation': self._format_file_operation_result,
            'translation': self._format_translation_result
        }
    
    async def process_results(self, tool_calls: List[ToolCall], 
                            execution_results: List[ToolExecutionResult],
                            request: PipelineRequest) -> str:
        """处理工具执行结果，生成最终响应"""
        try:
            self.logger.info(f"Processing {len(execution_results)} tool execution results")
            
            if not execution_results:
                return "抱歉，没有可用的结果。"
            
            # 按调用顺序处理结果
            results_by_call = {result.call_id: result for result in execution_results}
            
            responses = []
            for tool_call in tool_calls:
                if tool_call.call_id in results_by_call:
                    result = results_by_call[tool_call.call_id]
                    response = await self._process_single_result(tool_call, result, request)
                    if response:
                        responses.append(response)
            
            # 合并多个响应
            final_response = await self._merge_responses(responses, request)
            
            self.logger.info("Tool result processing completed")
            return final_response
            
        except Exception as e:
            self.logger.error(f"Tool result processing failed: {e}")
            return "抱歉，处理结果时出现错误。"
    
    async def _process_single_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                   request: PipelineRequest) -> Optional[str]:
        """处理单个工具执行结果"""
        try:
            if not result.success:
                return await self._format_error_response(tool_call, result)
            
            # 使用相应的格式化器处理结果
            formatter = self.formatters.get(tool_call.tool_name)
            if formatter:
                return await formatter(tool_call, result, request)
            else:
                return await self._format_generic_result(tool_call, result, request)
                
        except Exception as e:
            self.logger.error(f"Failed to process result for {tool_call.tool_name}: {e}")
            return await self._format_error_response(tool_call, result, str(e))
    
    async def _format_error_response(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                   error_msg: str = None) -> str:
        """格式化错误响应"""
        tool_name = tool_call.tool_name
        templates = self.response_templates.get(tool_name, {})
        
        error_message = error_msg or result.error_message or "未知错误"
        
        # 使用错误模板
        template = templates.get('error', '工具"{tool_name}"执行失败：{error}')
        
        return template.format(
            tool_name=tool_name,
            error=error_message,
            **tool_call.parameters
        )
    
    async def _format_generic_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                   request: PipelineRequest) -> str:
        """格式化通用结果"""
        tool_name = tool_call.tool_name
        
        if isinstance(result.result, dict):
            # 如果结果是字典，转换为可读格式
            formatted_result = json.dumps(result.result, ensure_ascii=False, indent=2)
        elif isinstance(result.result, (list, tuple)):
            # 如果结果是列表，转换为文本
            formatted_result = '\n'.join(str(item) for item in result.result)
        else:
            formatted_result = str(result.result)
        
        return f"工具'{tool_name}'执行成功。\n结果：{formatted_result}"
    
    # 具体的格式化器实现
    async def _format_weather_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                   request: PipelineRequest) -> str:
        """格式化天气结果"""
        location = tool_call.parameters.get('location', '未知地点')
        
        if not result.success:
            return f"抱歉，无法获取{location}的天气信息。"
        
        weather_data = result.result
        if isinstance(weather_data, dict):
            weather = weather_data.get('weather', '未知')
            temperature = weather_data.get('temperature', '未知')
            humidity = weather_data.get('humidity', '未知')
            
            return f"今天{location}的天气是{weather}，气温{temperature}度，湿度{humidity}%。"
        else:
            return f"今天{location}的天气信息：{weather_data}"
    
    async def _format_search_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                  request: PipelineRequest) -> str:
        """格式化搜索结果"""
        query = tool_call.parameters.get('query', '搜索内容')
        
        if not result.success:
            return f"抱歉，没有找到关于'{query}'的信息。"
        
        search_results = result.result
        if isinstance(search_results, list):
            if not search_results:
                return f"抱歉，没有找到关于'{query}'的相关信息。"
            
            # 限制显示结果数量
            max_results = min(5, len(search_results))
            formatted_results = []
            
            for i, item in enumerate(search_results[:max_results], 1):
                if isinstance(item, dict):
                    title = item.get('title', f'结果{i}')
                    snippet = item.get('snippet', str(item))
                    formatted_results.append(f"{i}. {title}\n   {snippet}")
                else:
                    formatted_results.append(f"{i}. {item}")
            
            result_text = '\n'.join(formatted_results)
            return f"为您找到了以下关于'{query}'的信息：\n{result_text}"
        else:
            return f"搜索结果：{search_results}"
    
    async def _format_calculator_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                      request: PipelineRequest) -> str:
        """格式化计算结果"""
        expression = tool_call.parameters.get('expression', '')
        
        if not result.success:
            return f"计算'{expression}'时出现错误。"
        
        calc_result = result.result
        return f"计算结果：{expression} = {calc_result}"
    
    async def _format_time_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                request: PipelineRequest) -> str:
        """格式化时间结果"""
        if not result.success:
            return "抱歉，无法获取当前时间。"
        
        time_data = result.result
        if isinstance(time_data, dict):
            current_time = time_data.get('current_time', '')
            date = time_data.get('date', '')
            return f"当前时间：{current_time}，日期：{date}"
        else:
            return f"当前时间：{time_data}"
    
    async def _format_system_info_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                       request: PipelineRequest) -> str:
        """格式化系统信息结果"""
        if not result.success:
            return "抱歉，无法获取系统信息。"
        
        system_info = result.result
        if isinstance(system_info, dict):
            info_lines = []
            for key, value in system_info.items():
                info_lines.append(f"{key}：{value}")
            return "系统信息：\n" + '\n'.join(info_lines)
        else:
            return f"系统信息：{system_info}"
    
    async def _format_network_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                   request: PipelineRequest) -> str:
        """格式化网络检查结果"""
        if not result.success:
            return f"网络检查失败：{result.error_message}"
        
        network_data = result.result
        if isinstance(network_data, dict):
            status = network_data.get('status', '未知')
            latency = network_data.get('latency', '未知')
            return f"网络状态：{status}，延迟：{latency}ms"
        else:
            return f"网络状态：{network_data}"
    
    async def _format_file_operation_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                          request: PipelineRequest) -> str:
        """格式化文件操作结果"""
        operation = tool_call.parameters.get('operation', '')
        path = tool_call.parameters.get('path', '')
        
        if not result.success:
            return f"文件操作'{operation}'失败：{result.error_message}"
        
        operation_result = result.result
        return f"文件操作'{operation}'已完成：{operation_result}"
    
    async def _format_translation_result(self, tool_call: ToolCall, result: ToolExecutionResult, 
                                       request: PipelineRequest) -> str:
        """格式化翻译结果"""
        if not result.success:
            return f"翻译失败：{result.error_message}"
        
        translated_text = result.result
        return f"翻译结果：{translated_text}"
    
    async def _merge_responses(self, responses: List[str], request: PipelineRequest) -> str:
        """合并多个响应"""
        if not responses:
            return "抱歉，没有可用的结果。"
        
        if len(responses) == 1:
            return responses[0]
        
        # 如果有多个结果，用适当的连接词合并
        if len(responses) == 2:
            return f"{responses[0]}\n\n另外，{responses[1]}"
        else:
            merged = responses[0]
            for i, response in enumerate(responses[1:-1], 1):
                merged += f"\n\n另外，{response}"
            merged += f"\n\n最后，{responses[-1]}"
            return merged
    
    async def add_custom_formatter(self, tool_name: str, formatter: callable):
        """添加自定义格式化器"""
        self.formatters[tool_name] = formatter
        self.logger.info(f"Added custom formatter for tool: {tool_name}")
    
    async def update_response_template(self, tool_name: str, template_type: str, template: str):
        """更新响应模板"""
        if tool_name not in self.response_templates:
            self.response_templates[tool_name] = {}
        
        self.response_templates[tool_name][template_type] = template
        self.logger.info(f"Updated response template for {tool_name}.{template_type}")
    
    async def get_response_preview(self, tool_name: str, result_data: Any) -> str:
        """获取响应预览"""
        try:
            # 创建一个模拟的工具调用和结果
            mock_tool_call = ToolCall(tool_name=tool_name, parameters={})
            mock_result = ToolExecutionResult(
                call_id="preview",
                success=True,
                result=result_data
            )
            
            formatter = self.formatters.get(tool_name)
            if formatter:
                return await formatter(mock_tool_call, mock_result, None)
            else:
                return await self._format_generic_result(mock_tool_call, mock_result, None)
                
        except Exception as e:
            self.logger.error(f"Failed to generate response preview: {e}")
            return f"预览生成失败：{str(e)}"
    
    async def validate_result_format(self, tool_name: str, result_data: Any) -> Dict[str, Any]:
        """验证结果格式"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # 检查结果是否为空
            if result_data is None:
                validation_result['warnings'].append("结果为空")
                return validation_result
            
            # 检查数据类型
            if tool_name == 'calculator':
                if not isinstance(result_data, (int, float, str)):
                    validation_result['errors'].append("计算结果应该是数字或字符串")
                    validation_result['valid'] = False
            
            elif tool_name == 'weather':
                if not isinstance(result_data, dict):
                    validation_result['errors'].("天气结果应该是字典格式")
                    validation_result['valid'] = False
                else:
                    required_fields = ['weather', 'temperature']
                    for field in required_fields:
                        if field not in result_data:
                            validation_result['warnings'].append(f"缺少建议字段：{field}")
            
            elif tool_name == 'search':
                if not isinstance(result_data, list):
                    validation_result['warnings'].append("搜索结果建议为列表格式")
            
            # 其他工具的验证逻辑可以在这里添加
            
        except Exception as e:
            validation_result['errors'].append(f"验证过程中出现错误：{str(e)}")
            validation_result['valid'] = False
        
        return validation_result