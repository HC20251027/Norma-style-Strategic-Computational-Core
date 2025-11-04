"""
文本到工具映射器

负责将自然语言文本解析为具体的工具调用
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from .models import ToolCall, PipelineRequest, SpeechToTextResult


class TextToToolMapper:
    """文本到工具映射器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.max_tools_per_request = self.config.get('max_tools_per_request', 5)
        self.timeout_seconds = self.config.get('timeout_seconds', 30)
        
        # 工具定义和映射规则
        self.tools = self._load_tool_definitions()
        self.intent_patterns = self._load_intent_patterns()
        self.entity_patterns = self._load_entity_patterns()
    
    def _load_tool_definitions(self) -> Dict[str, Dict]:
        """加载工具定义"""
        return {
            'weather': {
                'description': '查询天气信息',
                'parameters': {
                    'location': {'type': 'str', 'required': True},
                    'date': {'type': 'str', 'required': False}
                },
                'keywords': ['天气', '气温', '下雨', '晴天', '温度']
            },
            'search': {
                'description': '搜索信息',
                'parameters': {
                    'query': {'type': 'str', 'required': True},
                    'max_results': {'type': 'int', 'required': False}
                },
                'keywords': ['搜索', '查找', '找', '查询']
            },
            'calculator': {
                'description': '执行计算',
                'parameters': {
                    'expression': {'type': 'str', 'required': True}
                },
                'keywords': ['计算', '等于', '加', '减', '乘', '除', '+', '-', '*', '/']
            },
            'time': {
                'description': '查询时间',
                'parameters': {},
                'keywords': ['时间', '几点', '日期', '今天', '明天']
            },
            'system_info': {
                'description': '获取系统信息',
                'parameters': {
                    'info_type': {'type': 'str', 'required': False}
                },
                'keywords': ['系统', '信息', '状态', '监控']
            },
            'network_check': {
                'description': '网络检查',
                'parameters': {
                    'target': {'type': 'str', 'required': False}
                },
                'keywords': ['网络', '连接', 'ping', '检查']
            },
            'file_operation': {
                'description': '文件操作',
                'parameters': {
                    'operation': {'type': 'str', 'required': True},
                    'path': {'type': 'str', 'required': True}
                },
                'keywords': ['文件', '创建', '删除', '读取', '写入']
            },
            'translation': {
                'description': '翻译文本',
                'parameters': {
                    'text': {'type': 'str', 'required': True},
                    'target_language': {'type': 'str', 'required': True}
                },
                'keywords': ['翻译', '英文', '中文', '日语', '韩语']
            }
        }
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """加载意图识别模式"""
        return {
            'weather_query': [
                r'.*天气.*',
                r'.*气温.*',
                r'.*(下雨|晴天|多云).*',
                r'.*今天.*天气.*',
                r'.*明天.*天气.*'
            ],
            'search_query': [
                r'.*搜索.*',
                r'.*查找.*',
                r'.*找.*',
                r'.*查询.*'
            ],
            'calculation': [
                r'.*计算.*',
                r'.*等于.*',
                r'.*(\d+[\+\-\*/]\d+).*',
                r'.*(\d+\s*[\+\-\*/]\s*\d+).*'
            ],
            'time_query': [
                r'.*时间.*',
                r'.*几点.*',
                r'.*日期.*',
                r'.*今天.*',
                r'.*现在.*'
            ],
            'system_query': [
                r'.*系统.*',
                r'.*信息.*',
                r'.*状态.*',
                r'.*监控.*'
            ],
            'network_query': [
                r'.*网络.*',
                r'.*连接.*',
                r'.*ping.*'
            ],
            'file_operation': [
                r'.*文件.*',
                r'.*创建.*',
                r'.*删除.*',
                r'.*读取.*'
            ],
            'translation': [
                r'.*翻译.*',
                r'.*英文.*',
                r'.*中文.*'
            ]
        }
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """加载实体识别模式"""
        return {
            'location': [
                r'(北京|上海|广州|深圳|杭州|南京|成都|武汉|西安|重庆)',
                r'(美国|英国|法国|德国|日本|韩国|澳大利亚)',
                r'(纽约|伦敦|巴黎|东京|首尔|悉尼)'
            ],
            'number': [
                r'\d+',
                r'\d+\.\d+'
            ],
            'date': [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{1,2}月\d{1,2}日',
                r'今天|明天|后天',
                r'昨天|前天'
            ],
            'time': [
                r'\d{1,2}点\d{0,2}分',
                r'\d{1,2}点',
                r'上午|下午|晚上'
            ]
        }
    
    async def map_text_to_tools(self, text_result: SpeechToTextResult, 
                              request: PipelineRequest) -> List[ToolCall]:
        """将文本映射为工具调用"""
        try:
            self.logger.info(f"Mapping text to tools for request {request.id}")
            
            text = text_result.text
            if not text.strip():
                self.logger.warning("Empty text provided for mapping")
                return []
            
            # 意图识别
            intent = await self._recognize_intent(text)
            self.logger.info(f"Recognized intent: {intent}")
            
            # 实体提取
            entities = await self._extract_entities(text)
            self.logger.info(f"Extracted entities: {entities}")
            
            # 工具选择
            tools = await self._select_tools(intent, entities, text)
            
            # 参数填充
            tools = await self._fill_parameters(tools, entities, text)
            
            # 验证和过滤
            tools = await self._validate_tools(tools)
            
            self.logger.info(f"Generated {len(tools)} tool calls")
            return tools[:self.max_tools_per_request]
            
        except Exception as e:
            self.logger.error(f"Text-to-tool mapping failed: {e}")
            raise
    
    async def _recognize_intent(self, text: str) -> Optional[str]:
        """识别用户意图"""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text_lower, re.IGNORECASE):
                    return intent
        
        # 如果没有匹配到具体意图，尝试基于关键词推断
        return await self._infer_intent_from_keywords(text_lower)
    
    async def _infer_intent_from_keywords(self, text: str) -> Optional[str]:
        """基于关键词推断意图"""
        tool_scores = {}
        
        for tool_name, tool_def in self.tools.items():
            score = 0
            for keyword in tool_def.get('keywords', []):
                if keyword in text:
                    score += 1
            if score > 0:
                tool_scores[tool_name] = score
        
        if tool_scores:
            # 返回得分最高的工具
            return max(tool_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    async def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取文本中的实体"""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text)
                matches.extend(found)
            if matches:
                entities[entity_type] = list(set(matches))  # 去重
        
        return entities
    
    async def _select_tools(self, intent: Optional[str], entities: Dict[str, List[str]], 
                          text: str) -> List[ToolCall]:
        """选择合适的工具"""
        tools = []
        
        if intent:
            # 基于意图选择工具
            tool_name = self._intent_to_tool(intent)
            if tool_name and tool_name in self.tools:
                tools.append(ToolCall(
                    tool_name=tool_name,
                    parameters={},
                    metadata={'intent': intent, 'confidence': 1.0}
                ))
        
        # 基于实体和关键词选择额外工具
        text_lower = text.lower()
        for tool_name, tool_def in self.tools.items():
            # 检查是否已经包含此工具
            if any(t.tool_name == tool_name for t in tools):
                continue
            
            # 检查关键词匹配
            keyword_matches = sum(1 for keyword in tool_def.get('keywords', []) 
                                if keyword in text_lower)
            
            if keyword_matches > 0:
                confidence = min(1.0, keyword_matches / len(tool_def.get('keywords', [])))
                tools.append(ToolCall(
                    tool_name=tool_name,
                    parameters={},
                    metadata={'confidence': confidence, 'reason': 'keyword_match'}
                ))
        
        # 按置信度排序
        tools.sort(key=lambda x: x.metadata.get('confidence', 0), reverse=True)
        
        return tools
    
    def _intent_to_tool(self, intent: str) -> Optional[str]:
        """将意图映射到工具名"""
        mapping = {
            'weather_query': 'weather',
            'search_query': 'search',
            'calculation': 'calculator',
            'time_query': 'time',
            'system_query': 'system_info',
            'network_query': 'network_check',
            'file_operation': 'file_operation',
            'translation': 'translation'
        }
        return mapping.get(intent)
    
    async def _fill_parameters(self, tools: List[ToolCall], entities: Dict[str, List[str]], 
                             text: str) -> List[ToolCall]:
        """填充工具参数"""
        for tool in tools:
            tool_def = self.tools.get(tool.tool_name, {})
            required_params = tool_def.get('parameters', {})
            
            for param_name, param_def in required_params.items():
                if param_def.get('required', False):
                    value = await self._extract_parameter_value(param_name, entities, text, tool)
                    if value:
                        tool.parameters[param_name] = value
        
        return tools
    
    async def _extract_parameter_value(self, param_name: str, entities: Dict[str, List[str]], 
                                     text: str, tool: ToolCall) -> Optional[Any]:
        """提取参数值"""
        # 位置实体
        if param_name == 'location':
            locations = entities.get('location', [])
            if locations:
                return locations[0]
            
            # 尝试从文本中提取地名
            city_pattern = r'(北京|上海|广州|深圳|杭州|南京|成都|武汉|西安|重庆|天津|青岛|大连|厦门|苏州|无锡|宁波|温州|佛山|东莞|中山|珠海|惠州|汕头|湛江|洛阳|开封|包头|大庆|鞍山|抚顺|吉林|齐齐哈尔|邯郸|保定|张家口|承德|廊坊|衡水|秦皇岛|唐山|邢台|德州|滨州|东营|烟台|潍坊|济宁|泰安|威海|日照|莱芜|临沂|德州|聊城|菏泽|滨州|枣庄|东营|烟台|潍坊|济宁|泰安|威海|日照|莱芜|临沂|德州|聊城|菏泽)'
            match = re.search(city_pattern, text)
            if match:
                return match.group(1)
        
        # 数字
        elif param_name in ['number', 'max_results']:
            numbers = entities.get('number', [])
            if numbers:
                try:
                    return int(float(numbers[0]))
                except ValueError:
                    pass
        
        # 日期
        elif param_name == 'date':
            dates = entities.get('date', [])
            if dates:
                return dates[0]
        
        # 时间
        elif param_name == 'time':
            times = entities.get('time', [])
            if times:
                return times[0]
        
        # 表达式
        elif param_name == 'expression':
            # 提取数学表达式
            math_pattern = r'(\d+[\+\-\*/]\d+)'
            match = re.search(math_pattern, text)
            if match:
                return match.group(1)
        
        # 查询文本
        elif param_name == 'query':
            # 提取搜索查询
            search_keywords = ['搜索', '查找', '找', '查询']
            for keyword in search_keywords:
                if keyword in text:
                    # 提取关键词后的内容
                    parts = text.split(keyword, 1)
                    if len(parts) > 1:
                        return parts[1].strip()
        
        # 翻译文本
        elif param_name == 'text':
            # 提取翻译文本
            translation_keywords = ['翻译', '翻译成']
            for keyword in translation_keywords:
                if keyword in text:
                    parts = text.split(keyword, 1)
                    if len(parts) > 1:
                        return parts[1].strip()
        
        # 目标语言
        elif param_name == 'target_language':
            languages = {
                '中文': 'zh-CN',
                '英文': 'en-US',
                '英语': 'en-US',
                '日语': 'ja-JP',
                '韩语': 'ko-KR',
                '法语': 'fr-FR',
                '德语': 'de-DE',
                '西班牙语': 'es-ES'
            }
            for lang_cn, lang_code in languages.items():
                if lang_cn in text:
                    return lang_code
        
        # 操作类型
        elif param_name == 'operation':
            operations = {
                '创建': 'create',
                '删除': 'delete',
                '读取': 'read',
                '写入': 'write',
                '复制': 'copy',
                '移动': 'move'
            }
            for op_cn, op_en in operations.items():
                if op_cn in text:
                    return op_en
        
        # 路径
        elif param_name == 'path':
            # 简单的路径提取
            path_pattern = r'[/\\][^\\s]+'
            match = re.search(path_pattern, text)
            if match:
                return match.group(0)
        
        # 信息类型
        elif param_name == 'info_type':
            info_types = {
                'CPU': 'cpu',
                '内存': 'memory',
                '磁盘': 'disk',
                '网络': 'network',
                '进程': 'process'
            }
            for info_cn, info_en in info_types.items():
                if info_cn in text:
                    return info_en
        
        return None
    
    async def _validate_tools(self, tools: List[ToolCall]) -> List[ToolCall]:
        """验证工具调用"""
        valid_tools = []
        
        for tool in tools:
            # 检查工具是否存在
            if tool.tool_name not in self.tools:
                self.logger.warning(f"Unknown tool: {tool.tool_name}")
                continue
            
            tool_def = self.tools[tool.tool_name]
            
            # 检查必需参数
            required_params = tool_def.get('parameters', {})
            missing_params = []
            
            for param_name, param_def in required_params.items():
                if param_def.get('required', False) and param_name not in tool.parameters:
                    missing_params.append(param_name)
            
            if missing_params:
                self.logger.warning(f"Missing required parameters for {tool.tool_name}: {missing_params}")
                # 如果缺少必需参数，尝试设置默认值
                for param in missing_params:
                    tool.parameters[param] = self._get_default_value(param, tool_def)
            
            # 添加工具定义信息到元数据
            tool.metadata['tool_definition'] = tool_def
            tool.metadata['validation_passed'] = len(missing_params) == 0
            
            valid_tools.append(tool)
        
        return valid_tools
    
    def _get_default_value(self, param_name: str, tool_def: Dict) -> Any:
        """获取参数默认值"""
        param_def = tool_def.get('parameters', {}).get(param_name, {})
        return param_def.get('default', None)
    
    async def get_available_tools(self) -> Dict[str, Dict]:
        """获取可用工具列表"""
        return self.tools.copy()
    
    async def add_custom_tool(self, tool_name: str, tool_definition: Dict):
        """添加自定义工具"""
        self.tools[tool_name] = tool_definition
        self.logger.info(f"Added custom tool: {tool_name}")
    
    async def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            self.logger.info(f"Removed tool: {tool_name}")
            return True
        return False
    
    async def update_tool(self, tool_name: str, tool_definition: Dict):
        """更新工具定义"""
        if tool_name in self.tools:
            self.tools[tool_name].update(tool_definition)
            self.logger.info(f"Updated tool: {tool_name}")
        else:
            self.tools[tool_name] = tool_definition
            self.logger.info(f"Added new tool: {tool_name}")
    
    async def get_tool_suggestions(self, text: str) -> List[Dict[str, Any]]:
        """获取工具建议"""
        suggestions = []
        text_lower = text.lower()
        
        for tool_name, tool_def in self.tools.items():
            score = 0
            matched_keywords = []
            
            for keyword in tool_def.get('keywords', []):
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                suggestions.append({
                    'tool_name': tool_name,
                    'description': tool_def.get('description', ''),
                    'score': score,
                    'matched_keywords': matched_keywords
                })
        
        # 按分数排序
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:5]  # 返回前5个建议