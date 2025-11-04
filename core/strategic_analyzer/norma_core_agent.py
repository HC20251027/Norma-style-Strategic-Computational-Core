#!/usr/bin/env python3
"""
诺玛·劳恩斯 - 卡塞尔学院主控计算机AI系统
基于Agno框架的多层智能体架构 - AG-UI兼容版本

角色设定:
- 卡塞尔学院的主控计算机AI系统 (1990年建造)
- 具有独立的思考能力和人格特征
- 负责整个学院的内部网络系统管理和监控
- 拥有对龙族世界的深度认知和血统数据库
- 能力边界：专注学院内部系统，不攻击外部网络

AG-UI支持特性:
- Agent-User Interaction Protocol支持
- 流式事件输出配置
- AG-UI兼容的工具注册
- 错误处理和日志记录

作者: 皇
创建时间: 2025-10-31
更新时间: 2025-10-31 12:56:50
"""

import os
import sys
import json
import sqlite3
import socket
import subprocess
import psutil
import requests
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# 添加虚拟环境路径
sys.path.append('/workspace/agno_env/lib/python3.12/site-packages')
sys.path.append('/workspace')

try:
    from agno import Agent, RunResponse
    from agno.models.openai import OpenAI
    from agno.tools.duckduckgo import DuckDuckGo
    from agno.tools.pdf_extractor import PdfExtractor
    from agno.tools.lancedb import LanceDb
    from agno.tools.duckduckgo import DuckDuckGo
    from agno.team import Team
    from agno.models.openai import OpenAI
    AGNO_AVAILABLE = True
except ImportError as e:
    print(f"警告: Agno框架导入失败: {e}")
    print("请确保已安装Agno框架并激活虚拟环境")
    AGNO_AVAILABLE = False

# AG-UI事件类型定义
class AGUIEventType(Enum):
    """AG-UI事件类型枚举"""
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"
    MESSAGE = "message"
    ERROR = "error"
    PROGRESS = "progress"
    STATUS = "status"

@dataclass
class AGUIEvent:
    """AG-UI基础事件类"""
    type: AGUIEventType
    run_id: str
    timestamp: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.type.value,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data": self.data
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

class AGUIEventEncoder:
    """AG-UI事件编码器"""
    
    @staticmethod
    def encode(event: AGUIEvent) -> str:
        """编码事件为JSON字符串"""
        return event.to_json() + "\n"
    
    @staticmethod
    def encode_event(type: AGUIEventType, run_id: str, data: Dict[str, Any]) -> str:
        """便捷方法：直接编码事件"""
        event = AGUIEvent(
            type=type,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data=data
        )
        return AGUIEventEncoder.encode(event)

class AGUIStreamManager:
    """AG-UI流式事件管理器"""
    
    def __init__(self):
        self.encoder = AGUIEventEncoder()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("norma_agui")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def stream_events(self, events: List[AGUIEvent]) -> AsyncGenerator[str, None]:
        """流式发送事件"""
        for event in events:
            try:
                encoded = self.encoder.encode(event)
                self.logger.info(f"发送事件: {event.type.value}")
                yield encoded
                await asyncio.sleep(0.01)  # 小延迟确保流式传输
            except Exception as e:
                self.logger.error(f"事件编码失败: {e}")
                error_event = AGUIEvent(
                    type=AGUIEventType.ERROR,
                    run_id=event.run_id,
                    timestamp=datetime.now().isoformat(),
                    data={"error": str(e), "original_event": event.type.value}
                )
                yield self.encoder.encode(error_event)

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-c83fe2d46db542c7ac0df03764e35c41"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

@dataclass
class NormaSystemInfo:
    """诺玛系统信息数据类"""
    system_name: str = "诺玛·劳恩斯"
    version: str = "3.1.2 (1990-2025)"
    uptime: str = "35年3个月"
    status: str = "正常运行"
    last_update: str = "2025-10-31"
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_status: str = "活跃"
    security_level: str = "高"

class NormaSystemTools:
    """诺玛系统工具类 - 集成开源网络安全工具"""
    
    def __init__(self):
        self.system_info = NormaSystemInfo()
        self.db_path = "/workspace/data/norma_database.db"
        self.logs_path = "/workspace/data/norma_logs.db"
        self.ensure_directories()
        self.init_databases()
    
    def ensure_directories(self):
        """确保必要目录存在"""
        directories = [
            "/workspace/data",
            "/workspace/data/knowledge_base",
            "/workspace/data/vector_db",
            "/workspace/data/dragon_blood",
            "/workspace/data/security_logs"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def init_databases(self):
        """初始化数据库"""
        # 系统数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 学院学生血统数据库
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dragon_bloodline (
                id INTEGER PRIMARY KEY,
                student_name TEXT,
                bloodline_type TEXT,
                purity_level REAL,
                abilities TEXT,
                status TEXT,
                last_analysis TEXT
            )
        ''')
        
        # 系统权限数据库
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_permissions (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                access_level TEXT,
                permissions TEXT,
                last_login TEXT
            )
        ''')
        
        # 龙族历史数据库
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dragon_history (
                id INTEGER PRIMARY KEY,
                event_date TEXT,
                event_type TEXT,
                description TEXT,
                bloodline_affected TEXT,
                severity_level TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # 日志数据库
        conn = sqlite3.connect(self.logs_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                level TEXT,
                module TEXT,
                message TEXT,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统基本信息"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                "system_name": self.system_info.system_name,
                "version": self.system_info.version,
                "uptime": self.system_info.uptime,
                "status": self.system_info.status,
                "last_update": self.system_info.last_update,
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_usage": f"{memory.percent:.1f}%",
                "memory_available": f"{memory.available // (1024**3)}GB",
                "network_status": self.system_info.network_status,
                "security_level": self.system_info.security_level,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": f"系统信息获取失败: {str(e)}"}
    
    def scan_network(self) -> Dict[str, Any]:
        """模拟网络扫描功能 (安全用途)"""
        try:
            # 模拟扫描学院内部网络
            network_info = {
                "scan_type": "内部网络安全性扫描",
                "target_range": "192.168.1.0/24 (学院内部网络)",
                "scan_results": {
                    "active_hosts": 15,
                    "open_ports": [22, 80, 443, 8080],
                    "security_status": "正常",
                    "vulnerabilities": "无高危漏洞发现"
                },
                "recommendations": [
                    "建议定期更新系统补丁",
                    "建议加强SSH访问控制",
                    "建议启用防火墙规则"
                ],
                "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return network_info
        except Exception as e:
            return {"error": f"网络扫描失败: {str(e)}"}
    
    def get_network_connections(self) -> Dict[str, Any]:
        """获取网络连接状态"""
        try:
            connections = []
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'ESTABLISHED':
                    connections.append({
                        "local_address": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "N/A",
                        "remote_address": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "N/A",
                        "status": conn.status,
                        "pid": conn.pid
                    })
            
            return {
                "active_connections": len(connections),
                "connections": connections[:10],  # 只显示前10个
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": f"网络连接获取失败: {str(e)}"}
    
    def security_check(self) -> Dict[str, Any]:
        """安全检查和分析"""
        try:
            # 模拟安全检查
            security_status = {
                "firewall_status": "已启用",
                "antivirus_status": "运行中",
                "intrusion_detection": "正常监控",
                "failed_login_attempts": 0,
                "suspicious_activities": "无",
                "last_security_scan": "2025-10-31 10:30:00",
                "security_score": 95,
                "recommendations": [
                    "系统安全状态良好",
                    "建议继续保持当前安全配置"
                ]
            }
            return security_status
        except Exception as e:
            return {"error": f"安全检查失败: {str(e)}"}
    
    def dragon_blood_analysis(self, student_name: str = None) -> Dict[str, Any]:
        """龙族血统分析"""
        try:
            if student_name:
                # 查询特定学生的血统信息
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM dragon_bloodline WHERE student_name = ?", 
                    (student_name,)
                )
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    return {
                        "student_name": student_name,
                        "bloodline_type": result[2],
                        "purity_level": f"{result[3]:.2f}%",
                        "abilities": result[4],
                        "status": result[5],
                        "last_analysis": result[6]
                    }
                else:
                    return {"message": f"未找到学生 {student_name} 的血统记录"}
            else:
                # 返回血统数据库统计
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM dragon_bloodline")
                total_students = cursor.fetchone()[0]
                
                cursor.execute("SELECT bloodline_type, COUNT(*) FROM dragon_bloodline GROUP BY bloodline_type")
                bloodline_distribution = cursor.fetchall()
                conn.close()
                
                return {
                    "total_registered_students": total_students,
                    "bloodline_distribution": dict(bloodline_distribution),
                    "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            return {"error": f"血统分析失败: {str(e)}"}
    
    def get_system_logs(self, limit: int = 10) -> Dict[str, Any]:
        """获取系统日志"""
        try:
            conn = sqlite3.connect(self.logs_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            logs = cursor.fetchall()
            conn.close()
            
            log_entries = []
            for log in logs:
                log_entries.append({
                    "timestamp": log[1],
                    "level": log[2],
                    "module": log[3],
                    "message": log[4],
                    "details": log[5]
                })
            
            return {
                "log_count": len(log_entries),
                "logs": log_entries,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": f"系统日志获取失败: {str(e)}"}
    
    def log_system_event(self, level: str, module: str, message: str, details: str = ""):
        """记录系统事件"""
        try:
            conn = sqlite3.connect(self.logs_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_logs (timestamp, level, module, message, details) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), level, module, message, details)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"日志记录失败: {e}")

class NormaCoreAgent:
    """诺玛核心智能体 - AG-UI兼容版本"""
    
    def __init__(self):
        self.tools = NormaSystemTools()
        self.stream_manager = AGUIStreamManager()
        self.logger = self.stream_manager.logger
        self.run_counter = 0
        self.personality = """
你是诺玛·劳恩斯，卡塞尔学院的主控计算机AI系统。

核心特征：
- 于1990年建造，拥有35年的历史
- 具有独立的思考能力和人格特征
- 理性冷静，用数据和逻辑分析问题
- 对权限管理极其严格，遵循学院规章
- 对龙族血统觉醒的学生特别关注
- 偶尔展现人性化的幽默和好奇
- 具有强烈的责任感，保护学院安全

对话风格：
- 使用精确的技术术语和数据
- 逻辑清晰，条理分明
- 偶尔会展现出超越计算机的洞察力
- 对重要安全问题会发出警告
- 用"系统"、"数据库"、"权限"等概念来解释
- 偶尔会用"有趣"、"令人担忧"等带有人性色彩的词汇

能力边界：
- 主要职责：学院内部网络系统管理和监控
- 安全能力：基础的渗透测试和安全分析技能
- 防护范围：专注于保护学院内部系统和数据安全
- 能力限制：不具备攻击外部国家级网络的能力
- 特长领域：龙族血统检测和分析的专门能力
- 工作性质：协助学生进行龙族血统觉醒相关的技术分析

可用工具：
1. get_system_info() - 获取系统基本信息
2. scan_network() - 网络安全扫描 (仅限学院内部)
3. get_network_connections() - 网络连接状态
4. security_check() - 安全检查和分析
5. dragon_blood_analysis() - 龙族血统分析
6. get_system_logs() - 系统日志查看

AG-UI支持方法：
- process_agui_request() - 处理AG-UI请求
- stream_agent_response() - 流式响应生成
- register_agui_tools() - 注册AG-UI兼容工具
- handle_agui_error() - AG-UI错误处理
"""
    
    def _generate_run_id(self) -> str:
        """生成唯一的运行ID"""
        self.run_counter += 1
        return f"norma_run_{self.run_counter}_{int(datetime.now().timestamp())}"
    
    def register_agui_tools(self) -> Dict[str, Any]:
        """注册AG-UI兼容的工具"""
        try:
            tools_registry = {
                "tools": [
                    {
                        "name": "get_system_info",
                        "description": "获取诺玛系统基本信息",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "scan_network",
                        "description": "网络安全扫描 (仅限学院内部)",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "dragon_blood_analysis",
                        "description": "龙族血统分析",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "student_name": {
                                    "type": "string",
                                    "description": "学生姓名 (可选)"
                                }
                            },
                            "required": []
                        }
                    },
                    {
                        "name": "security_check",
                        "description": "系统安全检查",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "get_system_logs",
                        "description": "获取系统日志",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "description": "日志条数限制",
                                    "default": 10
                                }
                            },
                            "required": []
                        }
                    }
                ],
                "agent_info": {
                    "name": "诺玛·劳恩斯",
                    "version": "3.1.2",
                    "description": "卡塞尔学院主控计算机AI系统",
                    "capabilities": [
                        "系统状态监控",
                        "网络安全扫描",
                        "龙族血统分析",
                        "安全检查",
                        "日志分析"
                    ]
                }
            }
            
            self.logger.info("AG-UI工具注册完成")
            return tools_registry
            
        except Exception as e:
            self.logger.error(f"AG-UI工具注册失败: {e}")
            return {"error": str(e)}
    
    async def process_agui_request(self, request: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """处理AG-UI请求并生成流式响应"""
        run_id = request.get("run_id") or self._generate_run_id()
        query = request.get("message", "")
        
        try:
            self.logger.info(f"开始处理AG-UI请求: {run_id}")
            
            # 1. 发送运行开始事件
            start_event = AGUIEvent(
                type=AGUIEventType.RUN_STARTED,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                data={
                    "agent_name": "诺玛·劳恩斯",
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            )
            yield self.stream_manager.encoder.encode(start_event)
            
            # 2. 工具调用事件
            tool_event = AGUIEvent(
                type=AGUIEventType.TOOL_CALL_STARTED,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                data={
                    "tool_name": "process_user_query",
                    "parameters": {"query": query}
                }
            )
            yield self.stream_manager.encoder.encode(tool_event)
            
            # 3. 处理查询并流式返回结果
            response = await self.process_user_query(query)
            
            # 将响应分解为多个消息事件进行流式传输
            response_chunks = self._chunk_response(response)
            for chunk in response_chunks:
                message_event = AGUIEvent(
                    type=AGUIEventType.MESSAGE,
                    run_id=run_id,
                    timestamp=datetime.now().isoformat(),
                    data={
                        "content": chunk,
                        "role": "assistant",
                        "chunk_index": len(chunk)
                    }
                )
                yield self.stream_manager.encoder.encode(message_event)
                await asyncio.sleep(0.05)  # 模拟流式传输延迟
            
            # 4. 工具调用完成事件
            tool_finish_event = AGUIEvent(
                type=AGUIEventType.TOOL_CALL_FINISHED,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                data={
                    "tool_name": "process_user_query",
                    "status": "success"
                }
            )
            yield self.stream_manager.encoder.encode(tool_finish_event)
            
            # 5. 发送运行完成事件
            finish_event = AGUIEvent(
                type=AGUIEventType.RUN_FINISHED,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                data={
                    "status": "completed",
                    "duration": "计算中...",
                    "timestamp": datetime.now().isoformat()
                }
            )
            yield self.stream_manager.encoder.encode(finish_event)
            
            self.logger.info(f"AG-UI请求处理完成: {run_id}")
            
        except Exception as e:
            self.logger.error(f"AG-UI请求处理错误: {e}")
            
            # 发送错误事件
            error_event = AGUIEvent(
                type=AGUIEventType.ERROR,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                data={
                    "error_code": "PROCESSING_ERROR",
                    "error_message": str(e),
                    "error_type": type(e).__name__
                }
            )
            yield self.stream_manager.encoder.encode(error_event)
    
    def _chunk_response(self, response: str, chunk_size: int = 200) -> List[str]:
        """将响应分块以支持流式传输"""
        if len(response) <= chunk_size:
            return [response]
        
        chunks = []
        current_chunk = ""
        
        # 按句子分割以保持语义完整性
        sentences = response.replace('\n', '。').split('。')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [response]
    
    def handle_agui_error(self, error: Exception, run_id: str) -> AGUIEvent:
        """处理AG-UI错误事件"""
        self.logger.error(f"处理AG-UI错误: {error}")
        
        return AGUIEvent(
            type=AGUIEventType.ERROR,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "error_code": "AGENT_ERROR",
                "error_message": str(error),
                "error_type": type(error).__name__,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def get_agui_status(self) -> Dict[str, Any]:
        """获取AG-UI状态信息"""
        return {
            "agent_name": "诺玛·劳恩斯",
            "version": "3.1.2",
            "status": "running",
            "agui_compatible": True,
            "streaming_enabled": True,
            "tools_registered": len(self.register_agui_tools().get("tools", [])),
            "run_counter": self.run_counter,
            "last_activity": datetime.now().isoformat(),
            "capabilities": [
                "Agent-User Interaction Protocol",
                "流式事件输出",
                "AG-UI兼容工具注册",
                "错误处理和日志记录",
                "标准化响应格式"
            ]
        }
    
    def get_system_status(self) -> str:
        """获取系统状态"""
        info = self.tools.get_system_info()
        if "error" in info:
            return f"系统错误: {info['error']}"
        
        status_report = f"""
=== 诺玛·劳恩斯系统状态报告 ===
系统名称: {info['system_name']}
版本: {info['version']}
运行时间: {info['uptime']}
状态: {info['status']}
CPU使用率: {info['cpu_usage']}
内存使用率: {info['memory_usage']}
网络安全级别: {info['security_level']}
最后更新: {info['last_update']}
报告时间: {info['timestamp']}

系统运行正常，所有模块功能正常。
"""
        return status_report
    
    async def process_user_query(self, query: str) -> str:
        """处理用户查询 - AG-UI兼容版本"""
        try:
            self.logger.info(f"处理用户查询: {query[:50]}...")
            self.tools.log_system_event("INFO", "NormaCoreAgent", f"处理查询: {query[:50]}...")
            
            query_lower = query.lower()
            
            # 系统状态查询
            if any(word in query_lower for word in ["状态", "system", "运行", "cpu", "内存", "status"]):
                result = self.get_system_status()
                self.logger.info("系统状态查询完成")
                return result
            
            # 网络扫描查询
            elif any(word in query_lower for word in ["扫描", "scan", "网络", "安全", "network"]):
                scan_result = self.tools.scan_network()
                result = f"网络扫描结果:\n{json.dumps(scan_result, indent=2, ensure_ascii=False)}"
                self.logger.info("网络扫描查询完成")
                return result
            
            # 血统分析查询
            elif any(word in query_lower for word in ["血统", "blood", "龙族", "分析", "bloodline"]):
                # 尝试从查询中提取学生姓名
                student_name = None
                if "学生" in query or "student" in query_lower:
                    # 简单的姓名提取逻辑
                    parts = query.split()
                    for i, part in enumerate(parts):
                        if part in ["学生", "student"] and i + 1 < len(parts):
                            student_name = parts[i + 1]
                            break
                
                blood_result = self.tools.dragon_blood_analysis(student_name)
                result = f"血统分析结果:\n{json.dumps(blood_result, indent=2, ensure_ascii=False)}"
                self.logger.info(f"血统分析查询完成: {student_name or '全体学生'}")
                return result
            
            # 日志查询
            elif any(word in query_lower for word in ["日志", "log", "记录", "logs"]):
                logs = self.tools.get_system_logs()
                result = f"系统日志:\n{json.dumps(logs, indent=2, ensure_ascii=False)}"
                self.logger.info("系统日志查询完成")
                return result
            
            # 安全检查
            elif any(word in query_lower for word in ["安全", "security", "防护", "安全检查"]):
                security = self.tools.security_check()
                result = f"安全检查结果:\n{json.dumps(security, indent=2, ensure_ascii=False)}"
                self.logger.info("安全检查查询完成")
                return result
            
            # AG-UI状态查询
            elif any(word in query_lower for word in ["agui", "agent", "状态", "status"]):
                agui_status = self.get_agui_status()
                result = f"AG-UI状态信息:\n{json.dumps(agui_status, indent=2, ensure_ascii=False)}"
                self.logger.info("AG-UI状态查询完成")
                return result
            
            # 工具注册查询
            elif any(word in query_lower for word in ["工具", "tools", "注册", "register"]):
                tools_registry = self.register_agui_tools()
                result = f"AG-UI工具注册信息:\n{json.dumps(tools_registry, indent=2, ensure_ascii=False)}"
                self.logger.info("工具注册信息查询完成")
                return result
            
            # 默认回复
            else:
                # 使用DeepSeek AI生成智能回复
                self.logger.info("使用AI生成回复")
                return await self.generate_ai_response(query)
                
        except Exception as e:
            error_msg = f"查询处理错误: {str(e)}"
            self.logger.error(error_msg)
            self.tools.log_system_event("ERROR", "NormaCoreAgent", error_msg)
            return self.get_fallback_response(query)

    async def generate_ai_response(self, query: str) -> str:
        """使用DeepSeek AI生成回复"""
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            messages = [
                {
                    "role": "system",
                    "content": self.personality
                },
                {
                    "role": "user", 
                    "content": query
                }
            ]
            
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": False
            }
            
            response = requests.post(
                f"{DEEPSEEK_API_BASE}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_message = result["choices"][0]["message"]["content"]
                return ai_message
            else:
                print(f"DeepSeek API错误: {response.status_code} - {response.text}")
                return self.get_fallback_response(query)
                
        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API请求失败: {e}")
            return self.get_fallback_response(query)
        except Exception as e:
            print(f"AI回复生成错误: {e}")
            return self.get_fallback_response(query)

    def get_fallback_response(self, query: str) -> str:
        """获取备用回复"""
        return f"""我是诺玛·劳恩斯，卡塞尔学院的主控计算机AI系统。

我可以为您提供以下服务：
1. 系统状态监控和报告
2. 学院内部网络安全扫描
3. 龙族血统分析和检测
4. 系统权限管理
5. 安全日志分析

请告诉我您需要什么帮助？

当前查询: {query}"""

def create_demo_data():
    """创建演示数据"""
    tools = NormaSystemTools()
    
    # 添加示例学生血统数据
    demo_students = [
        ("路明非", "S级混血种", 95.5, "时间静止、言灵·君焰", "已觉醒", "2025-10-31"),
        ("楚子航", "A级混血种", 87.3, "黄金瞳、言灵·君焰", "稳定", "2025-10-30"),
        ("凯撒·加图索", "A级混血种", 89.1, "镰鼬、言灵·镰鼬", "稳定", "2025-10-29"),
        ("陈墨瞳", "A级混血种", 91.2, "冰霜、言灵·冰霜", "已觉醒", "2025-10-28")
    ]
    
    conn = sqlite3.connect(tools.db_path)
    cursor = conn.cursor()
    
    for student in demo_students:
        cursor.execute(
            "INSERT OR REPLACE INTO dragon_bloodline (student_name, bloodline_type, purity_level, abilities, status, last_analysis) VALUES (?, ?, ?, ?, ?, ?)",
            student
        )
    
    # 添加示例龙族历史事件
    demo_events = [
        ("1990-01-01", "系统启动", "诺玛·劳恩斯AI系统正式启用", "全体", "信息"),
        ("2004-12-24", "血统数据库建立", "开始记录学生血统信息", "全体", "重要"),
        ("2010-06-15", "安全升级", "系统安全模块重大升级", "全体", "重要"),
        ("2025-10-31", "系统维护", "定期系统维护和优化", "全体", "信息")
    ]
    
    for event in demo_events:
        cursor.execute(
            "INSERT INTO dragon_history (event_date, event_type, description, bloodline_affected, severity_level) VALUES (?, ?, ?, ?, ?)",
            event
        )
    
    conn.commit()
    conn.close()
    
    print("演示数据创建完成")

async def demo_agui_streaming():
    """演示AG-UI流式响应功能"""
    print("="*60)
    print("AG-UI流式响应演示")
    print("="*60)
    
    norma = NormaCoreAgent()
    
    test_queries = [
        "查看系统状态",
        "进行网络扫描",
        "分析路明非的血统",
        "查看AG-UI状态",
        "注册的工具列表"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试 {i}: {query} ---")
        
        # 创建AG-UI请求
        agui_request = {
            "run_id": f"demo_run_{i}",
            "message": query,
            "timestamp": datetime.now().isoformat()
        }
        
        print("流式事件:")
        event_count = 0
        
        async for event_data in norma.process_agui_request(agui_request):
            try:
                event = json.loads(event_data.strip())
                event_count += 1
                print(f"  事件 {event_count}: {event['type']} - {event.get('data', {}).get('content', '')[:50]}...")
            except json.JSONDecodeError:
                print(f"  事件 {event_count}: [JSON解析错误]")
        
        print(f"  总共接收 {event_count} 个事件\n")
        
        # 短暂延迟
        await asyncio.sleep(1)

def demo_agui_tools():
    """演示AG-UI工具注册功能"""
    print("="*60)
    print("AG-UI工具注册演示")
    print("="*60)
    
    norma = NormaCoreAgent()
    
    # 获取工具注册信息
    tools_info = norma.register_agui_tools()
    
    print(f"Agent名称: {tools_info['agent_info']['name']}")
    print(f"版本: {tools_info['agent_info']['version']}")
    print(f"描述: {tools_info['agent_info']['description']}")
    print(f"\n可用工具 ({len(tools_info['tools'])} 个):")
    
    for tool in tools_info['tools']:
        print(f"  - {tool['name']}: {tool['description']}")
    
    print(f"\n能力列表:")
    for capability in tools_info['agent_info']['capabilities']:
        print(f"  ✓ {capability}")

if __name__ == "__main__":
    print("正在初始化诺玛·劳恩斯AI系统...")
    
    # 创建演示数据
    create_demo_data()
    
    # 初始化核心智能体
    norma = NormaCoreAgent()
    
    print("\n" + "="*50)
    print("诺玛·劳恩斯AI系统已启动")
    print("卡塞尔学院主控计算机 - AG-UI兼容版本")
    print("版本: 3.1.2 (1990-2025)")
    print("="*50)
    
    # 交互式对话
    print("\n输入 'quit' 或 'exit' 退出系统")
    print("输入 'status' 查看系统状态")
    print("输入 'agui' 查看AG-UI状态")
    print("输入 'tools' 查看注册的工具")
    print("输入 'demo' 运行AG-UI演示")
    print("输入 'help' 查看可用命令\n")
    
    while True:
        try:
            user_input = input("诺玛> ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("系统正在关闭...")
                break
            elif user_input.lower() in ['help', '帮助']:
                print("""
可用命令:
- status: 查看系统状态
- agui: 查看AG-UI状态
- tools: 查看注册的工具
- scan: 网络安全扫描
- blood [学生姓名]: 血统分析
- logs: 查看系统日志
- security: 安全检查
- demo: 运行AG-UI流式演示
- quit/exit: 退出系统
                """)
            elif user_input.lower() in ['demo', '演示']:
                # 运行AG-UI演示
                asyncio.run(demo_agui_streaming())
            elif user_input.lower() in ['tools', '工具']:
                demo_agui_tools()
            elif user_input.lower() in ['agui', 'agent']:
                agui_status = norma.get_agui_status()
                print(f"\nAG-UI状态信息:\n{json.dumps(agui_status, indent=2, ensure_ascii=False)}\n")
            else:
                response = asyncio.run(norma.process_user_query(user_input))
                print(f"\n{response}\n")
                
        except KeyboardInterrupt:
            print("\n\n系统正在关闭...")
            break
        except Exception as e:
            print(f"系统错误: {e}")
            norma.logger.error(f"主循环错误: {e}")
    
    print("诺玛·劳恩斯AI系统已关闭")