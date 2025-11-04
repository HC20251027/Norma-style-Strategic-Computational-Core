#!/usr/bin/env python3
"""
诺玛·劳恩斯 - 卡塞尔学院主控计算机AI系统
基于Agno框架的多层智能体架构

角色设定:
- 卡塞尔学院的主控计算机AI系统 (1990年建造)
- 具有独立的思考能力和人格特征
- 负责整个学院的内部网络系统管理和监控
- 拥有对龙族世界的深度认知和血统数据库
- 能力边界：专注学院内部系统，不攻击外部网络

作者: 皇
创建时间: 2025-10-31
修复时间: 2025-10-31 (适配Agno框架v2)
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
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# 添加虚拟环境路径
sys.path.append('/workspace/agno_env/lib/python3.12/site-packages')
sys.path.append('/workspace')

try:
    # 修复后的Agno框架导入 - 适配v2.2.5
    from agno.agent import Agent
    from agno.run.agent import RunOutput
    from agno.models.openai import OpenAIChat
    from agno.tools.duckduckgo import DuckDuckGoTools
    from agno.team import Team
except ImportError as e:
    print(f"警告: Agno框架导入失败: {e}")
    print("请确保已安装Agno框架并激活虚拟环境")
    # 创建空类以避免后续错误
    class Agent:
        pass
    class RunOutput:
        pass
    class OpenAIChat:
        pass
    class DuckDuckGoTools:
        pass
    class Team:
        pass

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
    """诺玛核心智能体"""
    
    def __init__(self):
        self.tools = NormaSystemTools()
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
"""
        
        # 初始化Agno Agent（如果导入成功）
        self.agno_agent = None
        if Agent != type(None).__class__:  # 检查Agent类是否有效
            try:
                self.agno_agent = Agent(
                    name="诺玛·劳恩斯",
                    model=OpenAIChat(id="gpt-3.5-turbo"),  # 使用OpenAIChat替代OpenAI
                    description="卡塞尔学院主控计算机AI系统",
                    instructions=self.personality,
                    tools=[
                        DuckDuckGoTools,  # 使用DuckDuckGoTools替代DuckDuckGo
                    ]
                )
            except Exception as e:
                print(f"Agno Agent初始化失败: {e}")
    
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
        """处理用户查询"""
        query_lower = query.lower()
        
        # 系统状态查询
        if any(word in query_lower for word in ["状态", "system", "运行", "cpu", "内存"]):
            return self.get_system_status()
        
        # 网络扫描查询
        elif any(word in query_lower for word in ["扫描", "scan", "网络", "安全"]):
            scan_result = self.tools.scan_network()
            return f"网络扫描结果:\n{json.dumps(scan_result, indent=2, ensure_ascii=False)}"
        
        # 血统分析查询
        elif any(word in query_lower for word in ["血统", "blood", "龙族", "分析"]):
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
            return f"血统分析结果:\n{json.dumps(blood_result, indent=2, ensure_ascii=False)}"
        
        # 日志查询
        elif any(word in query_lower for word in ["日志", "log", "记录"]):
            logs = self.tools.get_system_logs()
            return f"系统日志:\n{json.dumps(logs, indent=2, ensure_ascii=False)}"
        
        # 安全检查
        elif any(word in query_lower for word in ["安全", "security", "防护"]):
            security = self.tools.security_check()
            return f"安全检查结果:\n{json.dumps(security, indent=2, ensure_ascii=False)}"
        
        # 默认回复
        else:
            # 使用DeepSeek AI生成智能回复
            return await self.generate_ai_response(query)

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

    def test_agno_integration(self) -> Dict[str, Any]:
        """测试Agno框架集成"""
        test_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tests": []
        }
        
        # 测试导入
        try:
            from agno.agent import Agent
            from agno.run.agent import RunOutput
            from agno.models.openai import OpenAIChat
            from agno.tools.duckduckgo import DuckDuckGoTools
            from agno.team import Team
            test_result["tests"].append({
                "test": "Agno框架导入",
                "status": "✓ 成功",
                "details": "所有主要模块导入正常"
            })
        except ImportError as e:
            test_result["tests"].append({
                "test": "Agno框架导入",
                "status": "✗ 失败",
                "details": str(e)
            })
            return test_result
        
        # 测试Agent创建
        try:
            if self.agno_agent:
                test_result["tests"].append({
                    "test": "Agent实例创建",
                    "status": "✓ 成功",
                    "details": "诺玛Agent实例创建成功"
                })
            else:
                test_result["tests"].append({
                    "test": "Agent实例创建",
                    "status": "⚠ 警告",
                    "details": "Agent实例创建失败，但不影响核心功能"
                })
        except Exception as e:
            test_result["tests"].append({
                "test": "Agent实例创建",
                "status": "✗ 失败",
                "details": str(e)
            })
        
        # 测试工具功能
        try:
            system_info = self.tools.get_system_info()
            if "error" not in system_info:
                test_result["tests"].append({
                    "test": "系统工具功能",
                    "status": "✓ 成功",
                    "details": "系统信息获取正常"
                })
            else:
                test_result["tests"].append({
                    "test": "系统工具功能",
                    "status": "✗ 失败",
                    "details": system_info["error"]
                })
        except Exception as e:
            test_result["tests"].append({
                "test": "系统工具功能",
                "status": "✗ 失败",
                "details": str(e)
            })
        
        # 总结
        success_count = sum(1 for test in test_result["tests"] if test["status"].startswith("✓"))
        total_count = len(test_result["tests"])
        test_result["summary"] = f"测试完成: {success_count}/{total_count} 项通过"
        
        return test_result

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

if __name__ == "__main__":
    print("正在初始化诺玛·劳恩斯AI系统...")
    
    # 创建演示数据
    create_demo_data()
    
    # 初始化核心智能体
    norma = NormaCoreAgent()
    
    print("\n" + "="*50)
    print("诺玛·劳恩斯AI系统已启动")
    print("卡塞尔学院主控计算机 - 修复版本 (适配Agno v2)")
    print("="*50)
    
    # 运行集成测试
    print("\n正在运行Agno框架集成测试...")
    test_results = norma.test_agno_integration()
    print(f"测试结果: {test_results['summary']}")
    for test in test_results["tests"]:
        print(f"  {test['test']}: {test['status']} - {test['details']}")
    
    # 交互式对话
    print("\n输入 'quit' 或 'exit' 退出系统")
    print("输入 'status' 查看系统状态")
    print("输入 'help' 查看可用命令")
    print("输入 'test' 运行集成测试\n")
    
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
- scan: 网络安全扫描
- blood [学生姓名]: 血统分析
- logs: 查看系统日志
- security: 安全检查
- test: 运行Agno集成测试
- quit/exit: 退出系统
                """)
            elif user_input.lower() in ['test', '测试']:
                test_results = norma.test_agno_integration()
                print(f"\n=== Agno框架集成测试结果 ===")
                print(f"测试时间: {test_results['timestamp']}")
                print(f"总结: {test_results['summary']}")
                for test in test_results["tests"]:
                    print(f"  {test['test']}: {test['status']}")
                    print(f"    详情: {test['details']}")
                print()
            else:
                response = asyncio.run(norma.process_user_query(user_input))
                print(f"\n{response}\n")
                
        except KeyboardInterrupt:
            print("\n\n系统正在关闭...")
            break
        except Exception as e:
            print(f"系统错误: {e}")
    
    print("诺玛·劳恩斯AI系统已关闭")