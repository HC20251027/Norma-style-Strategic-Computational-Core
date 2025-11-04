#!/usr/bin/env python3
"""
诺玛·劳恩斯AI系统 - 主启动程序

整合所有功能模块:
1. 核心智能体系统
2. 高级功能演示
3. 交互式界面
4. 知识库管理

作者: 皇
创建时间: 2025-10-31
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# 添加代码路径
sys.path.append('/workspace/code')

from norma_core_agent import NormaCoreAgent, create_demo_data
from norma_advanced_features import NormaAdvancedFeatures, run_advanced_features_demo

class NormaMainSystem:
    """诺玛主系统类"""
    
    def __init__(self):
        self.core_agent = NormaCoreAgent()
        self.advanced_features = NormaAdvancedFeatures()
        self.system_initialized = False
    
    def initialize_system(self):
        """初始化系统"""
        print("正在初始化诺玛·劳恩斯AI系统...")
        
        # 创建演示数据
        create_demo_data()
        
        # 创建知识库示例数据
        self.create_sample_knowledge_base()
        
        self.system_initialized = True
        print("系统初始化完成")
    
    def create_sample_knowledge_base(self):
        """创建示例知识库"""
        knowledge_base_dir = Path("/workspace/data/knowledge_base")
        knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建示例知识文档
        knowledge_docs = {
            "dragon_blood_detection.txt": """
龙族血统检测技术详解

1. 基因序列分析技术
   - 基于第三代DNA测序技术
   - 检测精度达到99.7%
   - 检测时间缩短至30分钟

2. 血统纯度评估方法
   - 计算龙族基因在个体基因组中的占比
   - 纯度等级分类：A级(90%+)、B级(70-89%)、C级(50-69%)、D级(<50%)

3. 能力测试系统
   - 言灵能力测试
   - 血统觉醒程度评估
   - 潜力预测分析

4. 家族谱系追踪
   - 历史记录查询
   - 血统来源追踪
   - 遗传特征分析

注意事项：
- 检测过程需要专业设备和人员
- 结果需要多次验证确保准确性
- 涉及个人隐私信息需严格保密
            """,
            
            "security_protocols.txt": """
卡塞尔学院网络安全协议

1. 网络架构
   - 内部网络：192.168.1.0/24
   - DMZ区域：192.168.100.0/24
   - 管理网络：192.168.200.0/24

2. 安全措施
   - 防火墙规则配置
   - 入侵检测系统(IDS)
   - 定期安全扫描
   - 访问控制列表(ACL)

3. 监控机制
   - 实时网络流量监控
   - 异常行为检测
   - 日志分析和告警
   - 定期安全评估

4. 应急响应
   - 安全事件分类
   - 响应流程规范
   - 备份和恢复程序
   - 事件记录和报告

5. 权限管理
   - 用户身份认证
   - 角色权限分配
   - 最小权限原则
   - 定期权限审查
            """,
            
            "norma_capabilities.txt": """
诺玛·劳恩斯AI系统能力说明

1. 系统管理
   - 服务器监控和管理
   - 网络设备配置
   - 性能监控和优化
   - 故障诊断和修复

2. 数据处理
   - 大数据分析和处理
   - 实时数据流处理
   - 数据备份和恢复
   - 数据安全和加密

3. 智能分析
   - 机器学习算法应用
   - 预测性分析
   - 模式识别
   - 异常检测

4. 安全防护
   - 威胁检测和分析
   - 安全事件响应
   - 漏洞评估
   - 合规性检查

5. 用户支持
   - 智能客服系统
   - 技术支持
   - 培训和教育
   - 知识库管理

技术规格：
- 处理器：多核并行处理
- 内存：128GB DDR4
- 存储：10TB SSD阵列
- 网络：10Gbps以太网
- 操作系统：定制Linux系统
            """
        }
        
        # 保存知识文档
        for filename, content in knowledge_docs.items():
            file_path = knowledge_base_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"已创建 {len(knowledge_docs)} 个知识库文档")
    
    def show_main_menu(self):
        """显示主菜单"""
        print("\n" + "="*60)
        print("诺玛·劳恩斯AI系统 - 主控制台")
        print("="*60)
        print("1. 基础对话模式")
        print("2. 系统状态监控")
        print("3. 高级功能演示")
        print("4. 知识库管理")
        print("5. 龙族血统分析")
        print("6. 安全检查")
        print("7. 系统日志查看")
        print("8. 帮助信息")
        print("9. 退出系统")
        print("="*60)
    
    def interactive_mode(self):
        """交互式模式"""
        if not self.system_initialized:
            self.initialize_system()
        
        print("\n欢迎使用诺玛·劳恩斯AI系统！")
        print("输入 'menu' 返回主菜单，'help' 查看帮助，'quit' 退出")
        
        while True:
            try:
                user_input = input("\n诺玛> ").strip()
                
                if not user_input:
                    continue
                
                command = user_input.lower()
                
                # 退出命令
                if command in ['quit', 'exit', '退出', 'q']:
                    print("\n正在关闭诺玛·劳恩斯AI系统...")
                    break
                
                # 主菜单
                elif command in ['menu', '菜单', 'm']:
                    self.show_main_menu()
                    continue
                
                # 帮助信息
                elif command in ['help', '帮助', 'h']:
                    self.show_help()
                    continue
                
                # 基础对话
                elif command == '1' or command in ['对话', 'chat', 'c']:
                    self.chat_mode()
                
                # 系统状态
                elif command == '2' or command in ['状态', 'status', 's']:
                    self.show_system_status()
                
                # 高级功能演示
                elif command == '3' or command in ['高级', 'advanced', 'a']:
                    self.run_advanced_demo()
                
                # 知识库管理
                elif command == '4' or command in ['知识库', 'knowledge', 'k']:
                    self.knowledge_base_management()
                
                # 血统分析
                elif command == '5' or command in ['血统', 'blood', 'b']:
                    self.dragon_blood_analysis()
                
                # 安全检查
                elif command == '6' or command in ['安全', 'security', 'sec']:
                    self.security_check()
                
                # 系统日志
                elif command == '7' or command in ['日志', 'log', 'l']:
                    self.show_system_logs()
                
                # 直接查询处理
                else:
                    response = self.core_agent.process_user_query(user_input)
                    print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n\n正在关闭诺玛·劳恩斯AI系统...")
                break
            except Exception as e:
                print(f"系统错误: {e}")
    
    def chat_mode(self):
        """聊天模式"""
        print("\n=== 基础对话模式 ===")
        print("输入 'back' 返回主菜单")
        
        while True:
            try:
                user_input = input("\n聊天> ").strip()
                
                if user_input.lower() in ['back', '返回', 'b']:
                    break
                
                if user_input:
                    response = self.core_agent.process_user_query(user_input)
                    print(f"\n{response}\n")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")
    
    def show_system_status(self):
        """显示系统状态"""
        print("\n=== 系统状态监控 ===")
        status = self.core_agent.get_system_status()
        print(status)
        
        # 添加网络连接信息
        connections = self.core_agent.tools.get_network_connections()
        if "error" not in connections:
            print(f"\n网络连接状态:")
            print(f"活跃连接数: {connections['active_connections']}")
            if connections['connections']:
                print("主要连接:")
                for conn in connections['connections'][:3]:
                    print(f"  - {conn['local_address']} -> {conn['remote_address']}")
    
    def run_advanced_demo(self):
        """运行高级功能演示"""
        print("\n=== 高级功能演示 ===")
        demo_results = run_advanced_features_demo()
        print(f"\n演示完成，共运行 {len(demo_results)} 项功能测试")
    
    def knowledge_base_management(self):
        """知识库管理"""
        print("\n=== 知识库管理 ===")
        
        knowledge_dir = Path("/workspace/data/knowledge_base")
        if knowledge_dir.exists():
            files = list(knowledge_dir.glob("*.txt"))
            print(f"知识库文档数量: {len(files)}")
            print("文档列表:")
            for i, file in enumerate(files, 1):
                print(f"  {i}. {file.name}")
        else:
            print("知识库目录不存在")
    
    def dragon_blood_analysis(self):
        """龙族血统分析"""
        print("\n=== 龙族血统分析 ===")
        
        # 显示数据库统计
        stats = self.core_agent.tools.dragon_blood_analysis()
        print("血统数据库统计:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # 允许查询特定学生
        print("\n输入学生姓名查询详细信息 (或输入 'back' 返回):")
        while True:
            student_name = input("学生姓名> ").strip()
            
            if student_name.lower() in ['back', '返回', 'b']:
                break
            
            if student_name:
                result = self.core_agent.tools.dragon_blood_analysis(student_name)
                print(f"\n查询结果:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
    
    def security_check(self):
        """安全检查"""
        print("\n=== 安全检查 ===")
        security_result = self.core_agent.tools.security_check()
        print("安全检查结果:")
        print(json.dumps(security_result, indent=2, ensure_ascii=False))
    
    def show_system_logs(self):
        """显示系统日志"""
        print("\n=== 系统日志 ===")
        logs = self.core_agent.tools.get_system_logs(10)
        print("最近系统日志:")
        print(json.dumps(logs, indent=2, ensure_ascii=False))
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
=== 诺玛·劳恩斯AI系统帮助 ===

系统介绍:
诺玛·劳恩斯是卡塞尔学院的主控计算机AI系统，于1990年建造，
具有独立的思考能力和人格特征，负责学院内部网络系统管理、
安全监控和龙族血统检测。

主要功能:
1. 基础对话 - 与诺玛进行自然语言交互
2. 系统状态 - 监控学院内部系统运行状态
3. 高级功能 - 演示搜索、PDF处理、RAG检索、多智能体协作
4. 知识库 - 管理学院相关知识文档
5. 血统分析 - 龙族血统检测和分析
6. 安全检查 - 网络安全和系统安全监控
7. 系统日志 - 查看系统运行日志

常用命令:
- menu: 显示主菜单
- help: 显示此帮助信息
- quit/exit: 退出系统
- status: 查看系统状态
- scan: 网络安全扫描
- blood [姓名]: 血统分析
- logs: 查看系统日志
- security: 安全检查

注意事项:
- 本系统专注于学院内部网络管理
- 不具备攻击外部网络的能力
- 所有功能均基于合法合规的技术
- 涉及个人隐私信息需严格保密

技术特点:
- 基于Agno框架开发
- 集成多种开源工具
- 支持多智能体协作
- 具备向量数据库RAG功能
- 实时系统监控能力
        """
        print(help_text)

def main():
    """主函数"""
    print("正在启动诺玛·劳恩斯AI系统...")
    
    try:
        # 创建主系统实例
        norma_system = NormaMainSystem()
        
        # 显示欢迎信息
        print("\n" + "="*60)
        print("诺玛·劳恩斯AI系统 v3.1.2")
        print("卡塞尔学院主控计算机 - 现实化版本")
        print("构建时间: 1990-2025")
        print("当前时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*60)
        
        # 启动交互式系统
        norma_system.interactive_mode()
        
    except Exception as e:
        print(f"系统启动失败: {e}")
        return 1
    
    print("\n诺玛·劳恩斯AI系统已关闭")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)