#!/usr/bin/env python3
"""
创建示例文档和测试数据

作者: 皇
创建时间: 2025-10-31
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def create_sample_documents():
    """创建示例文档"""
    
    # 创建PDF目录
    pdf_dir = Path("/workspace/data/pdf_documents")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建一些示例文本文件（模拟PDF内容）
    documents = {
        "龙族血统检测技术手册.pdf.txt": """
龙族血统检测技术手册 v2.0

第一章 概述
龙族血统检测是卡塞尔学院的核心技术之一，通过先进的基因测序技术，
能够准确识别个体中的龙族基因成分，评估血统纯度和觉醒潜力。

第二章 检测原理
2.1 基因序列分析
采用第三代DNA测序技术，对个体基因组进行全面分析，
识别特定的龙族基因标记序列。

2.2 血统纯度计算
通过计算龙族基因片段在总体基因组中的占比，
确定血统纯度等级：
- S级：纯度 ≥ 95%
- A级：纯度 85-94%
- B级：纯度 70-84%
- C级：纯度 50-69%
- D级：纯度 < 50%

第三章 检测流程
3.1 样本采集
- 采集静脉血样 5ml
- 记录基本个人信息
- 签署知情同意书

3.2 实验分析
- DNA提取和纯化
- 基因测序分析
- 数据处理和计算

3.3 结果报告
- 生成检测报告
- 专家解读分析
- 制定后续计划

第四章 准确性验证
4.1 技术精度
- 检测准确率：99.7%
- 重复性：CV < 2%
- 特异性：> 99.5%

4.2 质控措施
- 标准品对照
- 空白对照
- 阳性对照
- 阴性对照

第五章 注意事项
5.1 检测限制
- 仅适用于混血种检测
- 需要专业设备和技术人员
- 检测结果需要专业解读

5.2 伦理考虑
- 保护个人隐私
- 遵循知情同意原则
- 避免基因歧视

附录A：技术规格
附录B：参考标准
附录C：故障排除指南
        """,
        
        "网络安全协议.pdf.txt": """
卡塞尔学院网络安全协议 v3.1

第一章 网络架构
1.1 网络分段
- 教学网络：192.168.10.0/24
- 研究网络：192.168.20.0/24
- 管理网络：192.168.30.0/24
- DMZ区域：192.168.100.0/24

1.2 网络设备
- 核心交换机：2台冗余配置
- 接入交换机：按楼层分布
- 无线控制器：集中管理
- 防火墙：边界防护

第二章 安全策略
2.1 访问控制
- 基于角色的访问控制(RBAC)
- 多因素身份认证(MFA)
- 最小权限原则
- 定期权限审查

2.2 数据保护
- 数据分类和标记
- 加密传输和存储
- 备份和恢复策略
- 数据泄露防护

2.3 监控和审计
- 实时安全监控
- 日志收集和分析
- 异常行为检测
- 合规性审计

第三章 威胁防护
3.1 恶意软件防护
- 终端防护软件
- 邮件安全网关
- Web过滤网关
- 定期扫描更新

3.2 网络攻击防护
- 入侵检测系统(IDS)
- 入侵防护系统(IPS)
- DDoS防护
- 漏洞管理

3.3 内部威胁防护
- 用户行为分析
- 权限监控
- 数据丢失防护(DLP)
- 内部威胁检测

第四章 应急响应
4.1 事件分类
- 一级：重大安全事件
- 二级：重要安全事件
- 三级：一般安全事件
- 四级：安全告警

4.2 响应流程
1. 事件检测和报告
2. 影响评估和分类
3. 应急响应启动
4. 事件调查和分析
5. 系统恢复和验证
6. 事后总结和改进

4.3 沟通协调
- 内部沟通机制
- 外部协调渠道
- 媒体应对策略
- 法律合规要求

第五章 合规要求
5.1 法律法规
- 网络安全法
- 数据保护法
- 个人信息保护法
- 行业标准规范

5.2 内部政策
- 信息安全政策
- 数据管理政策
- 访问控制政策
- 事件响应政策

附录A：联系人信息
附录B：应急流程图
附录C：合规检查清单
        """,
        
        "诺玛系统技术文档.pdf.txt": """
诺玛·劳恩斯AI系统技术文档 v3.1.2

第一章 系统概述
1.1 系统简介
诺玛·劳恩斯是卡塞尔学院的主控计算机AI系统，于1990年投入使用，
经过多次升级改造，现已成为学院信息化建设的核心支撑平台。

1.2 核心特性
- 高可用性：99.99%系统可用率
- 高性能：支持10000+并发用户
- 高安全性：多层安全防护机制
- 高扩展性：模块化架构设计

第二章 技术架构
2.1 硬件架构
- 计算节点：64核CPU，128GB内存
- 存储系统：10TB SSD阵列
- 网络接口：10Gbps以太网
- 备份系统：异地容灾备份

2.2 软件架构
- 操作系统：定制Linux系统
- 数据库：PostgreSQL + Redis
- 应用框架：微服务架构
- 监控系统：Prometheus + Grafana

2.3 AI引擎
- 自然语言处理：基于Transformer架构
- 机器学习：支持多种算法框架
- 知识图谱：实体关系推理
- 决策引擎：规则引擎 + 机器学习

第三章 核心功能
3.1 系统管理
- 服务器监控和管理
- 网络设备配置
- 服务状态监控
- 性能优化调优

3.2 数据处理
- 实时数据流处理
- 大数据分析
- 数据质量管控
- 数据安全和加密

3.3 智能分析
- 异常检测算法
- 预测性分析
- 模式识别
- 智能推荐

3.4 安全防护
- 威胁检测和分析
- 安全事件响应
- 漏洞评估和修复
- 合规性检查

第四章 API接口
4.1 认证授权
- OAuth 2.0认证
- JWT令牌管理
- API密钥管理
- 权限控制

4.2 核心API
- /api/v1/system/status
- /api/v1/security/scan
- /api/v1/bloodline/analyze
- /api/v1/logs/query

4.3 响应格式
{
  "status": "success",
  "code": 200,
  "message": "操作成功",
  "data": {},
  "timestamp": "2025-10-31T11:00:00Z"
}

第五章 部署运维
5.1 部署架构
- 生产环境：主备部署
- 测试环境：单节点部署
- 开发环境：容器化部署
- 监控环境：独立部署

5.2 运维监控
- 系统性能监控
- 应用性能监控
- 业务指标监控
- 日志聚合分析

5.3 故障处理
- 自动故障检测
- 故障自动恢复
- 故障告警通知
- 故障根因分析

附录A：配置参数
附录B：API文档
附录C：故障排除
附录D：性能调优
        """
    }
    
    # 保存文档
    for filename, content in documents.items():
        file_path = pdf_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"已创建 {len(documents)} 个示例文档")
    
    # 创建测试脚本
    test_script = """#!/usr/bin/env python3
\"\"\"
诺玛系统功能测试脚本

作者: 皇
创建时间: 2025-10-31
\"\"\"

import sys
import os
sys.path.append('/workspace/code')

from norma_core_agent import NormaCoreAgent, create_demo_data
from norma_advanced_features import NormaAdvancedFeatures

def test_core_functions():
    \"\"\"测试核心功能\"\"\"
    print("=== 测试诺玛核心功能 ===")
    
    # 初始化
    create_demo_data()
    norma = NormaCoreAgent()
    
    # 测试系统状态
    print("1. 测试系统状态查询...")
    status = norma.get_system_status()
    print(status)
    
    # 测试网络扫描
    print("\\n2. 测试网络扫描...")
    scan_result = norma.tools.scan_network()
    print(f"扫描结果: {scan_result}")
    
    # 测试血统分析
    print("\\n3. 测试血统分析...")
    blood_result = norma.tools.dragon_blood_analysis("路明非")
    print(f"血统分析: {blood_result}")
    
    # 测试安全检查
    print("\\n4. 测试安全检查...")
    security_result = norma.tools.security_check()
    print(f"安全检查: {security_result}")
    
    print("\\n核心功能测试完成")

def test_advanced_features():
    \"\"\"测试高级功能\"\"\"
    print("\\n=== 测试诺玛高级功能 ===")
    
    advanced = NormaAdvancedFeatures()
    
    # 测试搜索功能
    print("1. 测试搜索功能...")
    search_result = advanced.demo_search_functionality("龙族血统检测")
    
    # 测试PDF处理
    print("\\n2. 测试PDF处理...")
    pdf_result = advanced.demo_pdf_processing()
    
    # 测试RAG检索
    print("\\n3. 测试RAG检索...")
    rag_result = advanced.demo_rag_functionality()
    
    # 测试多智能体协作
    print("\\n4. 测试多智能体协作...")
    collab_result = advanced.demo_multi_agent_collaboration()
    
    print("\\n高级功能测试完成")

if __name__ == "__main__":
    print("诺玛系统功能测试开始...")
    test_core_functions()
    test_advanced_features()
    print("\\n所有测试完成！")
"""
    
    test_file = Path("/workspace/code/test_norma_system.py")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("已创建测试脚本")

if __name__ == "__main__":
    create_sample_documents()
    print("示例文档创建完成")