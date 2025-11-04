#!/usr/bin/env python3
"""
诺玛Agent知识库和记忆系统功能演示
展示系统的核心能力和应用场景

演示场景:
1. 知识库基础操作
2. 智能记忆管理
3. RAG检索增强生成
4. 个性化用户交互
5. 系统性能监控

作者: 皇
创建时间: 2025-11-01
版本: 1.0.0
"""

import asyncio
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any

# 导入诺玛知识记忆系统
try:
    from norma_knowledge_memory_system import (
        NormaKnowledgeMemoryOrchestrator,
        KnowledgeEntry,
        UserProfile,
        ContextMemory
    )
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 系统导入失败: {e}")
    SYSTEM_AVAILABLE = False


class NormaKnowledgeMemoryDemo:
    """诺玛知识记忆系统演示"""
    
    def __init__(self):
        """初始化演示"""
        self.system = None
        self.demo_user_id = "demo_user_001"
        self.demo_session_id = "demo_session_001"
        
    async def setup_demo(self):
        """设置演示环境"""
        print("🔧 初始化诺玛知识记忆系统演示环境...")
        
        if not SYSTEM_AVAILABLE:
            print("❌ 系统不可用，跳过演示")
            return False
        
        try:
            self.system = NormaKnowledgeMemoryOrchestrator()
            print("✅ 系统初始化成功")
            return True
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            return False
    
    async def demo_knowledge_operations(self):
        """演示知识库操作"""
        print("\n" + "="*60)
        print("📚 演示1: 知识库基础操作")
        print("="*60)
        
        try:
            # 添加知识条目
            knowledge_items = [
                {
                    "content": "Python是一种高级编程语言，具有简洁的语法和强大的功能，广泛应用于Web开发、数据科学、人工智能等领域。",
                    "metadata": {"category": "编程语言", "keywords": ["Python", "编程", "开发"], "difficulty": "入门"},
                    "source": "技术文档"
                },
                {
                    "content": "机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习规律和模式。",
                    "metadata": {"category": "人工智能", "keywords": ["机器学习", "AI", "算法"], "difficulty": "中级"},
                    "source": "学术资料"
                },
                {
                    "content": "向量数据库是专门用于存储和检索高维向量数据的数据库系统，在AI应用中发挥重要作用。",
                    "metadata": {"category": "数据库技术", "keywords": ["向量数据库", "检索", "AI"], "difficulty": "高级"},
                    "source": "技术博客"
                },
                {
                    "content": "深度学习使用多层神经网络来学习数据的复杂模式，是机器学习的重要分支。",
                    "metadata": {"category": "人工智能", "keywords": ["深度学习", "神经网络", "模式识别"], "difficulty": "高级"},
                    "source": "学术论文"
                }
            ]
            
            print("📝 添加知识条目:")
            added_entries = []
            for i, item in enumerate(knowledge_items, 1):
                entry_id = await self.system.knowledge_manager.add_knowledge(
                    item["content"],
                    metadata=item["metadata"],
                    source=item["source"]
                )
                added_entries.append(entry_id)
                print(f"  {i}. {item['metadata']['category']}: {item['content'][:50]}...")
            
            print(f"\n✅ 成功添加 {len(added_entries)} 个知识条目")
            
            # 演示搜索功能
            print("\n🔍 演示知识搜索:")
            search_queries = ["Python编程", "人工智能", "数据库技术"]
            
            for query in search_queries:
                print(f"\n搜索查询: '{query}'")
                results = await self.system.knowledge_manager.search_knowledge(query, limit=3)
                
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['content'][:80]}...")
                    print(f"     相关性: {result['relevance_score']:.2f}")
                    print(f"     来源: {result['source']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 知识库操作演示失败: {e}")
            return False
    
    async def demo_memory_management(self):
        """演示记忆管理功能"""
        print("\n" + "="*60)
        print("🧠 演示2: 智能记忆管理")
        print("="*60)
        
        try:
            # 模拟用户对话
            conversation = [
                {"role": "user", "content": "你好！我是一个刚开始学习编程的新手，想了解Python语言。"},
                {"role": "assistant", "content": "欢迎学习编程！Python是一个很好的入门语言，语法简洁易懂。"},
                {"role": "user", "content": "那Python和人工智能有什么关系吗？我对AI很感兴趣。"},
                {"role": "assistant", "content": "Python在AI领域应用非常广泛，有很多优秀的库如TensorFlow、PyTorch等。"},
                {"role": "user", "content": "听起来很有意思！能推荐一些学习资源吗？"},
                {"role": "assistant", "content": "当然可以！我推荐从基础语法开始，然后学习机器学习相关库。"}
            ]
            
            print("💬 模拟用户对话:")
            for i, message in enumerate(conversation, 1):
                role_name = "👤 用户" if message["role"] == "user" else "🤖 助手"
                print(f"  {i}. {role_name}: {message['content']}")
                
                # 存储到记忆系统
                await self.system.memory_manager.store_context_memory(
                    self.demo_session_id, self.demo_user_id, message
                )
            
            print(f"\n💾 对话已存储到记忆系统")
            
            # 获取上下文记忆
            print("\n📖 获取上下文记忆:")
            context_memory = await self.system.memory_manager.get_context_memory(self.demo_session_id)
            
            if context_memory:
                print(f"  会话摘要: {context_memory.context_summary}")
                print(f"  关键主题: {', '.join(context_memory.key_topics)}")
                print(f"  消息数量: {len(context_memory.messages)}")
            
            # 获取用户画像
            print("\n👤 获取用户画像:")
            user_profile = await self.system.memory_manager.get_user_profile(self.demo_user_id)
            
            if user_profile:
                print(f"  用户ID: {user_profile.user_id}")
                print(f"  沟通风格: {user_profile.communication_style}")
                print(f"  专业知识水平: {user_profile.expertise_level}")
                print(f"  交互次数: {len(user_profile.interaction_history)}")
            
            # 搜索相关记忆
            print("\n🔍 搜索相关记忆:")
            search_results = await self.system.memory_manager.search_context_memories(
                self.demo_user_id, "Python"
            )
            print(f"  找到 {len(search_results)} 个相关记忆")
            
            return True
            
        except Exception as e:
            print(f"❌ 记忆管理演示失败: {e}")
            return False
    
    async def demo_rag_system(self):
        """演示RAG检索增强生成系统"""
        print("\n" + "="*60)
        print("🔍 演示3: RAG检索增强生成")
        print("="*60)
        
        try:
            # RAG查询示例
            rag_queries = [
                "请介绍一下Python的特点和应用场景",
                "机器学习和深度学习有什么区别？",
                "向量数据库在AI中有什么作用？"
            ]
            
            for i, query in enumerate(rag_queries, 1):
                print(f"\n🔎 RAG查询 {i}: {query}")
                print("-" * 50)
                
                # 执行RAG检索
                retrieval_result = await self.system.rag_system.retrieve_relevant_knowledge(
                    query, self.demo_user_id, self.demo_session_id
                )
                
                print(f"📊 检索结果:")
                print(f"  找到知识源: {len(retrieval_result['knowledge_results'])} 个")
                print(f"  使用上下文: {'是' if retrieval_result['context_info'] else '否'}")
                print(f"  应用用户画像: {'是' if retrieval_result['user_profile'] else '否'}")
                
                # 显示检索到的知识源
                for j, source in enumerate(retrieval_result['knowledge_results'], 1):
                    print(f"  {j}. {source['content'][:100]}...")
                    print(f"     相关性: {source['relevance_score']:.2f}")
                    print(f"     来源: {source['source']}")
                
                # 生成增强响应
                print(f"\n🤖 生成增强响应:")
                enhanced_response = await self.system.rag_system.generate_enhanced_response(
                    query, self.demo_user_id, self.demo_session_id
                )
                
                print(f"  响应长度: {len(enhanced_response['response'])} 字符")
                print(f"  引用源数量: {len(enhanced_response['sources'])}")
                print(f"  上下文应用: {'是' if enhanced_response['context_used'] else '否'}")
                print(f"  用户画像应用: {'是' if enhanced_response['user_profile_applied'] else '否'}")
                
                # 显示响应摘要
                response_preview = enhanced_response['response'][:200] + "..." if len(enhanced_response['response']) > 200 else enhanced_response['response']
                print(f"  响应预览: {response_preview}")
            
            return True
            
        except Exception as e:
            print(f"❌ RAG系统演示失败: {e}")
            return False
    
    async def demo_personalized_interaction(self):
        """演示个性化交互"""
        print("\n" + "="*60)
        print("👤 演示4: 个性化智能交互")
        print("="*60)
        
        try:
            # 模拟不同类型用户的交互
            user_scenarios = [
                {
                    "user_id": "beginner_user",
                    "session_id": "beginner_session",
                    "input": "我刚开始学编程，应该从什么语言开始？",
                    "expected_style": "详细解释"
                },
                {
                    "user_id": "expert_user", 
                    "session_id": "expert_session",
                    "input": "最新的深度学习架构有什么突破？",
                    "expected_style": "技术深度"
                },
                {
                    "user_id": "business_user",
                    "session_id": "business_session", 
                    "input": "AI技术能为我的公司带来什么价值？",
                    "expected_style": "商业导向"
                }
            ]
            
            print("🎭 模拟不同用户类型的交互:")
            
            for scenario in user_scenarios:
                print(f"\n👤 用户类型: {scenario['expected_style']}")
                print(f"💬 输入: {scenario['input']}")
                
                # 处理用户输入
                result = await self.system.process_user_input(
                    scenario["input"],
                    scenario["user_id"],
                    scenario["session_id"]
                )
                
                if "response" in result:
                    response = result["response"]["response"]
                    print(f"🤖 个性化响应: {response[:150]}...")
                    print(f"📊 响应特点: 基于用户画像和上下文生成")
                else:
                    print("❌ 响应生成失败")
            
            # 展示用户画像分析
            print(f"\n📈 用户画像分析:")
            for scenario in user_scenarios:
                user_profile = await self.system.memory_manager.get_user_profile(scenario["user_id"])
                if user_profile:
                    print(f"  {scenario['user_id']}:")
                    print(f"    沟通风格: {user_profile.communication_style}")
                    print(f"    专业水平: {user_profile.expertise_level}")
            
            return True
            
        except Exception as e:
            print(f"❌ 个性化交互演示失败: {e}")
            return False
    
    async def demo_system_monitoring(self):
        """演示系统监控"""
        print("\n" + "="*60)
        print("📊 演示5: 系统性能监控")
        print("="*60)
        
        try:
            # 获取系统统计信息
            print("📈 获取系统统计信息...")
            system_stats = await self.system.get_system_stats()
            
            if "error" not in system_stats:
                # 知识库统计
                knowledge_stats = system_stats.get("knowledge_base", {})
                print(f"\n📚 知识库统计:")
                print(f"  总条目数: {knowledge_stats.get('total_entries', 0)}")
                print(f"  内容类型分布: {knowledge_stats.get('content_types', {})}")
                print(f"  总访问次数: {knowledge_stats.get('total_access_count', 0)}")
                print(f"  平均相关性分数: {knowledge_stats.get('average_relevance_score', 0):.2f}")
                
                # 记忆系统统计
                memory_stats = system_stats.get("memory_system", {})
                print(f"\n🧠 记忆系统统计:")
                print(f"  上下文记忆数: {memory_stats.get('context_memories_count', 0)}")
                print(f"  用户画像数: {memory_stats.get('user_profiles_count', 0)}")
                print(f"  总交互次数: {memory_stats.get('total_interactions', 0)}")
                
                # RAG系统统计
                rag_stats = system_stats.get("rag_system", {})
                print(f"\n🔍 RAG系统统计:")
                print(f"  处理查询数: {rag_stats.get('total_queries_processed', 0)}")
                print(f"  平均每个查询的源数: {rag_stats.get('average_sources_per_query', 0):.1f}")
                
                # 系统运行时间
                print(f"\n⏰ 系统信息:")
                print(f"  运行时间: {system_stats.get('system_uptime', 'N/A')}")
                
                # 展示最受欢迎的知识
                most_accessed = knowledge_stats.get("most_accessed", [])
                if most_accessed:
                    print(f"\n🔥 最受欢迎的知识:")
                    for i, item in enumerate(most_accessed[:3], 1):
                        print(f"  {i}. {item.get('content', '')[:60]}...")
                        print(f"     访问次数: {item.get('access_count', 0)}")
                
                # 展示最新添加的知识
                recent_entries = knowledge_stats.get("recent_entries", [])
                if recent_entries:
                    print(f"\n🆕 最新添加的知识:")
                    for i, item in enumerate(recent_entries[:3], 1):
                        print(f"  {i}. {item.get('content', '')[:60]}...")
                        print(f"     添加时间: {item.get('created_at', 'N/A')}")
            
            else:
                print(f"❌ 获取系统统计失败: {system_stats['error']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 系统监控演示失败: {e}")
            return False
    
    async def cleanup_demo(self):
        """清理演示环境"""
        print("\n🧹 清理演示环境...")
        
        try:
            if self.system:
                await self.system.cleanup()
                print("✅ 系统清理完成")
        except Exception as e:
            print(f"⚠️ 清理过程中出现警告: {e}")
    
    async def run_complete_demo(self):
        """运行完整演示"""
        print("🚀 诺玛Agent知识库和记忆系统功能演示")
        print("="*80)
        print("本演示将展示系统的核心能力和应用场景")
        print()
        
        # 设置演示环境
        setup_success = await self.setup_demo()
        if not setup_success:
            return
        
        # 运行各个演示模块
        demo_modules = [
            ("知识库基础操作", self.demo_knowledge_operations),
            ("智能记忆管理", self.demo_memory_management),
            ("RAG检索增强生成", self.demo_rag_system),
            ("个性化智能交互", self.demo_personalized_interaction),
            ("系统性能监控", self.demo_system_monitoring)
        ]
        
        demo_results = []
        
        for module_name, demo_func in demo_modules:
            print(f"\n🎯 开始演示模块: {module_name}")
            start_time = time.time()
            
            try:
                success = await demo_func()
                duration = time.time() - start_time
                demo_results.append({
                    "module": module_name,
                    "status": "SUCCESS" if success else "FAILED",
                    "duration": round(duration, 2)
                })
                
                status_icon = "✅" if success else "❌"
                print(f"{status_icon} {module_name}演示完成 (用时: {duration:.2f}秒)")
                
            except Exception as e:
                duration = time.time() - start_time
                demo_results.append({
                    "module": module_name,
                    "status": "ERROR",
                    "duration": round(duration, 2),
                    "error": str(e)
                })
                print(f"❌ {module_name}演示出错: {e}")
        
        # 清理演示环境
        await self.cleanup_demo()
        
        # 生成演示报告
        self.generate_demo_report(demo_results)
    
    def generate_demo_report(self, demo_results):
        """生成演示报告"""
        print("\n" + "="*80)
        print("📋 演示结果总结")
        print("="*80)
        
        total_modules = len(demo_results)
        successful_modules = sum(1 for result in demo_results if result["status"] == "SUCCESS")
        failed_modules = total_modules - successful_modules
        
        print(f"总演示模块: {total_modules}")
        print(f"成功模块: {successful_modules}")
        print(f"失败模块: {failed_modules}")
        print(f"成功率: {(successful_modules/total_modules)*100:.1f}%")
        
        print(f"\n详细结果:")
        for result in demo_results:
            status_icon = {
                "SUCCESS": "✅",
                "FAILED": "❌", 
                "ERROR": "⚠️"
            }.get(result["status"], "❓")
            
            print(f"{status_icon} {result['module']}: {result['status']}")
            print(f"   用时: {result['duration']}秒")
            if result["status"] == "ERROR":
                print(f"   错误: {result.get('error', 'Unknown error')}")
        
        # 保存演示报告
        report_data = {
            "demo_timestamp": datetime.now().isoformat(),
            "system_version": "1.0.0",
            "total_modules": total_modules,
            "successful_modules": successful_modules,
            "failed_modules": failed_modules,
            "success_rate": (successful_modules/total_modules)*100,
            "demo_results": demo_results,
            "features_demonstrated": [
                "向量数据库集成",
                "动态知识更新和学习", 
                "上下文记忆管理",
                "个性化用户画像",
                "RAG检索增强生成",
                "系统性能监控"
            ]
        }
        
        report_file = "/workspace/docs/诺玛知识记忆系统演示报告_2025-11-01.md"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 诺玛Agent知识库和记忆系统演示报告\n\n")
            f.write(f"**演示时间**: {report_data['demo_timestamp']}\n\n")
            f.write(f"**系统版本**: {report_data['system_version']}\n\n")
            f.write(f"**演示统计**:\n")
            f.write(f"- 总演示模块: {report_data['total_modules']}\n")
            f.write(f"- 成功模块: {report_data['successful_modules']}\n")
            f.write(f"- 失败模块: {report_data['failed_modules']}\n")
            f.write(f"- 成功率: {report_data['success_rate']:.1f}%\n\n")
            
            f.write("## 演示功能\n\n")
            for feature in report_data['features_demonstrated']:
                f.write(f"- {feature}\n")
            f.write("\n")
            
            f.write("## 详细演示结果\n\n")
            for result in report_data['demo_results']:
                f.write(f"### {result['module']}\n\n")
                f.write(f"**状态**: {result['status']}\n\n")
                f.write(f"**耗时**: {result['duration']}秒\n\n")
                if result["status"] == "ERROR":
                    f.write(f"**错误**: {result.get('error', 'Unknown error')}\n\n")
                f.write("---\n\n")
            
            f.write("## 演示结论\n\n")
            if report_data['success_rate'] >= 80:
                f.write("✅ 演示成功完成，系统功能正常运行。\n\n")
                f.write("诺玛Agent知识库和记忆系统展现出强大的能力：\n")
                f.write("- 高效的向量数据库操作\n")
                f.write("- 智能的记忆管理和用户画像\n")
                f.write("- 精准的RAG检索增强生成\n")
                f.write("- 完善的系统监控和性能统计\n")
            else:
                f.write("⚠️ 部分演示模块失败，需要进一步调试和优化。\n\n")
                f.write("建议检查：\n")
                f.write("1. 系统依赖和环境配置\n")
                f.write("2. 网络连接和模型下载\n")
                f.write("3. 内存和存储资源\n")
        
        print(f"\n💾 演示报告已保存至: {report_file}")


async def main():
    """主函数"""
    demo = NormaKnowledgeMemoryDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())