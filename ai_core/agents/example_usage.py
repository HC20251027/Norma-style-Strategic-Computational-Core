"""
知识库记忆系统使用示例
演示系统的各种功能和用法
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

# 导入系统组件
from . import IntegratedMemoryKnowledgeSystem
from .config import get_merged_config, get_config_template
from .core.memory_manager import MemoryType, MemoryImportance
from .core.knowledge_graph import EntityType, RelationType
from .management.knowledge_base_manager import KnowledgeType


async def basic_usage_example():
    """基础使用示例"""
    
    print("=== 知识库记忆系统基础使用示例 ===\n")
    
    # 1. 初始化系统
    config = get_merged_config('dev')
    system = IntegratedMemoryKnowledgeSystem(config)
    
    await system.initialize()
    
    # 2. 处理用户交互
    user_input = "我喜欢机器学习和人工智能，特别是深度学习"
    context = {"user_id": "user_001", "session_id": "session_123"}
    
    result = await system.process_interaction(user_input, context)
    
    print(f"用户输入: {user_input}")
    print(f"记忆ID: {result['memory_id']}")
    print(f"相关知识数量: {len(result['related_knowledge'])}")
    print(f"推理结果: {result['reasoning_result']}")
    print(f"学习结果: {result['learning_result']}")
    
    # 3. 获取上下文记忆
    query = "机器学习"
    contextual_memory = await system.get_contextual_memory(query, limit=5)
    
    print(f"\n查询: {query}")
    print(f"语义记忆数量: {len(contextual_memory['semantic_memories'])}")
    print(f"图连接数量: {len(contextual_memory['graph_connections'])}")
    
    # 4. 获取统计信息
    memory_stats = await system.memory_manager.get_memory_stats()
    print(f"\n记忆统计: {memory_stats}")
    
    await system.close()


async def knowledge_management_example():
    """知识管理示例"""
    
    print("\n=== 知识管理示例 ===\n")
    
    config = get_merged_config('dev')
    system = IntegratedMemoryKnowledgeSystem(config)
    
    await system.initialize()
    
    # 1. 添加各类知识
    knowledge_items = [
        {
            'content': 'Python是一种高级编程语言，由Guido van Rossum创建',
            'knowledge_type': KnowledgeType.FACTUAL,
            'confidence': 0.9,
            'source': 'user_input',
            'tags': ['编程', 'Python', '语言']
        },
        {
            'content': '机器学习是人工智能的一个分支，通过算法让计算机从数据中学习',
            'knowledge_type': KnowledgeType.CONCEPTUAL,
            'confidence': 0.8,
            'source': 'user_input',
            'tags': ['机器学习', '人工智能', '算法']
        },
        {
            'content': '深度学习使用多层神经网络来学习数据的表示',
            'knowledge_type': KnowledgeType.CONCEPTUAL,
            'confidence': 0.85,
            'source': 'user_input',
            'tags': ['深度学习', '神经网络', '机器学习']
        }
    ]
    
    # 添加知识到系统
    for item in knowledge_items:
        knowledge_id = await system.knowledge_manager.add_knowledge(
            content=item['content'],
            knowledge_type=item['knowledge_type'],
            confidence=item['confidence'],
            source=item['source'],
            tags=item['tags']
        )
        print(f"添加知识: {item['content'][:30]}... -> ID: {knowledge_id}")
    
    # 2. 搜索知识
    search_results = await system.knowledge_manager.search_knowledge(
        query="Python编程",
        knowledge_types=[KnowledgeType.FACTUAL, KnowledgeType.CONCEPTUAL],
        limit=5
    )
    
    print(f"\n搜索'Python编程'的结果:")
    for result in search_results:
        print(f"  - {result['content'][:50]}... (置信度: {result['confidence']:.2f})")
    
    # 3. 获取知识库统计
    kb_stats = await system.knowledge_manager.get_knowledge_statistics()
    print(f"\n知识库统计: {json.dumps(kb_stats, indent=2, ensure_ascii=False)}")
    
    await system.close()


async def memory_and_learning_example():
    """记忆和学习示例"""
    
    print("\n=== 记忆和学习示例 ===\n")
    
    config = get_merged_config('dev')
    system = IntegratedMemoryKnowledgeSystem(config)
    
    await system.initialize()
    
    user_id = "user_001"
    
    # 1. 模拟用户交互序列
    interactions = [
        "我想了解机器学习的基础知识",
        "机器学习有哪些主要算法？",
        "我比较喜欢监督学习的方法",
        "能详细介绍一下神经网络吗？",
        "深度学习和机器学习有什么区别？",
        "我喜欢Python进行机器学习开发"
    ]
    
    # 2. 处理每个交互并学习
    for i, interaction in enumerate(interactions):
        print(f"\n交互 {i+1}: {interaction}")
        
        # 处理交互
        result = await system.process_interaction(
            user_input=interaction,
            context={"user_id": user_id, "interaction_index": i}
        )
        
        # 显示学习结果
        learning_result = result['learning_result']
        if learning_result.get('preference_updates'):
            print(f"  偏好更新: {learning_result['preference_updates']}")
        
        if learning_result.get('behavior_updates', {}).get('patterns'):
            patterns = learning_result['behavior_updates']['patterns']
            print(f"  行为模式: {patterns.get('dominant_type', 'unknown')}")
    
    # 3. 获取个性化推荐
    recommendations = await system.learning_system.get_personalized_recommendations(user_id)
    
    print(f"\n个性化推荐:")
    for rec in recommendations['recommendations']:
        print(f"  - {rec['type']}: {rec.get('topic', rec.get('style'))} (置信度: {rec['confidence']:.2f})")
    
    # 4. 获取学习统计
    learning_stats = await system.learning_system.get_learning_statistics()
    print(f"\n学习统计: {json.dumps(learning_stats, indent=2, ensure_ascii=False)}")
    
    await system.close()


async def knowledge_graph_example():
    """知识图谱示例"""
    
    print("\n=== 知识图谱示例 ===\n")
    
    config = get_merged_config('dev')
    system = IntegratedMemoryKnowledgeSystem(config)
    
    await system.initialize()
    
    # 1. 添加实体
    entities = [
        ("Python", EntityType.CONCEPT),
        ("机器学习", EntityType.CONCEPT),
        ("深度学习", EntityType.CONCEPT),
        ("神经网络", EntityType.CONCEPT),
        ("监督学习", EntityType.CONCEPT),
        ("编程语言", EntityType.CONCEPT)
    ]
    
    entity_ids = {}
    for name, entity_type in entities:
        entity_id = await system.knowledge_graph.add_entity(
            name=name,
            entity_type=entity_type,
            properties={"description": f"{name}的概念实体"}
        )
        entity_ids[name] = entity_id
        print(f"添加实体: {name} -> {entity_id}")
    
    # 2. 添加关系
    relations = [
        ("Python", "编程语言", RelationType.IS_A),
        ("机器学习", "深度学习", RelationType.IS_A),
        ("深度学习", "神经网络", RelationType.USES),
        ("监督学习", "机器学习", RelationType.IS_A),
        ("Python", "机器学习", RelationType.SUPPORTS),
        ("机器学习", "神经网络", RelationType.USES)
    ]
    
    for source, target, relation_type in relations:
        source_id = entity_ids[source]
        target_id = entity_ids[target]
        
        relation_id = await system.knowledge_graph.add_relation(
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=relation_type,
            confidence=0.8
        )
        print(f"添加关系: {source} -> {target} ({relation_type.value})")
    
    # 3. 查找连接
    connections = await system.knowledge_graph.find_connections("Python", max_depth=3)
    
    print(f"\nPython的连接关系:")
    for connection in connections[:5]:
        print(f"  {connection['source']} -> {connection['target']} (路径长度: {connection['path_length']})")
    
    # 4. 实体推理
    python_id = entity_ids["Python"]
    reasoning_result = await system.reasoning_engine.reason_about_entity(python_id)
    
    print(f"\nPython实体推理结果:")
    print(f"  推理结论数量: {len(reasoning_result.get('conclusions', []))}")
    print(f"  中心性分数: {reasoning_result.get('centrality_score', 0):.3f}")
    
    # 5. 获取图统计
    graph_stats = await system.knowledge_graph.get_graph_statistics()
    print(f"\n图统计: {json.dumps(graph_stats, indent=2, ensure_ascii=False)}")
    
    await system.close()


async def semantic_search_example():
    """语义搜索示例"""
    
    print("\n=== 语义搜索示例 ===\n")
    
    config = get_merged_config('dev')
    system = IntegratedMemoryKnowledgeSystem(config)
    
    await system.initialize()
    
    # 1. 添加一些文档到向量数据库
    documents = [
        "Python是一种简单易学的编程语言，适合初学者",
        "机器学习是人工智能的重要分支，应用广泛",
        "深度学习使用神经网络处理复杂数据",
        "自然语言处理是计算机科学的一个领域",
        "数据科学结合统计学和编程技能"
    ]
    
    # 添加到记忆系统
    for i, doc in enumerate(documents):
        memory_id = await system.memory_manager.store_long_term_memory(
            content=doc,
            memory_type=MemoryType.SEMANTIC,
            importance=MemoryImportance.MEDIUM,
            tags=['document', f'doc_{i}']
        )
        print(f"添加文档 {i+1}: {doc[:30]}...")
    
    # 2. 执行语义搜索
    queries = [
        "编程语言",
        "人工智能",
        "神经网络",
        "数据处理"
    ]
    
    for query in queries:
        print(f"\n搜索查询: '{query}'")
        
        # 语义搜索
        results = await system.semantic_search.search(
            query=query,
            collection='memories',
            top_k=3
        )
        
        for result in results:
            print(f"  - {result.content[:50]}... (分数: {result.score:.3f})")
            if result.highlights:
                print(f"    高亮: {result.highlights[0][:40]}...")
    
    # 3. 模糊搜索
    print(f"\n模糊搜索示例:")
    fuzzy_results = await system.semantic_search.fuzzy_search(
        query="程序开发",
        collection='memories',
        top_k=3
    )
    
    for result in fuzzy_results:
        print(f"  - {result.content[:50]}... (分数: {result.score:.3f})")
    
    # 4. 获取搜索分析
    analytics = await system.semantic_search.get_search_analytics()
    print(f"\n搜索分析: {json.dumps(analytics, indent=2, ensure_ascii=False)}")
    
    await system.close()


async def system_integration_example():
    """系统集成示例"""
    
    print("\n=== 系统集成示例 ===\n")
    
    # 使用性能优化配置
    config = get_config_template('performance')
    config = get_merged_config('dev', env_overrides=False)
    
    system = IntegratedMemoryKnowledgeSystem(config)
    
    await system.initialize()
    
    # 模拟一个完整的对话场景
    conversation = [
        {
            "user": "你好，我想了解人工智能",
            "assistant": "人工智能是一个非常广泛的领域，包括机器学习、深度学习、自然语言处理等多个分支。您对哪个方面比较感兴趣？"
        },
        {
            "user": "我对机器学习比较感兴趣，特别是算法方面",
            "assistant": "机器学习算法主要包括监督学习、无监督学习和强化学习三大类。您想了解哪种类型的算法呢？"
        },
        {
            "user": "监督学习的算法有哪些？",
            "assistant": "监督学习的经典算法包括线性回归、逻辑回归、决策树、随机森林、支持向量机、神经网络等。每种算法都有其适用的场景。"
        },
        {
            "user": "神经网络和深度学习有什么关系？",
            "assistant": "深度学习是机器学习的一个分支，神经网络是深度学习的核心。深度学习使用多层神经网络来学习数据的复杂表示。"
        },
        {
            "user": "能推荐一些学习资源吗？",
            "assistant": "基于您的兴趣，我推荐您可以学习Python编程，然后逐步深入机器学习和深度学习。有很多优质的在线课程和书籍可以参考。"
        }
    ]
    
    # 处理对话
    for i, exchange in enumerate(conversation):
        print(f"\n--- 对话轮次 {i+1} ---")
        print(f"用户: {exchange['user']}")
        
        # 处理用户输入
        result = await system.process_interaction(
            user_input=exchange['user'],
            context={
                "turn": i+1,
                "conversation_id": "demo_001",
                "user_id": "demo_user"
            }
        )
        
        print(f"系统处理结果:")
        print(f"  记忆ID: {result['memory_id']}")
        print(f"  相关知识: {len(result['related_knowledge'])}")
        print(f"  推理结论: {len(result['reasoning_result'].get('conclusions', []))}")
        
        # 模拟系统回复
        print(f"助手: {exchange['assistant']}")
    
    # 获取最终统计
    print(f"\n=== 系统最终统计 ===")
    
    memory_stats = await system.memory_manager.get_memory_stats()
    print(f"记忆统计: {memory_stats}")
    
    learning_stats = await system.learning_system.get_learning_statistics()
    print(f"学习统计: {learning_stats}")
    
    kb_stats = await system.knowledge_manager.get_knowledge_statistics()
    print(f"知识库统计: {kb_stats}")
    
    search_analytics = await system.semantic_search.get_search_analytics()
    print(f"搜索分析: {search_analytics}")
    
    await system.close()


async def performance_benchmark_example():
    """性能基准测试示例"""
    
    print("\n=== 性能基准测试 ===\n")
    
    config = get_merged_config('dev')
    system = IntegratedMemoryKnowledgeSystem(config)
    
    await system.initialize()
    
    import time
    
    # 测试数据
    test_queries = [f"测试查询 {i}" for i in range(100)]
    
    # 1. 测试记忆存储性能
    print("测试记忆存储性能...")
    start_time = time.time()
    
    for query in test_queries:
        await system.memory_manager.store_short_term_memory(query)
    
    storage_time = time.time() - start_time
    print(f"存储100条记忆用时: {storage_time:.3f}秒")
    print(f"平均每条: {storage_time/100:.3f}秒")
    
    # 2. 测试搜索性能
    print("\n测试搜索性能...")
    start_time = time.time()
    
    for query in test_queries[:10]:  # 只测试前10个查询
        await system.semantic_search.search(query, top_k=5)
    
    search_time = time.time() - start_time
    print(f"搜索10次用时: {search_time:.3f}秒")
    print(f"平均每次: {search_time/10:.3f}秒")
    
    # 3. 测试推理性能
    print("\n测试推理性能...")
    start_time = time.time()
    
    for query in test_queries[:5]:
        await system.reasoning_engine.reason(query)
    
    reasoning_time = time.time() - start_time
    print(f"推理5次用时: {reasoning_time:.3f}秒")
    print(f"平均每次: {reasoning_time/5:.3f}秒")
    
    # 4. 系统资源使用
    print(f"\n系统资源统计:")
    
    memory_stats = await system.memory_manager.get_memory_stats()
    print(f"记忆项总数: {memory_stats.get('total_memories', 0)}")
    
    kb_stats = await system.knowledge_manager.get_knowledge_statistics()
    print(f"知识项总数: {kb_stats.get('total_items', 0)}")
    
    await system.close()


async def main():
    """主函数：运行所有示例"""
    
    print("知识库记忆系统完整示例")
    print("=" * 50)
    
    try:
        # 基础使用
        await basic_usage_example()
        
        # 知识管理
        await knowledge_management_example()
        
        # 记忆和学习
        await memory_and_learning_example()
        
        # 知识图谱
        await knowledge_graph_example()
        
        # 语义搜索
        await semantic_search_example()
        
        # 系统集成
        await system_integration_example()
        
        # 性能基准测试
        await performance_benchmark_example()
        
        print("\n所有示例运行完成！")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())