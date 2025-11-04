"""
知识库记忆系统 - 集成记忆管理和知识图谱的智能系统

该系统提供：
- 长期记忆存储和管理
- 知识图谱构建和推理
- 向量数据库和语义搜索
- 个性化学习和适应
- 知识关联发现
"""

from .core.memory_manager import MemoryManager
from .core.knowledge_graph import KnowledgeGraph
from .storage.vector_database import VectorDatabase
from .search.semantic_search import SemanticSearchEngine
from .reasoning.reasoning_engine import KnowledgeReasoningEngine
from .learning.personalized_learning import PersonalizedLearningSystem
from .management.knowledge_base_manager import KnowledgeBaseManager

__version__ = "1.0.0"
__all__ = [
    "MemoryManager",
    "KnowledgeGraph", 
    "VectorDatabase",
    "SemanticSearchEngine",
    "KnowledgeReasoningEngine",
    "PersonalizedLearningSystem",
    "KnowledgeBaseManager"
]

class IntegratedMemoryKnowledgeSystem:
    """集成知识库记忆系统主类"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # 初始化各个组件
        self.memory_manager = MemoryManager(config.get('memory', {}))
        self.knowledge_graph = KnowledgeGraph(config.get('knowledge_graph', {}))
        self.vector_db = VectorDatabase(config.get('vector_db', {}))
        self.semantic_search = SemanticSearchEngine(
            self.vector_db, 
            config.get('semantic_search', {})
        )
        self.reasoning_engine = KnowledgeReasoningEngine(
            self.knowledge_graph,
            config.get('reasoning', {})
        )
        self.learning_system = PersonalizedLearningSystem(
            self.memory_manager,
            config.get('learning', {})
        )
        self.knowledge_manager = KnowledgeBaseManager(
            self.memory_manager,
            self.knowledge_graph,
            self.vector_db,
            config.get('knowledge_base', {})
        )
    
    async def initialize(self):
        """初始化系统"""
        await self.memory_manager.initialize()
        await self.knowledge_graph.initialize()
        await self.vector_db.initialize()
        await self.knowledge_manager.initialize()
        print("知识库记忆系统初始化完成")
    
    async def process_interaction(self, user_input, context=None):
        """处理用户交互，整合记忆和知识"""
        # 存储短期记忆
        memory_id = await self.memory_manager.store_short_term_memory(
            content=user_input,
            context=context
        )
        
        # 语义搜索相关知识
        related_knowledge = await self.semantic_search.search(
            query=user_input,
            top_k=5
        )
        
        # 知识推理
        reasoning_result = await self.reasoning_engine.reason(
            query=user_input,
            context=related_knowledge
        )
        
        # 个性化学习
        learning_result = await self.learning_system.learn_from_interaction(
            user_input=user_input,
            context=context,
            memory_id=memory_id
        )
        
        # 更新知识库
        await self.knowledge_manager.update_from_interaction(
            user_input=user_input,
            memory_id=memory_id,
            reasoning_result=reasoning_result
        )
        
        return {
            'memory_id': memory_id,
            'related_knowledge': related_knowledge,
            'reasoning_result': reasoning_result,
            'learning_result': learning_result
        }
    
    async def get_contextual_memory(self, query, limit=10):
        """获取上下文相关的记忆"""
        # 语义搜索记忆
        semantic_memories = await self.semantic_search.search(
            query=query,
            collection='memories',
            top_k=limit
        )
        
        # 知识图谱关联
        graph_connections = await self.knowledge_graph.find_connections(query)
        
        return {
            'semantic_memories': semantic_memories,
            'graph_connections': graph_connections
        }
    
    async def close(self):
        """关闭系统"""
        await self.memory_manager.close()
        await self.vector_db.close()
        await self.knowledge_graph.close()
        print("知识库记忆系统已关闭")