"""
依赖关系分析器

分析任务依赖关系，进行拓扑排序和循环依赖检测
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque
import networkx as nx

from .models import Task, TaskDependency, TaskExecutionPlan, TaskStatus


class DependencyAnalyzer:
    """任务依赖关系分析器"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.task_map: Dict[str, Task] = {}
        self.dependency_map: Dict[str, TaskDependency] = {}
    
    def analyze_dependencies(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """
        分析任务执行计划的依赖关系
        
        Returns:
            分析结果字典，包含:
            - topological_order: 拓扑排序结果
            - critical_path: 关键路径
            - parallel_groups: 可并行执行的任务组
            - dependency_issues: 依赖关系问题
            - execution_phases: 执行阶段
        """
        # 构建依赖图
        self._build_dependency_graph(plan)
        
        # 检测循环依赖
        cycles = self._detect_cycles()
        
        # 拓扑排序
        topological_order = self._topological_sort()
        
        # 识别关键路径
        critical_path = self._find_critical_path()
        
        # 识别可并行执行的组
        parallel_groups = self._identify_parallel_groups()
        
        # 划分执行阶段
        execution_phases = self._create_execution_phases(topological_order)
        
        # 检测依赖问题
        dependency_issues = self._detect_dependency_issues(plan)
        
        return {
            "topological_order": topological_order,
            "critical_path": critical_path,
            "parallel_groups": parallel_groups,
            "execution_phases": execution_phases,
            "dependency_issues": dependency_issues,
            "cycles": cycles,
            "total_tasks": len(plan.tasks),
            "dependency_count": len(plan.dependencies)
        }
    
    def _build_dependency_graph(self, plan: TaskExecutionPlan):
        """构建依赖关系图"""
        self.graph.clear()
        self.task_map.clear()
        self.dependency_map.clear()
        
        # 添加任务节点
        for task in plan.tasks:
            self.graph.add_node(task.id)
            self.task_map[task.id] = task
        
        # 添加依赖边
        for dependency in plan.dependencies:
            self.graph.add_edge(
                dependency.source_task_id,
                dependency.target_task_id,
                dependency=dependency
            )
            self.dependency_map[dependency.id] = dependency
    
    def _detect_cycles(self) -> List[List[str]]:
        """检测循环依赖"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except Exception as e:
            print(f"循环依赖检测失败: {e}")
            return []
    
    def _topological_sort(self) -> List[str]:
        """拓扑排序"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            # 存在循环依赖，使用近似排序
            return self._approximate_topological_sort()
    
    def _approximate_topological_sort(self) -> List[str]:
        """近似拓扑排序（处理循环依赖）"""
        # 使用Kahn算法的变种
        in_degree = defaultdict(int)
        for node in self.graph.nodes():
            in_degree[node] = self.graph.in_degree(node)
        
        queue = deque([node for node in self.graph.nodes() if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in self.graph.successors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 添加剩余节点（形成环的节点）
        remaining = [node for node in self.graph.nodes() if node not in result]
        result.extend(remaining)
        
        return result
    
    def _find_critical_path(self) -> List[str]:
        """找到关键路径"""
        if not self.graph.nodes():
            return []
        
        try:
            # 计算每个节点的最早开始时间
            earliest_start = self._calculate_earliest_start_times()
            
            # 计算每个节点的最晚开始时间
            latest_start = self._calculate_latest_start_times(earliest_start)
            
            # 找到关键路径（最早开始时间等于最晚开始时间的任务）
            critical_nodes = []
            for node in self.graph.nodes():
                if abs(earliest_start[node] - latest_start[node]) < 0.001:  # 浮点数比较
                    critical_nodes.append(node)
            
            # 按时间顺序排列关键路径
            critical_path = sorted(critical_nodes, key=lambda x: earliest_start[x])
            return critical_path
            
        except Exception as e:
            print(f"关键路径计算失败: {e}")
            return []
    
    def _calculate_earliest_start_times(self) -> Dict[str, float]:
        """计算最早开始时间"""
        earliest_start = {}
        
        # 拓扑排序
        topo_order = self._topological_sort()
        
        for node in topo_order:
            # 获取所有前驱节点
            predecessors = list(self.graph.predecessors(node))
            
            if not predecessors:
                earliest_start[node] = 0.0
            else:
                # 最早开始时间是所有前驱节点完成时间的最大值
                max_finish_time = 0.0
                for pred in predecessors:
                    duration = self.task_map[pred].estimated_duration or 0
                    finish_time = earliest_start[pred] + duration
                    max_finish_time = max(max_finish_time, finish_time)
                
                earliest_start[node] = max_finish_time
        
        return earliest_start
    
    def _calculate_latest_start_times(self, earliest_start: Dict[str, float]) -> Dict[str, float]:
        """计算最晚开始时间"""
        latest_start = {}
        
        # 拓扑逆序
        topo_order = self._topological_sort()
        reversed_order = list(reversed(topo_order))
        
        # 找到项目总工期
        project_duration = max(earliest_start.values()) if earliest_start else 0
        
        for node in reversed_order:
            # 获取所有后继节点
            successors = list(self.graph.successors(node))
            
            if not successors:
                # 末端节点的最晚开始时间是项目总工期减去其持续时间
                duration = self.task_map[node].estimated_duration or 0
                latest_start[node] = project_duration - duration
            else:
                # 最晚开始时间是所有后继节点最晚开始时间的最小值减去当前任务持续时间
                min_start_time = float('inf')
                for succ in successors:
                    duration = self.task_map[node].estimated_duration or 0
                    start_time = latest_start[succ] - duration
                    min_start_time = min(min_start_time, start_time)
                
                latest_start[node] = min_start_time
        
        return latest_start
    
    def _identify_parallel_groups(self) -> List[List[str]]:
        """识别可并行执行的任务组"""
        groups = []
        processed = set()
        
        # 拓扑排序
        topo_order = self._topological_sort()
        
        i = 0
        while i < len(topo_order):
            current_level = []
            
            # 收集当前层级的任务（没有未处理的前驱任务）
            while i < len(topo_order):
                task_id = topo_order[i]
                
                # 检查是否所有前驱任务都已处理
                predecessors = list(self.graph.predecessors(task_id))
                if all(pred in processed for pred in predecessors):
                    current_level.append(task_id)
                    processed.add(task_id)
                    i += 1
                else:
                    break
            
            if current_level:
                groups.append(current_level)
        
        return groups
    
    def _create_execution_phases(self, topological_order: List[str]) -> List[Dict[str, Any]]:
        """创建执行阶段"""
        phases = []
        phase_tasks = []
        current_phase_start = 0
        
        for i, task_id in enumerate(topological_order):
            task = self.task_map[task_id]
            
            # 检查是否可以开始新阶段
            predecessors = list(self.graph.predecessors(task_id))
            can_start = all(
                self.task_map[pred].status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
                for pred in predecessors
            )
            
            if can_start and phase_tasks:
                # 创建当前阶段
                phases.append({
                    "phase_id": len(phases),
                    "tasks": phase_tasks.copy(),
                    "start_time": current_phase_start,
                    "estimated_duration": max(
                        self.task_map[tid].estimated_duration or 0 
                        for tid in phase_tasks
                    )
                })
                
                # 开始新阶段
                phase_tasks = [task_id]
                current_phase_start = i
            else:
                phase_tasks.append(task_id)
        
        # 添加最后一个阶段
        if phase_tasks:
            phases.append({
                "phase_id": len(phases),
                "tasks": phase_tasks,
                "start_time": current_phase_start,
                "estimated_duration": max(
                    self.task_map[tid].estimated_duration or 0 
                    for tid in phase_tasks
                )
            })
        
        return phases
    
    def _detect_dependency_issues(self, plan: TaskExecutionPlan) -> List[Dict[str, Any]]:
        """检测依赖关系问题"""
        issues = []
        
        # 检查不存在的任务依赖
        existing_task_ids = {task.id for task in plan.tasks}
        
        for dependency in plan.dependencies:
            if dependency.source_task_id not in existing_task_ids:
                issues.append({
                    "type": "missing_source_task",
                    "dependency_id": dependency.id,
                    "source_task_id": dependency.source_task_id,
                    "message": f"依赖的源任务不存在: {dependency.source_task_id}"
                })
            
            if dependency.target_task_id not in existing_task_ids:
                issues.append({
                    "type": "missing_target_task",
                    "dependency_id": dependency.id,
                    "target_task_id": dependency.target_task_id,
                    "message": f"依赖的目标任务不存在: {dependency.target_task_id}"
                })
        
        # 检查自依赖
        for task in plan.tasks:
            for dependency in task.dependencies:
                if dependency.source_task_id == task.id:
                    issues.append({
                        "type": "self_dependency",
                        "task_id": task.id,
                        "message": f"任务 '{task.name}' 存在自依赖"
                    })
        
        # 检查循环依赖
        cycles = self._detect_cycles()
        for cycle in cycles:
            issues.append({
                "type": "circular_dependency",
                "cycle": cycle,
                "message": f"发现循环依赖: {' -> '.join(cycle)} -> {cycle[0]}"
            })
        
        return issues
    
    def get_ready_tasks(self, completed_tasks: Set[str] = None) -> List[str]:
        """获取当前可执行的任务"""
        if completed_tasks is None:
            completed_tasks = set()
        
        ready_tasks = []
        for node in self.graph.nodes():
            if node in completed_tasks:
                continue
            
            # 检查所有前驱任务是否完成
            predecessors = list(self.graph.predecessors(node))
            if all(pred in completed_tasks for pred in predecessors):
                ready_tasks.append(node)
        
        return ready_tasks
    
    def get_task_dependencies(self, task_id: str) -> List[str]:
        """获取任务的所有依赖任务ID"""
        return list(self.graph.predecessors(task_id))
    
    def get_task_dependents(self, task_id: str) -> List[str]:
        """获取依赖指定任务的所有任务ID"""
        return list(self.graph.successors(task_id))
    
    def validate_execution_plan(self, plan: TaskExecutionPlan) -> Tuple[bool, List[str]]:
        """验证执行计划的可行性"""
        issues = self.analyze_dependencies(plan)
        
        is_valid = (
            len(issues["cycles"]) == 0 and
            len(issues["dependency_issues"]) == 0
        )
        
        error_messages = []
        for issue in issues["dependency_issues"]:
            error_messages.append(issue["message"])
        
        for cycle in issues["cycles"]:
            error_messages.append(f"循环依赖: {' -> '.join(cycle)}")
        
        return is_valid, error_messages