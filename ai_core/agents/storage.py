"""
持久化存储

负责任务状态、结果和配置的持久化存储
"""

import asyncio
import json
import sqlite3
import aiosqlite
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

from .task_models import Task, TaskResult, TaskStatus, TaskPriority


class TaskStorage:
    """任务持久化存储"""
    
    def __init__(
        self, 
        db_path: str = "tasks.db",
        backup_enabled: bool = True,
        auto_backup_interval: int = 3600  # 1小时
    ):
        self.db_path = db_path
        self.backup_enabled = backup_enabled
        self.auto_backup_interval = auto_backup_interval
        self.logger = logging.getLogger(__name__)
        
        self._backup_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start(self):
        """启动存储服务"""
        await self._init_database()
        
        if self.backup_enabled:
            self._backup_task = asyncio.create_task(self._auto_backup_loop())
            self._is_running = True
    
    async def stop(self):
        """停止存储服务"""
        self._is_running = False
        
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
    
    async def save_task(self, task: Task) -> bool:
        """保存任务"""
        try:
            async with self._get_db_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO tasks (
                        id, name, status, func_name, func_module, args, kwargs,
                        config, metrics, result, error, traceback,
                        dependencies_resolved, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id,
                    task.name,
                    task.status.value,
                    task.func.__name__ if task.func else None,
                    task.func.__module__ if task.func else None,
                    pickle.dumps(task.args),
                    pickle.dumps(task.kwargs),
                    pickle.dumps(task.config),
                    pickle.dumps(task.metrics),
                    pickle.dumps(task.result),
                    task.error,
                    task.traceback,
                    task.dependencies_resolved,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat()
                ))
                await db.commit()
            return True
        except Exception as e:
            self.logger.error(f"保存任务失败: {task.id}, 错误: {str(e)}")
            return False
    
    async def load_task(self, task_id: str) -> Optional[Task]:
        """加载任务"""
        try:
            async with self._get_db_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM tasks WHERE id = ?", (task_id,)
                )
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_task(row)
        except Exception as e:
            self.logger.error(f"加载任务失败: {task_id}, 错误: {str(e)}")
            return None
    
    async def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        try:
            async with self._get_db_connection() as db:
                await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                await db.execute("DELETE FROM task_results WHERE task_id = ?", (task_id,))
                await db.commit()
            return True
        except Exception as e:
            self.logger.error(f"删除任务失败: {task_id}, 错误: {str(e)}")
            return False
    
    async def list_tasks(
        self, 
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Task]:
        """列出任务"""
        try:
            async with self._get_db_connection() as db:
                if status:
                    cursor = await db.execute(
                        "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (status.value, limit, offset)
                    )
                else:
                    cursor = await db.execute(
                        "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset)
                    )
                
                rows = await cursor.fetchall()
                return [self._row_to_task(row) for row in rows]
        except Exception as e:
            self.logger.error(f"列出任务失败: {str(e)}")
            return []
    
    async def save_task_result(self, result: TaskResult) -> bool:
        """保存任务结果"""
        try:
            async with self._get_db_connection() as db:
                await db.execute("""
                    INSERT OR REPLACE INTO task_results (
                        task_id, status, result, error, traceback,
                        metrics, cached, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.task_id,
                    result.status.value,
                    pickle.dumps(result.result),
                    result.error,
                    result.traceback,
                    pickle.dumps(result.metrics),
                    result.cached,
                    datetime.now().isoformat()
                ))
                await db.commit()
            return True
        except Exception as e:
            self.logger.error(f"保存任务结果失败: {result.task_id}, 错误: {str(e)}")
            return False
    
    async def load_task_result(self, task_id: str) -> Optional[TaskResult]:
        """加载任务结果"""
        try:
            async with self._get_db_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM task_results WHERE task_id = ?", (task_id,)
                )
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_result(row)
        except Exception as e:
            self.logger.error(f"加载任务结果失败: {task_id}, 错误: {str(e)}")
            return None
    
    async def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """按状态获取任务"""
        return await self.list_tasks(status=status)
    
    async def get_running_tasks(self) -> List[Task]:
        """获取运行中的任务"""
        return await self.get_tasks_by_status(TaskStatus.RUNNING)
    
    async def get_pending_tasks(self) -> List[Task]:
        """获取等待中的任务"""
        return await self.get_tasks_by_status(TaskStatus.PENDING)
    
    async def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """更新任务状态"""
        try:
            async with self._get_db_connection() as db:
                await db.execute(
                    "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                    (status.value, datetime.now().isoformat(), task_id)
                )
                await db.commit()
            return True
        except Exception as e:
            self.logger.error(f"更新任务状态失败: {task_id}, 错误: {str(e)}")
            return False
    
    async def get_task_count(self, status: Optional[TaskStatus] = None) -> int:
        """获取任务数量"""
        try:
            async with self._get_db_connection() as db:
                if status:
                    cursor = await db.execute(
                        "SELECT COUNT(*) FROM tasks WHERE status = ?", (status.value,)
                    )
                else:
                    cursor = await db.execute("SELECT COUNT(*) FROM tasks")
                
                result = await cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"获取任务数量失败: {str(e)}")
            return 0
    
    async def cleanup_old_tasks(self, days: int = 30) -> int:
        """清理旧任务"""
        try:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
            
            async with self._get_db_connection() as db:
                # 删除旧任务
                cursor = await db.execute(
                    "DELETE FROM tasks WHERE created_at < ?",
                    (datetime.fromtimestamp(cutoff_date).isoformat(),)
                )
                deleted_count = cursor.rowcount
                
                # 删除相关的任务结果
                await db.execute(
                    "DELETE FROM task_results WHERE created_at < ?",
                    (datetime.fromtimestamp(cutoff_date).isoformat(),)
                )
                
                await db.commit()
            
            self.logger.info(f"已清理 {deleted_count} 个旧任务")
            return deleted_count
        except Exception as e:
            self.logger.error(f"清理旧任务失败: {str(e)}")
            return 0
    
    async def export_tasks(self, file_path: str, format: str = "json") -> bool:
        """导出任务"""
        try:
            tasks = await self.list_tasks(limit=10000)  # 导出所有任务
            
            if format.lower() == "json":
                tasks_data = [task.to_dict() for task in tasks]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(tasks_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"已导出 {len(tasks)} 个任务到 {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"导出任务失败: {str(e)}")
            return False
    
    async def import_tasks(self, file_path: str, format: str = "json") -> int:
        """导入任务"""
        try:
            imported_count = 0
            
            if format.lower() == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                
                for task_data in tasks_data:
                    # 重建Task对象
                    task = self._dict_to_task(task_data)
                    if task:
                        await self.save_task(task)
                        imported_count += 1
            else:
                raise ValueError(f"不支持的导入格式: {format}")
            
            self.logger.info(f"已从 {file_path} 导入 {imported_count} 个任务")
            return imported_count
        except Exception as e:
            self.logger.error(f"导入任务失败: {str(e)}")
            return 0
    
    async def create_backup(self, backup_path: Optional[str] = None) -> str:
        """创建备份"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"tasks_backup_{timestamp}.db"
        
        try:
            # 复制数据库文件
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            self.logger.info(f"备份已创建: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"创建备份失败: {str(e)}")
            raise
    
    async def _init_database(self):
        """初始化数据库"""
        async with self._get_db_connection() as db:
            # 创建任务表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    func_name TEXT,
                    func_module TEXT,
                    args BLOB,
                    kwargs BLOB,
                    config BLOB,
                    metrics BLOB,
                    result BLOB,
                    error TEXT,
                    traceback TEXT,
                    dependencies_resolved BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # 创建任务结果表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS task_results (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result BLOB,
                    error TEXT,
                    traceback TEXT,
                    metrics BLOB,
                    cached BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE CASCADE
                )
            """)
            
            # 创建索引
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_results_created_at ON task_results(created_at)
            """)
            
            await db.commit()
    
    @asynccontextmanager
    async def _get_db_connection(self):
        """获取数据库连接"""
        async with aiosqlite.connect(self.db_path) as db:
            # 设置外键约束
            await db.execute("PRAGMA foreign_keys = ON")
            # 设置WAL模式以提高并发性能
            await db.execute("PRAGMA journal_mode = WAL")
            yield db
    
    def _row_to_task(self, row) -> Task:
        """将数据库行转换为Task对象"""
        from .task_models import TaskConfig, TaskMetrics
        
        # 重建config
        config_data = pickle.loads(row[7]) if row[7] else TaskConfig()
        if isinstance(config_data, dict):
            config = TaskConfig(**config_data)
        else:
            config = config_data
        
        # 重建metrics
        metrics_data = pickle.loads(row[8]) if row[8] else TaskMetrics()
        if isinstance(metrics_data, dict):
            metrics = TaskMetrics(**metrics_data)
        else:
            metrics = metrics_data
        
        task = Task(
            id=row[0],
            name=row[1],
            status=TaskStatus(row[2]),
            config=config,
            metrics=metrics,
            result=pickle.loads(row[9]) if row[9] else None,
            error=row[10],
            traceback=row[11],
            dependencies_resolved=bool(row[12]),
            created_at=datetime.fromisoformat(row[13]),
            updated_at=datetime.fromisoformat(row[14])
        )
        
        # 重建函数（这里简化处理，实际需要更复杂的反序列化）
        if row[3] and row[4]:
            # 注意：这里无法完全重建函数对象，需要特殊处理
            pass
        
        return task
    
    def _row_to_result(self, row) -> TaskResult:
        """将数据库行转换为TaskResult对象"""
        from .task_models import TaskMetrics
        
        metrics_data = pickle.loads(row[5]) if row[5] else TaskMetrics()
        if isinstance(metrics_data, dict):
            metrics = TaskMetrics(**metrics_data)
        else:
            metrics = metrics_data
        
        return TaskResult(
            task_id=row[0],
            status=TaskStatus(row[1]),
            result=pickle.loads(row[2]) if row[2] else None,
            error=row[3],
            traceback=row[4],
            metrics=metrics,
            cached=bool(row[6])
        )
    
    def _dict_to_task(self, data: Dict[str, Any]) -> Optional[Task]:
        """将字典转换为Task对象"""
        try:
            from .task_models import TaskConfig, TaskMetrics, TaskStatus
            
            # 重建config
            config_data = data.get("config", {})
            config = TaskConfig(**config_data)
            
            # 重建metrics
            metrics_data = data.get("metrics", {})
            metrics = TaskMetrics(
                created_at=datetime.fromisoformat(metrics_data.get("created_at")),
                started_at=datetime.fromisoformat(metrics_data.get("started_at")) if metrics_data.get("started_at") else None,
                completed_at=datetime.fromisoformat(metrics_data.get("completed_at")) if metrics_data.get("completed_at") else None,
                duration=metrics_data.get("duration"),
                retry_count=metrics_data.get("retry_count", 0),
                progress=metrics_data.get("progress", 0.0),
                memory_usage=metrics_data.get("memory_usage"),
                cpu_usage=metrics_data.get("cpu_usage")
            )
            
            task = Task(
                id=data["id"],
                name=data["name"],
                status=TaskStatus(data["status"]),
                config=config,
                metrics=metrics,
                dependencies_resolved=data.get("dependencies_resolved", False),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"])
            )
            
            return task
        except Exception as e:
            self.logger.error(f"转换任务字典失败: {str(e)}")
            return None
    
    async def _auto_backup_loop(self):
        """自动备份循环"""
        while self._is_running:
            try:
                await asyncio.sleep(self.auto_backup_interval)
                await self.create_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"自动备份失败: {str(e)}")


class JSONStorage:
    """JSON文件存储（轻量级选项）"""
    
    def __init__(self, storage_path: str = "tasks_storage.json"):
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def save_task(self, task: Task) -> bool:
        """保存任务到JSON文件"""
        async with self._lock:
            try:
                tasks_data = await self._load_all_tasks()
                tasks_data[task.id] = task.to_dict()
                await self._save_all_tasks(tasks_data)
                return True
            except Exception as e:
                self.logger.error(f"保存任务失败: {task.id}, 错误: {str(e)}")
                return False
    
    async def load_task(self, task_id: str) -> Optional[Task]:
        """从JSON文件加载任务"""
        try:
            tasks_data = await self._load_all_tasks()
            task_data = tasks_data.get(task_id)
            
            if task_data:
                from .task_models import Task
                return self._dict_to_task(task_data)
            
            return None
        except Exception as e:
            self.logger.error(f"加载任务失败: {task_id}, 错误: {str(e)}")
            return None
    
    async def _load_all_tasks(self) -> Dict[str, Any]:
        """加载所有任务数据"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            self.logger.error(f"加载任务数据失败: {str(e)}")
            return {}
    
    async def _save_all_tasks(self, tasks_data: Dict[str, Any]):
        """保存所有任务数据"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _dict_to_task(self, data: Dict[str, Any]) -> Task:
        """将字典转换为Task对象"""
        from .task_models import TaskConfig, TaskMetrics, TaskStatus
        
        # 重建config
        config_data = data.get("config", {})
        config = TaskConfig(**config_data)
        
        # 重建metrics
        metrics_data = data.get("metrics", {})
        metrics = TaskMetrics(
            created_at=datetime.fromisoformat(metrics_data.get("created_at")),
            started_at=datetime.fromisoformat(metrics_data.get("started_at")) if metrics_data.get("started_at") else None,
            completed_at=datetime.fromisoformat(metrics_data.get("completed_at")) if metrics_data.get("completed_at") else None,
            duration=metrics_data.get("duration"),
            retry_count=metrics_data.get("retry_count", 0),
            progress=metrics_data.get("progress", 0.0),
            memory_usage=metrics_data.get("memory_usage"),
            cpu_usage=metrics_data.get("cpu_usage")
        )
        
        return Task(
            id=data["id"],
            name=data["name"],
            status=TaskStatus(data["status"]),
            config=config,
            metrics=metrics,
            dependencies_resolved=data.get("dependencies_resolved", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )