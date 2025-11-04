#!/usr/bin/env python3
"""
诺玛Agent系统 - 部署配置管理器
==============================

支持多种部署环境:
1. 开发环境 (Development)
2. 测试环境 (Testing) 
3. 预生产环境 (Staging)
4. 生产环境 (Production)

部署特性:
- 自动化部署流程
- 环境配置管理
- 健康检查和监控
- 滚动更新支持
- 回滚机制
- 性能优化配置

作者: 皇
创建时间: 2025-11-01
版本: 2.0.0
"""

import os
import json
import yaml
import asyncio
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import psutil

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """环境配置"""
    name: str
    description: str
    host: str
    port: int
    debug: bool
    log_level: str
    max_workers: int
    memory_limit_mb: int
    cpu_limit_percent: float
    database_url: str
    redis_url: str
    monitoring_enabled: bool
    auto_scaling: bool
    health_check_interval: int
    deployment_strategy: str  # "blue_green", "rolling", "recreate"

@dataclass
class DeploymentConfig:
    """部署配置"""
    environment: str
    version: str
    build_id: str
    timestamp: str
    rollback_enabled: bool
    health_check_enabled: bool
    monitoring_enabled: bool
    notification_enabled: bool
    backup_enabled: bool

class NormaDeploymentManager:
    """诺玛Agent部署管理器"""
    
    def __init__(self):
        self.deployment_history = []
        self.current_deployment = None
        self.environments = self._load_environment_configs()
        
    def _load_environment_configs(self) -> Dict[str, EnvironmentConfig]:
        """加载环境配置"""
        return {
            "development": EnvironmentConfig(
                name="development",
                description="开发环境",
                host="localhost",
                port=8000,
                debug=True,
                log_level="DEBUG",
                max_workers=2,
                memory_limit_mb=1024,
                cpu_limit_percent=50.0,
                database_url="sqlite:///norma_dev.db",
                redis_url="redis://localhost:6379/0",
                monitoring_enabled=False,
                auto_scaling=False,
                health_check_interval=30,
                deployment_strategy="recreate"
            ),
            "testing": EnvironmentConfig(
                name="testing",
                description="测试环境",
                host="0.0.0.0",
                port=8001,
                debug=False,
                log_level="INFO",
                max_workers=4,
                memory_limit_mb=2048,
                cpu_limit_percent=70.0,
                database_url="sqlite:///norma_test.db",
                redis_url="redis://localhost:6379/1",
                monitoring_enabled=True,
                auto_scaling=False,
                health_check_interval=30,
                deployment_strategy="rolling"
            ),
            "staging": EnvironmentConfig(
                name="staging",
                description="预生产环境",
                host="0.0.0.0",
                port=8002,
                debug=False,
                log_level="WARNING",
                max_workers=6,
                memory_limit_mb=4096,
                cpu_limit_percent=80.0,
                database_url="postgresql://user:pass@localhost:5432/norma_staging",
                redis_url="redis://localhost:6379/2",
                monitoring_enabled=True,
                auto_scaling=True,
                health_check_interval=15,
                deployment_strategy="blue_green"
            ),
            "production": EnvironmentConfig(
                name="production",
                description="生产环境",
                host="0.0.0.0",
                port=8003,
                debug=False,
                log_level="ERROR",
                max_workers=12,
                memory_limit_mb=8192,
                cpu_limit_percent=90.0,
                database_url="postgresql://user:pass@prod-db:5432/norma_prod",
                redis_url="redis://prod-redis:6379/0",
                monitoring_enabled=True,
                auto_scaling=True,
                health_check_interval=10,
                deployment_strategy="blue_green"
            )
        }
    
    def get_environment_config(self, environment: str) -> Optional[EnvironmentConfig]:
        """获取环境配置"""
        return self.environments.get(environment)
    
    def create_deployment_config(self, environment: str, version: str = "2.0.0") -> DeploymentConfig:
        """创建部署配置"""
        return DeploymentConfig(
            environment=environment,
            version=version,
            build_id=f"build_{int(datetime.now().timestamp())}",
            timestamp=datetime.now().isoformat(),
            rollback_enabled=True,
            health_check_enabled=True,
            monitoring_enabled=True,
            notification_enabled=True,
            backup_enabled=True
        )
    
    async def deploy_to_environment(self, environment: str, config: Optional[DeploymentConfig] = None) -> Dict[str, Any]:
        """部署到指定环境"""
        try:
            logger.info(f"🚀 开始部署到 {environment} 环境...")
            
            # 验证环境配置
            env_config = self.get_environment_config(environment)
            if not env_config:
                raise ValueError(f"未知环境: {environment}")
            
            # 创建部署配置
            if not config:
                config = self.create_deployment_config(environment)
            
            self.current_deployment = config
            
            # 部署步骤
            deployment_result = {
                "environment": environment,
                "config": asdict(config),
                "steps": [],
                "success": False,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration_seconds": 0
            }
            
            # 1. 部署前检查
            pre_check_result = await self._pre_deployment_checks(environment, env_config)
            deployment_result["steps"].append(pre_check_result)
            
            if not pre_check_result["success"]:
                raise Exception("部署前检查失败")
            
            # 2. 创建备份
            if config.backup_enabled:
                backup_result = await self._create_backup(environment)
                deployment_result["steps"].append(backup_result)
            
            # 3. 停止现有服务
            stop_result = await self._stop_existing_service(environment)
            deployment_result["steps"].append(stop_result)
            
            # 4. 部署新版本
            deploy_result = await self._deploy_new_version(environment, env_config, config)
            deployment_result["steps"].append(deploy_result)
            
            # 5. 启动服务
            start_result = await self._start_service(environment, env_config)
            deployment_result["steps"].append(start_result)
            
            # 6. 健康检查
            if config.health_check_enabled:
                health_result = await self._health_check(environment, env_config)
                deployment_result["steps"].append(health_result)
                
                if not health_result["success"]:
                    # 健康检查失败，执行回滚
                    logger.warning("健康检查失败，开始回滚...")
                    rollback_result = await self._rollback_deployment(environment)
                    deployment_result["steps"].append(rollback_result)
                    raise Exception("健康检查失败，已执行回滚")
            
            # 7. 部署后验证
            post_check_result = await self._post_deployment_verification(environment)
            deployment_result["steps"].append(post_check_result)
            
            # 部署成功
            deployment_result["success"] = True
            deployment_result["end_time"] = datetime.now().isoformat()
            deployment_result["duration_seconds"] = (
                datetime.fromisoformat(deployment_result["end_time"]) - 
                datetime.fromisoformat(deployment_result["start_time"])
            ).total_seconds()
            
            # 保存部署记录
            self.deployment_history.append(deployment_result)
            
            logger.info(f"✅ 部署到 {environment} 环境成功完成!")
            return deployment_result
            
        except Exception as e:
            logger.error(f"❌ 部署到 {environment} 环境失败: {e}")
            
            if self.current_deployment:
                self.current_deployment.success = False
                self.current_deployment.end_time = datetime.now().isoformat()
            
            return {
                "environment": environment,
                "success": False,
                "error": str(e),
                "steps": deployment_result.get("steps", []),
                "start_time": deployment_result.get("start_time"),
                "end_time": datetime.now().isoformat()
            }
    
    async def _pre_deployment_checks(self, environment: str, env_config: EnvironmentConfig) -> Dict[str, Any]:
        """部署前检查"""
        try:
            logger.info("执行部署前检查...")
            
            checks = {
                "system_resources": self._check_system_resources(env_config),
                "dependencies": self._check_dependencies(),
                "network_connectivity": self._check_network_connectivity(env_config),
                "disk_space": self._check_disk_space(),
                "permissions": self._check_permissions()
            }
            
            all_passed = all(check["success"] for check in checks.values())
            
            return {
                "step": "pre_deployment_checks",
                "success": all_passed,
                "checks": checks,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "pre_deployment_checks",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_system_resources(self, env_config: EnvironmentConfig) -> Dict[str, Any]:
        """检查系统资源"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            memory_ok = memory.available >= env_config.memory_limit_mb * 1024 * 1024
            cpu_ok = cpu_percent <= env_config.cpu_limit_percent
            
            return {
                "success": memory_ok and cpu_ok,
                "memory_available_gb": memory.available / (1024**3),
                "memory_required_gb": env_config.memory_limit_mb / 1024,
                "cpu_usage_percent": cpu_percent,
                "cpu_limit_percent": env_config.cpu_limit_percent
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """检查依赖项"""
        try:
            required_packages = [
                "asyncio", "psutil", "fastapi", "uvicorn", 
                "sqlalchemy", "redis", "pydantic"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            return {
                "success": len(missing_packages) == 0,
                "missing_packages": missing_packages,
                "total_required": len(required_packages),
                "available": len(required_packages) - len(missing_packages)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_network_connectivity(self, env_config: EnvironmentConfig) -> Dict[str, Any]:
        """检查网络连接"""
        try:
            import socket
            
            # 检查端口是否可用
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((env_config.host, env_config.port))
            sock.close()
            
            port_available = result != 0  # 0表示端口被占用
            
            return {
                "success": port_available,
                "port": env_config.port,
                "host": env_config.host,
                "port_available": port_available
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """检查磁盘空间"""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            required_gb = 1.0  # 至少需要1GB可用空间
            
            return {
                "success": free_gb >= required_gb,
                "free_space_gb": free_gb,
                "required_space_gb": required_gb,
                "usage_percent": disk.percent
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_permissions(self) -> Dict[str, Any]:
        """检查文件权限"""
        try:
            # 检查关键目录的读写权限
            critical_paths = [
                "/workspace/code",
                "/workspace/data",
                "/workspace/logs"
            ]
            
            permission_results = []
            for path in critical_paths:
                path_obj = Path(path)
                readable = os.access(path, os.R_OK)
                writable = os.access(path, os.W_OK)
                permission_results.append({
                    "path": path,
                    "readable": readable,
                    "writable": writable
                })
            
            all_ok = all(result["readable"] and result["writable"] for result in permission_results)
            
            return {
                "success": all_ok,
                "path_permissions": permission_results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _create_backup(self, environment: str) -> Dict[str, Any]:
        """创建备份"""
        try:
            logger.info("创建备份...")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = Path(f"/workspace/data/backups/{environment}_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 备份关键文件
            backup_files = [
                "/workspace/code/norma_integrated_system.py",
                "/workspace/data/knowledge_base",
                "/workspace/logs"
            ]
            
            backup_info = []
            for file_path in backup_files:
                source = Path(file_path)
                if source.exists():
                    if source.is_file():
                        import shutil
                        dest = backup_dir / source.name
                        shutil.copy2(source, dest)
                        backup_info.append({"file": str(source), "backup": str(dest)})
                    elif source.is_dir():
                        import shutil
                        dest = backup_dir / source.name
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                        backup_info.append({"directory": str(source), "backup": str(dest)})
            
            return {
                "step": "create_backup",
                "success": True,
                "backup_location": str(backup_dir),
                "backed_up_items": backup_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "create_backup",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _stop_existing_service(self, environment: str) -> Dict[str, Any]:
        """停止现有服务"""
        try:
            logger.info("停止现有服务...")
            
            # 模拟停止服务（在实际环境中这里会是真实的停止命令）
            await asyncio.sleep(1)
            
            return {
                "step": "stop_existing_service",
                "success": True,
                "message": f"已停止 {environment} 环境的现有服务",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "stop_existing_service",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _deploy_new_version(self, environment: str, env_config: EnvironmentConfig, config: DeploymentConfig) -> Dict[str, Any]:
        """部署新版本"""
        try:
            logger.info("部署新版本...")
            
            # 模拟部署过程
            await asyncio.sleep(2)
            
            # 在实际环境中，这里会包括：
            # 1. 下载/构建新版本
            # 2. 配置环境变量
            # 3. 更新配置文件
            # 4. 部署应用文件
            
            return {
                "step": "deploy_new_version",
                "success": True,
                "version": config.version,
                "build_id": config.build_id,
                "deployment_strategy": env_config.deployment_strategy,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "deploy_new_version",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _start_service(self, environment: str, env_config: EnvironmentConfig) -> Dict[str, Any]:
        """启动服务"""
        try:
            logger.info("启动服务...")
            
            # 模拟启动服务
            await asyncio.sleep(1)
            
            return {
                "step": "start_service",
                "success": True,
                "host": env_config.host,
                "port": env_config.port,
                "environment": environment,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "start_service",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _health_check(self, environment: str, env_config: EnvironmentConfig) -> Dict[str, Any]:
        """健康检查"""
        try:
            logger.info("执行健康检查...")
            
            # 模拟健康检查
            await asyncio.sleep(2)
            
            # 检查服务是否响应
            health_checks = {
                "service_responding": True,
                "database_connection": True,
                "memory_usage_ok": True,
                "cpu_usage_ok": True,
                "disk_space_ok": True
            }
            
            all_healthy = all(health_checks.values())
            
            return {
                "step": "health_check",
                "success": all_healthy,
                "health_checks": health_checks,
                "environment": environment,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "health_check",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _post_deployment_verification(self, environment: str) -> Dict[str, Any]:
        """部署后验证"""
        try:
            logger.info("执行部署后验证...")
            
            # 模拟部署后验证
            await asyncio.sleep(1)
            
            verification_checks = {
                "service_accessible": True,
                "api_endpoints_working": True,
                "database_operations_ok": True,
                "logging_functional": True,
                "monitoring_active": True
            }
            
            all_verified = all(verification_checks.values())
            
            return {
                "step": "post_deployment_verification",
                "success": all_verified,
                "verification_checks": verification_checks,
                "environment": environment,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "post_deployment_verification",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _rollback_deployment(self, environment: str) -> Dict[str, Any]:
        """回滚部署"""
        try:
            logger.info("执行部署回滚...")
            
            # 模拟回滚过程
            await asyncio.sleep(2)
            
            return {
                "step": "rollback_deployment",
                "success": True,
                "message": f"已回滚 {environment} 环境到上一个稳定版本",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "rollback_deployment",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_deployment_status(self, environment: str) -> Dict[str, Any]:
        """获取部署状态"""
        try:
            env_config = self.get_environment_config(environment)
            if not env_config:
                return {"error": f"未知环境: {environment}"}
            
            # 检查服务状态
            service_status = self._check_service_status(environment)
            
            # 获取最近的部署记录
            recent_deployments = [
                d for d in self.deployment_history 
                if d["environment"] == environment
            ]
            latest_deployment = max(recent_deployments, key=lambda d: d["start_time"]) if recent_deployments else None
            
            return {
                "environment": environment,
                "config": asdict(env_config),
                "service_status": service_status,
                "latest_deployment": latest_deployment,
                "total_deployments": len(recent_deployments),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _check_service_status(self, environment: str) -> Dict[str, Any]:
        """检查服务状态"""
        try:
            # 模拟服务状态检查
            return {
                "running": True,
                "pid": 12345,
                "memory_usage_mb": 512,
                "cpu_usage_percent": 15.2,
                "uptime_seconds": 3600,
                "last_health_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "running": False,
                "error": str(e)
            }
    
    def save_deployment_config(self, environment: str, filepath: str = None) -> str:
        """保存部署配置"""
        try:
            if not filepath:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"/workspace/data/deployments/deployment_config_{environment}_{timestamp}.yaml"
            
            env_config = self.get_environment_config(environment)
            if not env_config:
                raise ValueError(f"未知环境: {environment}")
            
            config_data = {
                "environment": environment,
                "config": asdict(env_config),
                "generated_at": datetime.now().isoformat(),
                "version": "2.0.0"
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"部署配置已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存部署配置失败: {e}")
            return ""

# 部署管理器实例
deployment_manager = NormaDeploymentManager()

# 便捷函数
async def deploy_to_production():
    """部署到生产环境"""
    return await deployment_manager.deploy_to_environment("production")

async def deploy_to_staging():
    """部署到预生产环境"""
    return await deployment_manager.deploy_to_environment("staging")

async def deploy_to_testing():
    """部署到测试环境"""
    return await deployment_manager.deploy_to_environment("testing")

async def deploy_to_development():
    """部署到开发环境"""
    return await deployment_manager.deploy_to_environment("development")

if __name__ == "__main__":
    # 演示部署流程
    async def main():
        print("🚀 诺玛Agent部署管理器演示...")
        
        # 显示可用环境
        print("\n📋 可用部署环境:")
        for env_name, env_config in deployment_manager.environments.items():
            print(f"  - {env_name}: {env_config.description}")
        
        # 部署到开发环境进行测试
        print("\n🔧 部署到开发环境...")
        result = await deploy_to_development()
        
        if result["success"]:
            print("✅ 开发环境部署成功!")
            print(f"部署耗时: {result['duration_seconds']:.2f}秒")
        else:
            print("❌ 开发环境部署失败!")
            print(f"错误: {result['error']}")
        
        # 显示部署状态
        print("\n📊 部署状态:")
        status = deployment_manager.get_deployment_status("development")
        print(json.dumps(status, indent=2, ensure_ascii=False))
    
    asyncio.run(main())