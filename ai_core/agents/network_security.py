"""
网络安全工具实现
"""

import socket
import logging
import asyncio
import time
from typing import Dict, Any, List, Set
from datetime import datetime
import ipaddress
import subprocess
import json

from .base_tool import NetworkSecurityTool
from ..core.models import ToolParameter, ToolCategory, SecurityLevel


logger = logging.getLogger(__name__)


class PortScannerTool(NetworkSecurityTool):
    """端口扫描工具"""
    
    def get_tool_definition(self):
        from ..core.models import ToolDefinition
        return ToolDefinition(
            id="port_scanner",
            name="端口扫描器",
            description="扫描目标主机的开放端口",
            category=ToolCategory.NETWORK_SECURITY,
            security_level=SecurityLevel.HIGH,
            parameters=[
                ToolParameter("target", "str", "目标主机", True),
                ToolParameter("ports", "str", "端口范围或列表", False, "1-1000"),
                ToolParameter("timeout", "float", "连接超时时间", False, 1.0),
                ToolParameter("thread_count", "int", "并发线程数", False, 100)
            ],
            timeout=300,
            tags=["port", "scan", "security"],
            dependencies=["nmap"]
        )
    
    async def execute(self, target: str, ports: str = "1-1000", timeout: float = 1.0, thread_count: int = 100) -> Dict[str, Any]:
        """执行端口扫描"""
        self._update_execution_stats()
        
        start_time = time.time()
        
        try:
            # 解析端口范围
            port_list = self._parse_port_range(ports)
            
            # 验证目标主机
            if not self._validate_target(target):
                return {
                    "success": False,
                    "error": "无效的目标主机"
                }
            
            # 执行扫描
            open_ports = await self._scan_ports(target, port_list, timeout, thread_count)
            
            # 获取端口服务信息
            services = await self._get_port_services(target, open_ports)
            
            scan_time = time.time() - start_time
            
            return {
                "success": True,
                "target": target,
                "ports_scanned": len(port_list),
                "open_ports": sorted(open_ports),
                "services": services,
                "scan_time": round(scan_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"端口扫描失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "target": target
            }
    
    def _parse_port_range(self, ports: str) -> List[int]:
        """解析端口范围"""
        port_list = []
        
        # 处理逗号分隔的端口
        for part in ports.split(','):
            part = part.strip()
            
            # 处理范围
            if '-' in part:
                start, end = map(int, part.split('-'))
                port_list.extend(range(start, end + 1))
            else:
                port_list.append(int(part))
        
        return sorted(set(port_list))  # 去重并排序
    
    def _validate_target(self, target: str) -> bool:
        """验证目标主机"""
        try:
            # 尝试解析为IP地址
            ipaddress.ip_address(target)
            return True
        except ValueError:
            try:
                # 尝试解析为网络
                ipaddress.ip_network(target, strict=False)
                return True
            except ValueError:
                # 尝试解析域名
                socket.gethostbyname(target)
                return True
        except:
            return False
    
    async def _scan_ports(self, target: str, ports: List[int], timeout: float, thread_count: int) -> Set[int]:
        """扫描端口"""
        open_ports = set()
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(thread_count)
        
        async def scan_port(port: int):
            async with semaphore:
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(target, port),
                        timeout=timeout
                    )
                    writer.close()
                    await writer.wait_closed()
                    return port
                except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                    return None
        
        # 创建任务
        tasks = [scan_port(port) for port in ports]
        
        # 执行扫描
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集开放端口
        for result in results:
            if result is not None and isinstance(result, int):
                open_ports.add(result)
        
        return open_ports
    
    async def _get_port_services(self, target: str, ports: Set[int]) -> Dict[int, str]:
        """获取端口服务信息"""
        services = {}
        
        # 常见端口服务映射
        common_services = {
            21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
            80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 993: "IMAPS",
            995: "POP3S", 3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
            5900: "VNC", 6379: "Redis", 8080: "HTTP-Alt", 8443: "HTTPS-Alt"
        }
        
        for port in ports:
            if port in common_services:
                services[port] = common_services[port]
            else:
                # 尝试获取服务信息
                try:
                    service = socket.getservbyport(port)
                    services[port] = service
                except OSError:
                    services[port] = "unknown"
        
        return services


class VulnerabilityScannerTool(NetworkSecurityTool):
    """漏洞扫描工具"""
    
    def get_tool_definition(self):
        from ..core.models import ToolDefinition
        return ToolDefinition(
            id="vulnerability_scanner",
            name="漏洞扫描器",
            description="扫描目标主机的安全漏洞",
            category=ToolCategory.NETWORK_SECURITY,
            security_level=SecurityLevel.CRITICAL,
            parameters=[
                ToolParameter("target", "str", "目标主机", True),
                ToolParameter("scan_type", "str", "扫描类型", True, options=["quick", "full", "custom"]),
                ToolParameter("vuln_db", "str", "漏洞数据库路径", False, None),
                ToolParameter("timeout", "int", "扫描超时时间", False, 300)
            ],
            timeout=600,
            tags=["vulnerability", "scan", "security"],
            dependencies=["nmap", "nse"]
        )
    
    async def execute(self, target: str, scan_type: str = "quick", vuln_db: str = None, timeout: int = 300) -> Dict[str, Any]:
        """执行漏洞扫描"""
        self._update_execution_stats()
        
        start_time = time.time()
        
        try:
            # 构建nmap命令
            nmap_args = self._build_nmap_command(target, scan_type)
            
            # 执行扫描
            process = await asyncio.create_subprocess_exec(
                *nmap_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                scan_time = time.time() - start_time
                
                # 解析扫描结果
                scan_results = self._parse_nmap_output(stdout.decode())
                
                # 分析漏洞
                vulnerabilities = await self._analyze_vulnerabilities(target, scan_results, vuln_db)
                
                return {
                    "success": True,
                    "target": target,
                    "scan_type": scan_type,
                    "scan_results": scan_results,
                    "vulnerabilities": vulnerabilities,
                    "scan_time": round(scan_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
                
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": "扫描超时",
                    "target": target
                }
                
        except Exception as e:
            logger.error(f"漏洞扫描失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "target": target
            }
    
    def _build_nmap_command(self, target: str, scan_type: str) -> List[str]:
        """构建nmap命令"""
        base_cmd = ["nmap", "-sV", "-sC", "--script=vuln"]
        
        if scan_type == "quick":
            base_cmd.extend(["-T4", "--max-retries 2", "--host-timeout 60s"])
        elif scan_type == "full":
            base_cmd.extend(["-T1", "--max-retries 5", "--host-timeout 300s"])
        
        base_cmd.append(target)
        return base_cmd
    
    def _parse_nmap_output(self, output: str) -> Dict[str, Any]:
        """解析nmap输出"""
        results = {
            "hosts": [],
            "ports": {},
            "services": {},
            "os": {}
        }
        
        lines = output.split('\n')
        current_host = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Nmap scan report for"):
                current_host = line.split()[-1]
                results["hosts"].append(current_host)
                results["ports"][current_host] = []
                results["services"][current_host] = {}
                
            elif "open" in line and current_host:
                # 解析端口信息
                parts = line.split()
                if len(parts) >= 3:
                    port = parts[0].split('/')[0]
                    service = parts[2] if len(parts) > 2 else "unknown"
                    version = " ".join(parts[3:]) if len(parts) > 3 else ""
                    
                    results["ports"][current_host].append({
                        "port": int(port),
                        "service": service,
                        "version": version,
                        "state": "open"
                    })
                    
                    results["services"][current_host][port] = {
                        "name": service,
                        "version": version
                    }
        
        return results
    
    async def _analyze_vulnerabilities(self, target: str, scan_results: Dict[str, Any], vuln_db: str = None) -> List[Dict[str, Any]]:
        """分析漏洞"""
        vulnerabilities = []
        
        # 这里应该集成实际的漏洞数据库
        # 现在返回模拟数据
        
        for host, ports in scan_results.get("ports", {}).items():
            for port_info in ports:
                port = port_info["port"]
                service = port_info["service"]
                
                # 检查已知漏洞
                known_vulns = self._check_known_vulnerabilities(service, port_info.get("version", ""))
                
                for vuln in known_vulns:
                    vulnerabilities.append({
                        "host": host,
                        "port": port,
                        "service": service,
                        "vulnerability": vuln["name"],
                        "severity": vuln["severity"],
                        "description": vuln["description"],
                        "cve": vuln.get("cve", []),
                        "solution": vuln.get("solution", "")
                    })
        
        return vulnerabilities
    
    def _check_known_vulnerabilities(self, service: str, version: str) -> List[Dict[str, Any]]:
        """检查已知漏洞"""
        # 简化的漏洞数据库
        vuln_db = {
            "ftp": [
                {
                    "name": "FTP Anonymous Access",
                    "severity": "medium",
                    "description": "FTP服务器允许匿名访问",
                    "cve": [],
                    "solution": "禁用匿名访问或限制权限"
                }
            ],
            "ssh": [
                {
                    "name": "SSH Weak Encryption",
                    "severity": "low",
                    "description": "SSH服务器支持弱加密算法",
                    "cve": ["CVE-2016-10012"],
                    "solution": "更新SSH配置，禁用弱加密算法"
                }
            ],
            "http": [
                {
                    "name": "HTTP Security Headers Missing",
                    "severity": "low",
                    "description": "HTTP响应缺少安全头",
                    "cve": [],
                    "solution": "添加必要的安全头如X-Frame-Options, X-XSS-Protection等"
                }
            ]
        }
        
        return vuln_db.get(service.lower(), [])


class NetworkAnalyzerTool(NetworkSecurityTool):
    """网络流量分析工具"""
    
    def get_tool_definition(self):
        from ..core.models import ToolDefinition
        return ToolDefinition(
            id="network_analyzer",
            name="网络流量分析器",
            description="分析网络流量和连接",
            category=ToolCategory.NETWORK_SECURITY,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter("interface", "str", "网络接口", False, None),
                ToolParameter("duration", "int", "分析时长（秒）", False, 60),
                ToolParameter("filter", "str", "流量过滤条件", False, None),
                ToolParameter("output_format", "str", "输出格式", False, "json", options=["json", "pcap", "csv"])
            ],
            timeout=300,
            tags=["network", "traffic", "analysis"]
        )
    
    async def execute(self, interface: str = None, duration: int = 60, filter: str = None, output_format: str = "json") -> Dict[str, Any]:
        """执行网络流量分析"""
        self._update_execution_stats()
        
        start_time = time.time()
        
        try:
            # 使用tcpdump或tshark进行流量捕获
            capture_cmd = self._build_capture_command(interface, duration, filter)
            
            process = await asyncio.create_subprocess_exec(
                *capture_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=duration + 10
                )
                
                analysis_time = time.time() - start_time
                
                # 解析流量数据
                traffic_data = self._parse_traffic_data(stdout.decode())
                
                # 生成分析报告
                report = self._generate_traffic_report(traffic_data)
                
                return {
                    "success": True,
                    "interface": interface,
                    "duration": duration,
                    "filter": filter,
                    "output_format": output_format,
                    "analysis_time": round(analysis_time, 2),
                    "traffic_summary": report,
                    "raw_data": traffic_data[:100] if output_format == "json" else None,
                    "timestamp": datetime.now().isoformat()
                }
                
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": "流量分析超时",
                    "duration": duration
                }
                
        except Exception as e:
            logger.error(f"网络流量分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "interface": interface
            }
    
    def _build_capture_command(self, interface: str, duration: int, filter: str) -> List[str]:
        """构建流量捕获命令"""
        cmd = ["tcpdump"]
        
        if interface:
            cmd.extend(["-i", interface])
        
        if filter:
            cmd.extend(["-f", filter])
        
        cmd.extend([
            "-n",  # 不解析主机名
            "-tt",  # 时间戳
            "-c", "1000",  # 限制包数量
            "-w", "-",  # 输出到stdout
        ])
        
        return cmd
    
    def _parse_traffic_data(self, data: str) -> List[Dict[str, Any]]:
        """解析流量数据"""
        # 这里应该解析tcpdump输出
        # 简化实现
        packets = []
        lines = data.split('\n')
        
        for line in lines:
            if line.strip():
                # 简单的数据包解析
                parts = line.split()
                if len(parts) >= 6:
                    packet = {
                        "timestamp": parts[0],
                        "source": parts[2].split('.')[0] if '.' in parts[2] else parts[2],
                        "destination": parts[4].split('.')[0] if '.' in parts[4] else parts[4],
                        "protocol": parts[5] if len(parts) > 5 else "unknown"
                    }
                    packets.append(packet)
        
        return packets
    
    def _generate_traffic_report(self, traffic_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成流量分析报告"""
        if not traffic_data:
            return {"total_packets": 0}
        
        # 统计协议分布
        protocols = {}
        sources = {}
        destinations = {}
        
        for packet in traffic_data:
            # 协议统计
            proto = packet.get("protocol", "unknown")
            protocols[proto] = protocols.get(proto, 0) + 1
            
            # 源地址统计
            src = packet.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            
            # 目的地址统计
            dst = packet.get("destination", "unknown")
            destinations[dst] = destinations.get(dst, 0) + 1
        
        return {
            "total_packets": len(traffic_data),
            "protocols": protocols,
            "top_sources": dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_destinations": dict(sorted(destinations.items(), key=lambda x: x[1], reverse=True)[:10]),
            "unique_sources": len(sources),
            "unique_destinations": len(destinations)
        }