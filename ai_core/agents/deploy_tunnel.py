#!/usr/bin/env python3
"""
简单的隧道部署脚本
使用可用的工具创建外部可访问的隧道
"""

import subprocess
import time
import requests
import threading
import socket

def check_port_available(port):
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def create_simple_tunnel(local_port, tunnel_port):
    """创建简单的隧道"""
    print(f"尝试创建隧道: localhost:{local_port} -> 外部端口:{tunnel_port}")
    
    # 尝试使用Python的http.server模块创建隧道
    tunnel_script = f'''
import http.server
import socketserver
import urllib.request
import urllib.parse

class TunnelHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        target_url = f"http://localhost:{local_port}{{self.path}}"
        try:
            response = urllib.request.urlopen(target_url)
            content = response.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(502, str(e))

with socketserver.TCPServer(("", {tunnel_port}), TunnelHandler) as httpd:
    print("隧道服务器运行在端口 {tunnel_port}")
    httpd.serve_forever()
'''
    
    with open(f'/tmp/tunnel_{tunnel_port}.py', 'w') as f:
        f.write(tunnel_script)
    
    try:
        subprocess.Popen(['python', f'/tmp/tunnel_{tunnel_port}.py'])
        time.sleep(2)
        print(f"隧道创建成功: localhost:{tunnel_port}")
        return True
    except Exception as e:
        print(f"隧道创建失败: {e}")
        return False

def main():
    print("=== 诺玛AI系统隧道部署 ===")
    
    # 检查代理服务状态
    print("检查代理服务状态...")
    
    # 测试HTTP代理
    try:
        response = requests.get('http://localhost:9000/api/system/status', timeout=5)
        if response.status_code == 200:
            print("✓ HTTP代理服务正常 (localhost:9000)")
        else:
            print("✗ HTTP代理服务异常")
    except Exception as e:
        print(f"✗ HTTP代理服务连接失败: {e}")
    
    # 创建外部隧道
    tunnels_created = []
    
    # 为HTTP代理创建隧道
    if create_simple_tunnel(9000, 9002):
        tunnels_created.append(("HTTP", 9002))
    
    # 为WebSocket代理创建隧道
    if create_simple_tunnel(9001, 9003):
        tunnels_created.append(("WebSocket", 9003))
    
    print("\n=== 隧道信息 ===")
    for tunnel_type, port in tunnels_created:
        print(f"{tunnel_type}隧道: http://localhost:{port}")
    
    print("\n注意: 这些隧道仅在本地可用。要让外部访问，需要部署到云服务。")
    
    return tunnels_created

if __name__ == "__main__":
    main()
