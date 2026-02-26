#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gal Helper Backend - 自动启动脚本
使用 uvicorn 启动 FastAPI 服务
"""

import os
import sys
import subprocess
import signal
import time
import socket

# 配置
HOST = "0.0.0.0"
PORT = 8000
APP_MODULE = "app.main:app"
LOG_FILE = "auto_run.log"

# 添加当前目录到 PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONPATH"] = PROJECT_ROOT


def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_process_by_port(port: int) -> int:
    """获取占用指定端口的进程 PID"""
    try:
        result = subprocess.run(
            f"lsof -ti:{port}".split(),
            capture_output=True,
            text=True
        )
        if result.stdout:
            return int(result.stdout.strip().split('\n')[0])
    except:
        pass
    return None


def kill_process_on_port(port: int):
    """杀掉占用指定端口的进程"""
    pid = get_process_by_port(port)
    if pid:
        print(f"⚠️  端口 {port} 被占用，杀掉进程 {pid}")
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except:
            pass


def start_server():
    """启动 uvicorn 服务器"""
    # 检查端口
    if is_port_in_use(PORT):
        print(f"❌ 端口 {PORT} 已被占用")
        kill_process_on_port(PORT)
    
    print(f"🚀 启动 Gal Helper API 服务...")
    print(f"   项目路径: {PROJECT_ROOT}")
    print(f"   监听地址: http://{HOST}:{PORT}")
    print(f"   API 文档: http://{HOST}:{PORT}/docs")
    print(f"   日志文件: {LOG_FILE}")
    print("-" * 50)
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    
    # 启动命令
    cmd = [
        sys.executable, "-m", "uvicorn",
        APP_MODULE,
        "--host", HOST,
        "--port", str(PORT),
        "--reload"
    ]
    
    # 重定向日志到文件
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env
        )
        
        print(f"✅ 服务已启动! PID: {process.pid}")
        print(f"   按 Ctrl+C 停止服务")
        print("-" * 50)
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 停止服务...")
            process.terminate()
            process.wait()
            print("✅ 服务已停止")


if __name__ == "__main__":
    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    
    start_server()
