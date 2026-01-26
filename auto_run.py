import os
import sys
import subprocess

def get_user_input():
    print("AI RAG API Systemd")
    print("=" * 50)
    
    project_path = os.getcwd()
    python_path = "/root/miniconda3/bin/python"
    service_name = "ai-rag-api"
    
    return {
        'project_path': project_path,
        'python_path': python_path,
        'service_name': service_name
    }

def create_service_file(config):
    service_content = f"""[Unit]
Description=AI RAG API Service
After=network.target

[Service]
User=root
WorkingDirectory={config['project_path']}
ExecStart={config['python_path']} api_server.py
Restart=always
RestartSec=3
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier={config['service_name']}

[Install]
WantedBy=multi-user.target
"""
    
    service_file = f"/etc/systemd/system/{config['service_name']}.service"
    
    print(f"\n正在创建服务文件: {service_file}")
    print("-" * 40)
    print(service_content)
    print("-" * 40)
    
    try:
        with open(service_file, 'w') as f:
            f.write(service_content)
        print(f"服务文件创建成功")
        return True
    except PermissionError:
        print(f"权限不足，请使用 sudo 运行此脚本")
        return False
    except Exception as e:
        print(f"创建失败: {e}")
        return False

def setup_service(config):
    service_name = config['service_name']
    
    commands = [
        ("重新加载 systemd", f"systemctl daemon-reload"),
        ("启用开机启动", f"systemctl enable {service_name}"),
        ("启动服务", f"systemctl start {service_name}"),
        ("检查服务状态", f"systemctl status {service_name} --no-pager -l")
    ]
    
    print(f"\n正在配置服务: {service_name}")
    print("=" * 50)
    
    for description, cmd in commands:
        print(f"\n{description}...")
        print(f"命令: {cmd}")
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            if result.stdout:
                print("输出:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"     {line}")
            
            if result.stderr and "Warning" not in result.stderr:
                print(f"   ⚠️ 错误: {result.stderr.strip()}")
            
            if result.returncode != 0 and description != "检查服务状态":
                print(f"命令执行失败")
                return False
                
        except Exception as e:
            print(f"执行异常: {e}")
            return False
    
    return True

def main():
    if os.geteuid() != 0:
        print("请使用 sudo 运行此脚本:")
        print(f"   sudo python3 {sys.argv[0]}")
        sys.exit(1)
    
    config = get_user_input()
    
    if not os.path.exists(os.path.join(config['project_path'], 'api_server.py')):
        print(f"\n错误: 在 {config['project_path']} 中未找到 api_server.py")
        print("请确认项目路径是否正确")
        sys.exit(1)
    
    if not os.path.exists(config['python_path']):
        print(f"\n错误: Python 路径不存在: {config['python_path']}")
        print("请使用 'which python3' 命令查看正确的路径")
        sys.exit(1)
    
    print(f"\n配置摘要:")
    print(f"   项目路径: {config['project_path']}")
    print(f"   Python路径: {config['python_path']}")
    print(f"   服务名称: {config['service_name']}")
    
    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("操作已取消")
        sys.exit(0)
    
    # 执行配置
    if create_service_file(config) and setup_service(config):
        print("\n" + "=" * 50)
        print("服务配置完成!")
        print("=" * 50)
        print(f"\n服务名称: {config['service_name']}")
        print("已设置为开机自启")
        print("服务已启动")
        print(f"\n管理命令:")
        print(f"查看状态: sudo systemctl status {config['service_name']}")
        print(f"查看日志: sudo journalctl -u {config['service_name']} -f")
        print(f"重启服务: sudo systemctl restart {config['service_name']}")
        print(f"停止服务: sudo systemctl stop {config['service_name']}")
        print(f"\n访问地址: http://101.132.36.117:8000")
    else:
        print("\n配置失败，请检查错误信息")

if __name__ == "__main__":
    main()