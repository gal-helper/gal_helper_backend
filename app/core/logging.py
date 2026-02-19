# app/core/logging.py
import logging
import os
import sys
from .config import config


def _configure_stdout_encoding():
    try:
        stdout = sys.stdout
        stderr = sys.stderr
        if hasattr(stdout, "reconfigure"):
            stdout.reconfigure(encoding="utf-8", errors="replace")
            stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def setup_logging():
    log_dir = os.path.dirname(config.LOG_FILE_NAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    _configure_stdout_encoding()
    # 打印到控制台和文件里，强制使用 UTF-8 编码
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(config.LOG_FILE_NAME, encoding="utf-8")  # 指定 encoding

    # 显式设置 stream_handler 的编码
    stream_handler.encoding = 'utf-8'

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[stream_handler, file_handler],
    )