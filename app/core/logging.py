import logging
import os
from .config import config


def setup_logging():
    log_dir = os.path.dirname(config.LOG_FILE_NAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 打印到控制台和文件里，强制使用 UTF-8 编码
    stream_handler = logging.StreamHandler(encoding="utf-8")
    file_handler = logging.FileHandler(config.LOG_FILE_NAME, encoding="utf-8")

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[stream_handler, file_handler],
    )
