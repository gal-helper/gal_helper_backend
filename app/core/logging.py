import logging
import os
from .config import config


def setup_logging():
    log_dir = os.path.dirname(config.LOG_FILE_NAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 打印到控制台和文件里
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(config.LOG_FILE_NAME)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[stream_handler, file_handler],
    )
