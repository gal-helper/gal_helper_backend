import logging
from config import config

def setup_logging():
    # 屏幕/docker上的日志
    stream_handler = logging.StreamHandler()
    # 文件日志
    file_handler = logging.FileHandler(config.LOG_FILE_NAME)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            stream_handler,
            file_handler
        ]
    )