import traceback

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette import status


# 开发模式：返回详细错误信息
# 生产模式：返回简化错误信息
DEBUG_MODE = True # 开发模式

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    处理HTTP异常
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "data": None,
        },
    )

async def integrity_error_handler(request: Request, exc: IntegrityError):
    """
    处理数据库完整性约束错误 IntegrityError异常
    """
    error_msg = str(exc.orig)

    # 开发模式下输出错误信息
    error_data = None
    detail = "数据约束冲突，请检查输入"
    if DEBUG_MODE:
        error_data = {
            "error_type": "IntegrityError",
            "error_detail": error_msg,
            "path": str(request.url)
        }

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "code": status.HTTP_400_BAD_REQUEST,
            "message": detail,
            "data": error_data,
        },
    )

async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    """
    处理SQLAlchemy异常
    """
    # 获取错误信息
    error_data = None
    if DEBUG_MODE:
        error_data = {
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
            # 格式化异常信息为字符串，方便日志记录和调试
            "tracback": traceback.format_exc(),
            "path": str(request.url)
        }

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "数据库操作失败，请稍后重试",
            "data": error_data,
        },
    )

async def general_exception_handler(request: Request, exc: Exception):
    """
    处理其他异常
    """
    # 获取错误信息
    error_data = None
    if DEBUG_MODE:
        error_data = {
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
            # 获取异常的调用堆栈信息
            "tracback": traceback.format_exc(),
            "path": str(request.url)
        }

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "服务器内部错误，请稍后重试",
            "data": error_data,
        }
    )