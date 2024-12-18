import datetime
import os

class SimpleLogger:
    """
    简易日志记录类
    支持同时输出到控制台和文件
    
    Args:
        log_file: 日志文件路径，None则只输出到控制台
        append: 是否追加模式打开日志文件
    """
    def __init__(self, log_file=None, append=True):
        self.log_file = log_file
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.f = open(log_file, 'a' if append else 'w')
        else:
            self.f = None

    def log(self, msg, with_time=True):
        """
        记录日志
        Args:
            msg: 日志消息
            with_time: 是否添加时间戳
        """
        if with_time:
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{time_str}] {msg}"
        else:
            line = msg
            
        print(line)
        if self.f is not None:
            self.f.write(line + '\n')
            self.f.flush()

    def close(self):
        """关闭日志文件"""
        if self.f is not None:
            self.f.close()
            self.f = None

    def __del__(self):
        self.close()
