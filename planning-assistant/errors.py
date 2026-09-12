class ConfigError(Exception):
    """配置错误，比如 API Key 缺失"""
    pass

class APIError(Exception):
    """API 调用失败"""
    pass