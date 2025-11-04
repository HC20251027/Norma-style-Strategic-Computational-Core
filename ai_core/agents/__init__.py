import os

from external_api.data_sources import *
from external_api.function_utils import MCP_FUNCTION_LIST_JSON_FILE, ToolResult, load_function_proxys

proxies = {}
_, proxies = load_function_proxys(os.path.join(os.path.dirname(__file__), MCP_FUNCTION_LIST_JSON_FILE))
globals().update(proxies)

__all__ = ["ToolResult"] + list(proxies.keys())

if __name__ == "__main__":
    print(__all__)
    print(globals())
