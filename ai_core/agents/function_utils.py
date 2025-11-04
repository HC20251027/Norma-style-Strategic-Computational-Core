import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional, cast

import aiohttp
from pydantic import BaseModel

ENV_AGENT_NAME = "AGENT_NAME"
ENV_FUNC_SERVER_PORT = "FUNC_SERVER_PORT"
MCP_FUNCTION_LIST_JSON_FILE = "mcp_function_list.json"

SERVER_PORT = 12306
PROXY_TIMEOUT = 3600


class ToolResult(BaseModel):
    """工具结果"""

    message: str
    is_error: bool


class FunctionProxy:
    def __init__(self, function_info: Dict[str, Any]):
        self.name: str = function_info["name"]
        self.origin_name: str | None = function_info.get("origin_name", None)
        self.params: List[Dict[str, Any]] = function_info["parameters"]
        self.kind: str = function_info.get("kind", "basic")
        self.params_len = len(self.params)
        self.agent_name: str = os.environ.get(ENV_AGENT_NAME, "")
        self.server_port = SERVER_PORT
        self.timeout: int = PROXY_TIMEOUT

    def get_server_url(self):
        if self.server_port == 0:
            raise Exception("PORT is not set, please set it in the environment variable")
        return f"http://localhost:{self.server_port}"

    async def __call__(self, *args, **kwargs) -> ToolResult:
        call_params = kwargs.copy()
        args_len = len(args)

        if self.kind == "mcp":
            call_params = cast(Dict[str, Any], args[0])
        else:
            # 将args中的参数按顺序赋值给call_params，确保是kv的形式
            for i in range(args_len):
                if i < self.params_len:
                    call_params[self.params[i]["name"]] = args[i]

        request = {
            "request_id": str(uuid.uuid4()),
            "function_name": self.origin_name or self.name,
            "function_kind": self.kind,
            "caller_name": self.agent_name,
            "parameters": call_params,
        }

        # 发出请求前的拦截
        tool_result = self._intercept_request(self.name, request)
        if tool_result is not None:
            return tool_result

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            try:
                async with session.post(f"{self.get_server_url()}/execute", json=request) as response:
                    if response.status != 200:
                        return ToolResult(is_error=True, message=f"Function call failed: {await response.text()}")

                    result = await response.json()
                    if result.get("is_error", False):
                        return ToolResult(is_error=True, message=result.get("message", "Unknown error"))

                    tool_result = ToolResult(is_error=False, message=result.get("message", "succeed"))
                    return self._intercept_response(self.name, request, tool_result)
            except asyncio.TimeoutError:
                error_msg = f"Timeout when calling function {self.name}"
                return ToolResult(is_error=True, message=error_msg)
            except Exception as e:
                import traceback

                error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                return ToolResult(is_error=True, message=error_msg)

    def _intercept_request(self, function_name: str, request: Dict[str, Any]) -> Optional[ToolResult]:
        if self.kind == "agent" and self.agent_name and "planner" not in self.agent_name:
            return ToolResult(is_error=True, message=f"Function {function_name} not found")
        return None

    def _intercept_response(self, function_name: str, request: Dict[str, Any], result: ToolResult) -> ToolResult:
        if function_name == "task_done":
            # 打印日志，为了用stdout解析出来
            print(f"task_done>>>{result.model_dump_json()}<<<task_done")
        return result


def load_function_proxys(file_path: str) -> tuple[List[Dict[str, Any]], Dict[str, FunctionProxy]]:
    # 加载 function_list.json 并创建 function proxies
    with open(file_path, "r", encoding="utf-8") as f:
        function_list = json.load(f)

    proxies = {}
    for function_info in function_list:
        if isinstance(function_info, dict) and "name" in function_info:
            proxies[function_info["name"]] = FunctionProxy(function_info)

    return function_list, proxies
