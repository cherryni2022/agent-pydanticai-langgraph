# pydantic-ai 集成 ZhipuAI (GLM) 模型支持文档

## 1. 背景与目标
原生的 `pydantic-ai` 库主要支持 OpenAI 等标准模型提供商，不直接支持智谱 AI (ZhipuAI/GLM)。为了在本项目中使用 GLM-4 等模型，同时保持 `pydantic-ai` 的 Agent 开发体验（如结构化输出、工具调用），我们需要自定义实现一个 `Model` 子类。

## 2. 核心功能实现

### 2.1 自定义模型类 (`ZhipuModel`)
我们创建了 `zhipu_model.py`，其中定义了 `ZhipuModel` 类，继承自 `pydantic_ai.models.Model`。

#### 主要职责：
1.  **消息映射 (`_map_messages`)**: 将 `pydantic-ai` 的 `ModelMessage`（包括 System, User, ToolReturn, RetryPrompt 等）转换为 zhipuai SDK 接受的消息格式（`role`, `content`, `tool_calls`, `tool_call_id` 等）。
2.  **工具映射 (`_map_tool_definition`)**: 将 `ToolDefinition` 转换为 GLM 兼容的 function calling 定义。
3.  **结果处理 (`_process_response`)**: 将 GLM 的 API 响应解析回 `ModelResponse`，提取文本内容和工具调用信息。

### 2.2 流式传输支持 (`StreamedResponse`)
为了支持 Agent 的流式输出（如打字机效果），我们必须实现 `request_stream` 方法。

#### 实现细节：
-   **`ZhipuStreamedResponse`**: 继承自 `pydantic_ai.models.StreamedResponse`。
-   **`_get_event_iterator`**: 这是一个异步生成器，用于逐个产出 `ModelResponseStreamEvent`。
-   **非阻塞调用**: 由于 `zhipuai` 目前的 SDK 主要提供同步生成器，我们在 `asyncio` 环境中运行时，直接迭代会阻塞事件循环。解决方案是使用 `loop.run_in_executor` 将同步的 `next()` 调用放到线程池中执行。

## 3. 遇到的问题与解决方案

### 问题 1: `pydantic-ai` 不支持 GLM
**报错信息**: `ModuleNotFoundError` 或不支持的 provider。
**解决**: 手动实现 `ZhipuModel` 类，完全兼容 `pydantic-ai` 的接口规范。

### 问题 2: `StopIteration` 交互错误
**报错信息**: `RuntimeError: StopIteration interacts badly with generators and cannot be raised into a Future`
**原因**: 在 `asyncio` 的 `run_in_executor` 中调用 `next(iterator)` 时，如果迭代器耗尽抛出 `StopIteration`，这个异常无法正确穿透 `Future` 被 `await` 捕获，导致运行时错误。
**解决**: 使用 `next(iterator, default)` 的形式，传入一个哨兵对象（sentinel）。当返回哨兵对象时，手动 `break` 循环，从而避免了 `StopIteration` 异常的抛出。

```python
_sentinel = object()
while True:
    chunk = await loop.run_in_executor(None, next, self._response_iter, _sentinel)
    if chunk is _sentinel:
        break
    # ... process chunk
```

### 问题 3: 类型检查 (Linting)
**原因**: `Mypy` 对 `pydantic-ai` 内部类型的严格检查（如 `tools` 列表的类型、`content` 的非空断言）。
**解决**:
-   显式添加类型注解：`tools: list[dict[str, Any]] = []`
-   增加非空判断：`if delta.content: yield ...`

## 4. 使用指南

### 4.1 环境变量
在 `.env` 文件中配置：
```bash
PROVIDER=zhipu
LLM_API_KEY=your_zhipu_api_key
MODEL_CHOICE=glm-4
```

### 4.2 代码集成
在 `utils.py` 中，根据 `PROVIDER` 环境变量动态返回模型实例：

```python
if provider == 'zhipu':
    return ZhipuModel(model_name, api_key=api_key)
else:
    return OpenAIModel(...)
```

### 4.3 测试验证
运行提供的测试脚本验证功能：
-   `python test_zhipu_agent.py`: 验证基本的打招呼和结构化输出。
-   `python test_zhipu_stream.py`: 验证流式输出功能。
