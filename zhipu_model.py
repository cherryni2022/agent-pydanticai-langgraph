from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, AsyncIterator
from datetime import datetime, timezone
from typing import Literal, Any, cast
from functools import partial
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    ModelResponseStreamEvent,
    ModelResponsePartsManager
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage
from zhipuai import ZhipuAI
from zhipuai.types.chat.chat_completion import CompletionUsage


@dataclass
class ZhipuStreamedResponse(StreamedResponse):
    _model_name: str
    _response_iter: Any
    _timestamp: datetime
    _parts_manager: ModelResponsePartsManager = field(default_factory=ModelResponsePartsManager, init=False)
    _usage: Usage = field(default_factory=Usage, init=False)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        loop = asyncio.get_running_loop()
        
        _sentinel = object()
        while True:
            # Iterate the sync iterator in executor
            chunk = await loop.run_in_executor(None, next, self._response_iter, _sentinel)
            if chunk is _sentinel:
                break

            # Process chunk
            if not chunk.choices:
                # Verify usage in final chunk?
                if chunk.usage:
                     self._usage = Usage(
                        request_tokens=chunk.usage.prompt_tokens,
                        response_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens
                     )
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            
            # content
            if delta.content:
                yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=delta.content)
            
            # tool calls
            if delta.tool_calls:
                 for tc in delta.tool_calls:
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=tc.index,
                        tool_name=tc.function.name,
                        args=tc.function.arguments,
                        tool_call_id=tc.id,
                    )
                    if maybe_event is not None:
                        yield maybe_event
            
            # usage (sometimes in chunk with choices or separate)
            if chunk.usage:
                 self._usage = Usage(
                    request_tokens=chunk.usage.prompt_tokens,
                    response_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                 )


class ZhipuModel(Model):
    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        zhipu_client: ZhipuAI | None = None,
    ):
        self._model_name = model_name
        if zhipu_client:
            self.client = zhipu_client
        else:
            self.client = ZhipuAI(api_key=api_key, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return 'zhipuai'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> tuple[ModelResponse, Usage]:
        
        # Prepare parameters
        tools: list[dict[str, Any]] = []
        if model_request_parameters and model_request_parameters.function_tools:
            tools.extend(
                self._map_tool_definition(t) 
                for t in model_request_parameters.function_tools
            )
        
        if model_request_parameters and model_request_parameters.result_tools:
             tools.extend(
                self._map_tool_definition(t) 
                for t in model_request_parameters.result_tools
            )

        # Map messages
        glm_messages = self._map_messages(messages)

        # Let's check model settings
        kwargs = {}
        if model_settings:
            if model_settings.max_tokens is not None:
                kwargs['max_tokens'] = model_settings.max_tokens
            if model_settings.temperature is not None:
                kwargs['temperature'] = model_settings.temperature
            if model_settings.top_p is not None:
                kwargs['top_p'] = model_settings.top_p

        if tools:
            kwargs['tools'] = tools
            if model_request_parameters and not model_request_parameters.allow_text_result and model_request_parameters.result_tools:
                 kwargs['tool_choice'] = 'auto'

        # Make request
        # We'll use the sync client in a non-blocking way
        loop = asyncio.get_running_loop()
        func = partial(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=glm_messages,
            **kwargs
        )
        response = await loop.run_in_executor(None, func)
        
        return self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        
        # Prepare parameters
        tools: list[dict[str, Any]] = []
        if model_request_parameters and model_request_parameters.function_tools:
            tools.extend(
                self._map_tool_definition(t) 
                for t in model_request_parameters.function_tools
            )
        
        if model_request_parameters and model_request_parameters.result_tools:
             tools.extend(
                self._map_tool_definition(t) 
                for t in model_request_parameters.result_tools
            )

        glm_messages = self._map_messages(messages)

        kwargs = {}
        if model_settings:
            if model_settings.max_tokens is not None:
                kwargs['max_tokens'] = model_settings.max_tokens
            if model_settings.temperature is not None:
                kwargs['temperature'] = model_settings.temperature
            if model_settings.top_p is not None:
                kwargs['top_p'] = model_settings.top_p

        if tools:
            kwargs['tools'] = tools
            if model_request_parameters and not model_request_parameters.allow_text_result and model_request_parameters.result_tools:
                 kwargs['tool_choice'] = 'auto'

        kwargs['stream'] = True

        loop = asyncio.get_running_loop()
        func = partial(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=glm_messages,
            **kwargs
        )
        # response_iter is the sync iterator from zhipuai
        response_iter = await loop.run_in_executor(None, func)
        
        streamed_response = ZhipuStreamedResponse(
            _model_name=self.model_name,
            _response_iter=response_iter,
            _timestamp=datetime.now(timezone.utc)
        )
        
        yield streamed_response

    def _map_messages(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        glm_messages = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        glm_messages.append({'role': 'system', 'content': part.content})
                    elif isinstance(part, UserPromptPart):
                        glm_messages.append({'role': 'user', 'content': part.content})
                    elif isinstance(part, ToolReturnPart):
                        # Tool return is a tool message
                        glm_messages.append({
                            'role': 'tool',
                            'tool_call_id': part.tool_call_id,
                            'content': part.content if isinstance(part.content, str) else json.dumps(part.content)
                        })
                    elif isinstance(part, RetryPromptPart):
                        # Retry prompt is treated as user message
                        if part.content:
                             glm_messages.append({'role': 'user', 'content': part.content})
            elif isinstance(msg, ModelResponse):
                # Assistant message
                content = None
                tool_calls = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        content = part.content
                    elif isinstance(part, ToolCallPart):
                        tool_calls.append({
                            'id': part.tool_call_id,
                            'type': 'function',
                            'function': {
                                'name': part.tool_name,
                                'arguments': json.dumps(part.args) if isinstance(part.args, dict) else part.args
                            }
                        })
                
                message_dict = {'role': 'assistant'}
                if content:
                    message_dict['content'] = content
                if tool_calls:
                   message_dict['tool_calls'] = tool_calls
                
                glm_messages.append(message_dict)
                
        return glm_messages

    def _map_tool_definition(self, tool: ToolDefinition) -> dict[str, Any]:
        description = tool.description
        # Zhipu/GLM usually requires description
        if not description:
            description = tool.name # Fallback
            
        return {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': description,
                'parameters': tool.parameters_json_schema,
            },
        }

    def _process_response(self, response) -> tuple[ModelResponse, Usage]:
        choices = response.choices
        if not choices:
            # Handle empty choice or raise error
            # Some models return empty choice if only usage is present?
            pass
        
        choice = choices[0]
        message = choice.message
        
        parts = []
        if message.content:
            parts.append(TextPart(content=message.content))
        
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                parts.append(ToolCallPart(
                    tool_name=tc.function.name,
                    args=args, # pydantic-ai will parse integer/json string
                    tool_call_id=tc.id
                ))
        
        usage = Usage(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        return ModelResponse(parts=parts, timestamp=datetime.now()), usage
