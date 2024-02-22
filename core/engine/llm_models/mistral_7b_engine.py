## TO DO : BASE MODEL ENGINES TO BE ADDED.

import os
import sys
from queue import Empty
from threading import Thread
from typing import *
from uuid import UUID

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.llms import Ollama
from langchain_core.callbacks.base import BaseCallbackHandler


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        return self.q.empty()


class StreamingWebCallbackHandler(BaseCallbackHandler):
    tokens: List[str] = []
    is_responding: bool = False
    response_id: str
    response: str = None

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()
        self.tokens.append(token)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> Any:
        self.is_responding = True
        self.response_id = run_id
        self.response = None

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        self.is_responding = False
        self.response = outputs["response"]
        print("END: " + self.response)

    def get_response(self) -> str:
        response_result = self.response
        self.response = None

        return response_result


def stream(cb, queue):
    job_done = object()

    def task():
        queue.put(job_done)

    t = Thread(target=task)
    t.start()

    while True:
        try:
            item = queue.get(True, timeout=1)
            if item is job_done:
                break
            yield item
        except Empty:
            continue


# TO DO : Fix the pipeline. It should have session for conversations, states should be kept in the session.
# Sessions are simply conversations.
# TO DO : Redis can be used for cacheing time-span calls.


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class Mistral7BInstructModel:
    def __init__(
        self, model_path
    ) -> None:  # TO DO : Model configs can be jsonable later on for distribution.
        self.model_path = model_path

    def load_model(self):
        return Ollama(
            model="aisherpa/mistral-7b-instruct-v02:Q5_K_M",
            callback_manager=CallbackManager([AsyncIteratorCallbackHandler()]),
        )
