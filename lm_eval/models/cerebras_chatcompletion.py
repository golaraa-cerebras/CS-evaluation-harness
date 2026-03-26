import logging
import os
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


@register_model("cerebras-chat-completions")
class CerebrasChatCompletion(TemplateAPI):
    def __init__(self, **kwargs):
        kwargs.setdefault("model", "gpt-oss-120b")
        kwargs.setdefault("num_concurrent", 5)
        kwargs.setdefault("max_retries", 5)
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("tokenized_requests", False)
        super().__init__(
            base_url="https://api.cerebras.ai/v1/chat/completions",
            tokenizer_backend=None,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        load_dotenv()
        key = os.environ.get("CEREBRAS_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the CEREBRAS_API_KEY environment variable."
            )
        return key

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: str = None,
        **kwargs,
    ) -> dict:
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", None)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop] if stop else []
        return {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            try:
                tmp = [None] * len(out["choices"])
                for choice in out["choices"]:
                    tmp[choice["index"]] = choice["message"]["content"]
            except Exception as e:
                eval_logger.warning(f"Could not parse generations: {e}")
                tmp = [""]
            res = res + tmp
        return res

    @staticmethod
    def parse_logprobs(
        outputs: Union[Any, List[Any]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "Loglikelihood is not supported for Cerebras chat completions."
        )

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and multiple_choice-type tasks) is not supported for "
            "Cerebras chat completions because the API does not expose prompt logprobs."
        )


@register_model("cerebras-completions")
class CerebrasCompletion(TemplateAPI):
    """
    Cerebras completions endpoint — supports generation AND loglikelihood.

    Loglikelihood requires a tokenizer to compute context lengths.
    Pass a compatible HF tokenizer via model_args, e.g.:
        tokenizer=meta-llama/Llama-3.1-8B
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("model", "gpt-oss-120b")
        kwargs.setdefault("num_concurrent", 5)
        kwargs.setdefault("max_retries", 5)
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("tokenized_requests", False)
        super().__init__(
            base_url="https://api.cerebras.ai/v1/completions",
            tokenizer_backend=kwargs.pop("tokenizer_backend", "huggingface" if "tokenizer" in kwargs else None),
            **kwargs,
        )

    @cached_property
    def api_key(self):
        load_dotenv()
        key = os.environ.get("CEREBRAS_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the CEREBRAS_API_KEY environment variable."
            )
        return key

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos: str = None,
        **kwargs,
    ) -> dict:
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", None)
        if generate:
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            if not isinstance(stop, (list, tuple)):
                stop = [stop] if stop else []
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop[:4],
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 1,
                "echo": True,
                "seed": seed,
            }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choice in out["choices"]:
                tmp[choice["index"]] = choice["text"]
            res = res + tmp
        return res

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=lambda c: c["index"]), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                token_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                logprob_sum = sum(token_logprobs)
                is_greedy = all(
                    tok == max(top.values())
                    for tok, top in zip(token_logprobs, top_logprobs)
                )
                res.append((logprob_sum, is_greedy))
        return res
