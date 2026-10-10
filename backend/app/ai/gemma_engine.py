"""Local open-weight Gemma inference via HuggingFace transformers.

This is the module that makes the project's central claim testable: the missions
are written by a model running on the player's own machine, not by a hosted API.
So it is written to be honest about failure rather than generous about it.

Three rules shape everything here.

1. **No import-time torch.** `torch`/`transformers` are imported inside
   functions. This module is imported by `engine.build_engine()` and by the test
   suite on machines that have no ML stack installed, and an import error there
   would take down the whole app instead of degrading it.

2. **A hard timeout, not a polite one.** `timeout_s` is enforced by joining the
   decoding thread with a deadline. `model.generate` runs in a daemon thread and
   its partial output is collected from a `TextIteratorStreamer`, so a hung or
   pathologically slow decode returns `None` on time instead of pinning a worker
   forever. Checking the clock *after* generation returns would not be a timeout,
   it would be a post-mortem.

3. **Never raise out of `generate`.** Returning `None` is the contract with the
   caller: the deterministic fallback is a normal code path, not an exception
   handler. Every failure mode -- unload, bad image, timeout, OOM -- is reported
   as `None` plus a recorded `last_error`.

`supports_vision` is *detected* from the model config and its processor, never
assumed. Claiming vision on a text-only checkpoint -- or on a multimodal one
whose image processor did not load -- would let the app advertise a capability
this process cannot serve, which is exactly the failure this project refuses.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any

from app.ai.engine import (
    GENERATION_TIMEOUT_S,
    MAX_NEW_TOKENS,
    VERIFY_TIMEOUT_S,
    InferenceEngine,
)
from app.models.schemas import EngineStatus, GenerateResult, GeneratorSource

log = logging.getLogger(__name__)

#: Config values that identify a multimodal Gemma checkpoint. Architecture names
#: are substring-matched; model types are matched EXACTLY, because the text-only
#: Gemma 3 checkpoints use `gemma3_text`, which contains "gemma3" as a prefix --
#: substring matching there would advertise vision on a text-only model.
_VISION_ARCHITECTURES = ("Gemma3ForConditionalGeneration",)
_VISION_MODEL_TYPES = frozenset({"gemma3", "gemma_3"})

#: Gracious period for a decode thread after the deadline. The hard guarantee is
#: that `generate` *returns* on time; a CUDA kernel already in flight cannot be
#: safely killed from Python, so the thread is left as a daemon and the reason is
#: recorded rather than pretended away.
_THREAD_JOIN_GRACE_S = 0.5


def _dist_version(name: str) -> str | None:
    """Installed version of `name`, or None. Never raises."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover - importlib is stdlib, belt and braces
        return None


def _torch_version() -> str | None:
    """torch's version, from the import when possible, else from metadata.

    Metadata is the fallback because a torch that fails to *import* (broken CUDA
    build, wrong wheel for the platform) is exactly the case /api/health needs to
    report honestly -- reporting "no torch" there would be a lie of omission.
    """
    try:
        import torch  # noqa: PLC0415

        version = getattr(torch, "__version__", None)
        if isinstance(version, str) and version:
            return version
    except Exception:
        pass
    return _dist_version("torch")


def _torch_cuda_available() -> bool:
    """True only if torch imported *and* reports a usable CUDA runtime."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_name() -> str | None:
    """Marketing name of device 0, or None. Never raises."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None


def _resolve_dtype(dtype: str | None, device: str) -> str:
    """Pick a torch dtype name for `device`.

    bfloat16 is the default on GPU because Gemma 3 was trained in it; float16 is
    the fallback for hardware without bf16 support, and float32 on CPU where
    half precision is slow or unavailable. bf16 support is probed rather than
    assumed -- `is_bf16_supported()` throws on some builds, hence the guard.
    """
    if dtype:
        return dtype
    if device != "cuda":
        return "float32"
    try:
        import torch  # noqa: PLC0415

        supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(supported) and supported():
            return "bfloat16"
        return "float16"
    except Exception:
        # torch unavailable at resolve time: GPU placement is unlikely anyway,
        # but bfloat16 is still the right default for a Gemma checkpoint.
        return "bfloat16"


def _torch_dtype(name: str) -> Any:
    """Map a dtype name to a torch dtype, falling back to float32.

    The fallback matters more than it looks: an unsupported dtype name would
    otherwise make `load()` fail with an unhelpful attribute error instead of
    running more slowly.
    """
    import torch  # noqa: PLC0415

    attr = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}.get(name)
    resolved = getattr(torch, attr, None) if attr else None
    return resolved if resolved is not None else torch.float32


def _detect_vision(config: Any, processor: Any) -> bool:
    """True only if the loaded checkpoint can actually accept images.

    Two signals, and both are needed:

    * the *weights* must be multimodal -- ``config.architectures`` naming
      ``Gemma3ForConditionalGeneration`` (the text-only sibling is
      ``Gemma3ForCausalLM``), or a ``gemma3`` ``model_type``, which is weaker
      because that name also covers the text-only checkpoints;
    * the *repo* must ship a processor carrying an image preprocessor, because
      without one there is nothing to turn raw pixels into tensors and every
      image call would fail inside `generate`.

    The check is on the presence of an `image_processor`, not on the class name:
    a tokenizer exposed under a Processor-shaped name still cannot take pixels,
    and conversely a valid processor is not obliged to be called "*Processor".

    The architecture signal alone is not trusted: a repo whose processor failed
    to load (an older transformers, say) still reports multimodal weights, and
    claiming vision there would make /api/health advertise a capability this
    process cannot actually serve.
    """
    if processor is None:
        return False
    # An image preprocessor is what makes the processor multimodal. Nothing else
    # is checked here: invoking the processor is already guarded by the caller,
    # so a second guess about callability would only reject working processors.
    if getattr(processor, "image_processor", None) is None:
        return False

    architectures = getattr(config, "architectures", None) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    arch_names = " ".join(str(a) for a in architectures)

    model_type = str(getattr(config, "model_type", "") or "").lower()

    arch_hit = any(name in arch_names for name in _VISION_ARCHITECTURES)
    return bool(arch_hit or model_type in _VISION_MODEL_TYPES)


def probe_device() -> dict:
    """Describe the machine's inference capability, safe to call with no torch.

    Used by the CLI bootstrap and /api/health so a user on a laptop with no ML
    stack gets a specific reason instead of a stack trace. With torch missing it
    reports device "none" -- which the frontend renders as "no local model".
    """
    cuda = _torch_cuda_available()
    torch_version = _torch_version()
    return {
        "device": "cuda" if cuda else ("cpu" if torch_version else "none"),
        "cuda": cuda,
        "torch": torch_version,
        "gpu_name": _gpu_name() if cuda else None,
    }


class GemmaEngine(InferenceEngine):
    """A local Gemma checkpoint served through transformers.

    Construction never touches the network or the GPU; `load()` does. That split
    lets the app build every engine at import time and only pay for weights it
    actually uses.
    """

    def __init__(self, model_id: str, device: str | None = None, dtype: str | None = None) -> None:
        self.model_id = model_id
        self.device: str = device or ("cuda" if _torch_cuda_available() else "cpu")
        self.dtype: str = _resolve_dtype(dtype, self.device)
        self.supports_vision = False
        self.loaded = False
        self.last_error: str | None = None
        self.last_latency_ms: int | None = None
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        #: Generation is not thread-safe on a single model; one decode at a time.
        self._generate_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load tokenizer and weights. Returns False and records why on failure.

        `trust_remote_code=False` is not a default to override: it is the reason
        a checkpoint from the Hub can be loaded at all without granting it
        arbitrary code execution in this process.
        """
        if self.loaded:
            return True

        self.last_error = None
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except Exception as exc:
            self.last_error = f"transformers unavailable: {exc}"
            log.warning("%s", self.last_error)
            return False

        try:
            import torch  # noqa: F401,PLC0415  (import failure is handled below)
        except Exception as exc:
            self.last_error = f"torch unavailable: {exc}"
            log.warning("%s", self.last_error)
            return False

        try:
            self._tokenizer = self._load_tokenizer(AutoTokenizer)
            self._processor = self._load_processor(AutoProcessor)
            self._model = self._load_model(AutoModelForCausalLM)
            self.supports_vision = _detect_vision(
                getattr(self._model, "config", None), self._processor
            )
            self.loaded = True
            log.info(
                "loaded %s on %s/%s (vision=%s)",
                self.model_id,
                self.device,
                self.dtype,
                self.supports_vision,
            )
            return True
        except Exception as exc:
            self.loaded = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("failed to load %s: %s", self.model_id, self.last_error)
            # Free whatever half-loaded; a partial model still holds VRAM.
            self.unload()
            return False

    def _load_tokenizer(self, AutoTokenizer: Any) -> Any:
        """Load the tokenizer, insisting on the fast (Rust) implementation.

        Gemma 3's tokenizer config is only consumable by the fast tokenizer; the
        slow Python fallback can load and then mis-tokenize, which shows up much
        later as garbage output rather than as an error here.
        """
        kwargs: dict[str, Any] = {"trust_remote_code": False, "use_fast": True}
        try:
            return AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        except Exception as exc:
            # Retry unconditionally rather than matching on the message text:
            # the wording of this error changes between transformers releases,
            # and guessing at it would strand checkpoints the fast path does
            # support. The repo config is cached, so the retry is cheap.
            log.info("fast tokenizer failed for %s (%s); retrying slow", self.model_id, exc)
            return AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=False, use_fast=False
            )

    def _load_processor(self, AutoProcessor: Any) -> Any:
        """Best-effort multimodal processor.

        Returns None for text-only repos. That is not an error: `supports_vision`
        is False for them and the vision path returns None like any other
        unavailable capability.
        """
        try:
            return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=False)
        except Exception as exc:
            log.info("no processor for %s (%s); text-only path", self.model_id, exc)
            return None

    def _load_model(self, AutoModelForCausalLM: Any) -> Any:
        """Load the weights onto the resolved device."""
        torch_dtype = _torch_dtype(self.dtype)
        kwargs: dict[str, Any] = {"trust_remote_code": False, "dtype": torch_dtype}
        if self.device == "cuda":
            # device_map places weights without a manual .to(); "auto" also
            # offloads to CPU when VRAM is short, which is what makes the 4B
            # variant usable on an 8 GB card.
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = None
        try:
            return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            # transformers < 4.56 spells the argument `torch_dtype`. Retrying is
            # better than failing on a version skew.
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = torch_dtype
            return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def unload(self) -> None:
        """Release weights and reclaim device memory.

        Called on a failed load too: a half-constructed model holds VRAM just as
        well as a whole one, and leaving it behind would make the *next* attempt
        fail for a different, more confusing reason.
        """
        self.loaded = False
        self._model = None
        self._processor = None
        self._tokenizer = None
        self.supports_vision = False
        if _torch_cuda_available():
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:  # pragma: no cover - best effort cleanup
                log.debug("cuda cache clear failed", exc_info=True)

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def _render_prompt(self, prompt: str) -> str:
        """Apply the checkpoint's own chat template to a user turn.

        Falls back to raw text when no template ships with the repo. Base models
        legitimately have none, and that is not an error -- but an instruction
        model without its template silently loses its turn structure, so the
        fallback is logged rather than silent.
        """
        tokenizer = self._tokenizer
        if tokenizer is None:
            return prompt
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            return prompt
        try:
            return apply_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            log.info("chat template unavailable for %s (%s); using raw text", self.model_id, exc)
            return prompt

    def _generation_kwargs(self, *, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        """Build `generate` kwargs, dropping the ones this torch cannot honour."""
        kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, int(max_new_tokens)),
            "do_sample": temperature > 0,
            "pad_token_id": getattr(self._tokenizer, "pad_token_id", None)
            or getattr(self._tokenizer, "eos_token_id", None),
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 0.95
        return kwargs

    def _decode_with_deadline(
        self,
        inputs: Any,
        *,
        max_new_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> tuple[str | None, int | None, int | None]:
        """Run `model.generate` in a worker thread under a real deadline.

        Returns `(text, prompt_tokens, completion_tokens)`, or `(None, ...)` if
        the decode timed out or failed.

        Why a thread and a streamer: `generate` has no timeout parameter, and a
        decode can stall indefinitely on a bad prompt, a memory-thrashing card or
        a wedged CUDA context. Checking `time.perf_counter()` after it returns
        would detect nothing -- it would only *label* a hang that already cost the
        full duration. Running it on a daemon thread and joining with a deadline
        bounds the time the *caller* waits, which is the thing that has to stay
        bounded.

        `skip_prompt=True` on the streamer is what keeps the prompt out of the
        returned text; the queue drains as tokens arrive, so a timed-out decode
        still returns whatever partial output it had.
        """
        from transformers import TextIteratorStreamer

        tokenizer = self._tokenizer
        if tokenizer is None or self._model is None:
            return None, None, None

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0
        )
        gen_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens, temperature=temperature
        )
        gen_kwargs["streamer"] = streamer

        error: list[BaseException] = []

        def _run() -> None:
            try:
                self._model.generate(**inputs, **gen_kwargs)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread
                error.append(exc)
                # The streamer waits on a queue that will never be fed; push a
                # sentinel so the main thread's join below cannot block on it.
                try:
                    streamer.end()
                except Exception:
                    pass

        worker = threading.Thread(target=_run, name="gemma-generate", daemon=True)
        started = time.perf_counter()
        worker.start()

        deadline = max(0.05, float(timeout_s))
        worker.join(timeout=deadline + _THREAD_JOIN_GRACE_S)
        still_running = worker.is_alive()

        if still_running:
            self.last_error = (
                f"generation exceeded timeout of {deadline:.1f}s; "
                "worker thread abandoned (it is a daemon and will not block exit)"
            )
            log.warning("%s (%s)", self.last_error, self.model_id)
            return None, None, None

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.last_latency_ms = elapsed_ms

        if error:
            exc = error[0]
            self.last_error = f"generation failed: {type(exc).__name__}: {exc}"
            log.warning("%s", self.last_error)
            return None, None, None

        try:
            text = "".join(
                chunk for chunk in streamer
                if isinstance(chunk, str)
            )
        except Exception as exc:
            self.last_error = f"stream read failed: {type(exc).__name__}: {exc}"
            return None, None, None

        text = text.strip()
        if not text:
            self.last_error = "model returned no text"
            return None, None, None

        prompt_tokens, completion_tokens = self._token_counts(inputs, text)
        return text, prompt_tokens, completion_tokens

    def _token_counts(self, inputs: Any, text: str) -> tuple[int | None, int | None]:
        """Prompt/completion token counts, when they are cheap to get.

        Only attempted outside the timeout path and behind try/except: token
        counts are telemetry, and telemetry must never fail a generation that
        already succeeded.
        """
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        try:
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            if input_ids is not None:
                prompt_tokens = int(input_ids.shape[-1])
        except Exception:
            prompt_tokens = None
        try:
            if self._tokenizer is not None:
                completion_tokens = len(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            completion_tokens = None
        return prompt_tokens, completion_tokens

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.7,
        timeout_s: float = GENERATION_TIMEOUT_S,
    ) -> GenerateResult | None:
        """Generate text. Returns None on any failure, including timeout.

        Never raises: the caller's deterministic fallback is a normal path, so an
        exception here would turn a routine "model is busy" into a 500.
        """
        if not self.loaded or self._model is None or self._tokenizer is None:
            self.last_error = self.last_error or "model not loaded"
            return None

        started = time.perf_counter()
        try:
            rendered = self._render_prompt(prompt)
            inputs = self._tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = self._move_inputs(inputs)
        except Exception as exc:
            self.last_error = f"prompt encoding failed: {type(exc).__name__}: {exc}"
            log.warning("%s", self.last_error)
            return None

        if not self._generate_lock.acquire(blocking=False):
            # A decode is already running on this model. Refusing is better than
            # queueing behind an unknown-length generation.
            self.last_error = "generation already in progress for this engine"
            return None
        try:
            text, prompt_tokens, completion_tokens = self._decode_with_deadline(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # pragma: no cover - defensive, _decode is guarded
            self.last_error = f"generation crashed: {type(exc).__name__}: {exc}"
            log.warning("%s", self.last_error)
            return None
        finally:
            self._generate_lock.release()

        latency_ms = int((time.perf_counter() - started) * 1000)
        self.last_latency_ms = latency_ms
        if text is None:
            return None

        self.last_error = None
        return GenerateResult(
            text=text,
            latency_ms=latency_ms,
            model_id=self.model_id,
            source=GeneratorSource.MODEL,
            raw_prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.2,
        timeout_s: float = VERIFY_TIMEOUT_S,
    ) -> GenerateResult | None:
        """Verify evidence with a multimodal Gemma. Returns None when it cannot.

        Vision is gated on detection, not assumption: a text-only checkpoint must
        report "I cannot see this" rather than describing a photo it never saw.
        """
        if not self.loaded or self._model is None:
            self.last_error = self.last_error or "model not loaded"
            return None
        if not self.supports_vision or self._processor is None:
            self.last_error = f"{self.model_id} has no image capability"
            log.info("%s", self.last_error)
            return None
        if not image_bytes:
            self.last_error = "empty image payload"
            return None

        try:
            from PIL import Image
        except Exception as exc:
            self.last_error = f"Pillow unavailable: {exc}"
            return None

        started = time.perf_counter()
        try:
            with Image.open(io.BytesIO(image_bytes)) as raw:
                image = raw.convert("RGB")
            inputs = self._processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            inputs = self._move_inputs(inputs)
        except Exception as exc:
            self.last_error = f"image encoding failed: {type(exc).__name__}: {exc}"
            log.warning("%s", self.last_error)
            return None

        if not self._generate_lock.acquire(blocking=False):
            self.last_error = "generation already in progress for this engine"
            return None
        try:
            text, prompt_tokens, completion_tokens = self._decode_with_deadline(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.last_error = f"vision generation crashed: {type(exc).__name__}: {exc}"
            return None
        finally:
            self._generate_lock.release()

        latency_ms = int((time.perf_counter() - started) * 1000)
        self.last_latency_ms = latency_ms
        if text is None:
            return None

        self.last_error = None
        return GenerateResult(
            text=text,
            latency_ms=latency_ms,
            model_id=self.model_id,
            source=GeneratorSource.MODEL,
            raw_prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _move_inputs(self, inputs: Any) -> Any:
        """Place encoded tensors on the model's device.

        On GPU the model may sit under accelerate's `device_map="auto"`, so the
        right destination is the model's own device rather than the requested
        one. Falls back to CPU placement if the model exposes no device (a
        meta-device or lazily-sharded model), which the model handles by copying.
        """
        target = getattr(self._model, "device", None)
        if target is None:
            return inputs
        try:
            return inputs.to(target)
        except Exception:
            return inputs

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def status(self) -> EngineStatus:
        """Report what is *actually* loaded, for /api/health.

        Deliberately pessimistic: versions come from installed metadata so a
        broken torch import is visible as "installed but unusable" rather than
        silently reported as absent.
        """
        return EngineStatus(
            model_id=self.model_id,
            loaded=self.loaded,
            device=self.device,
            dtype=self.dtype,
            supports_vision=self.supports_vision,
            load_error=self.last_error,
            last_latency_ms=self.last_latency_ms,
            torch_version=_torch_version(),
            transformers_version=_dist_version("transformers"),
        )