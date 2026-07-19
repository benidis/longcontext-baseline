import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.inputs import TextPrompt, TokensPrompt
from transformers import AutoTokenizer
import logging

logger = logging.getLogger("eval")


async def process_requests(
    engine: AsyncLLMEngine,
    instances: List[Dict[str, Any]],
    output_path: str,
    concurrency: int,
    model_id: str,
    max_input_length: int,
    lora_request: Optional[LoRARequest] = None,
) -> None:
    """
    Process inference requests using vLLM AsyncLLMEngine.

    Args:
        engine: AsyncLLMEngine instance
        instances: List of instance dicts containing 'messages' and 'answer' fields
        output_path: Path to write JSONL results
        concurrency: Number of concurrent requests
        model_id: Model ID for loading the tokenizer and chat template
        lora_request: Optional LoRA request for adapter inference
    """

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=20,
        top_p=0.8,
        top_k=20,
    )

    # Load tokenizer for chat template
    logger.info(f"Loading tokenizer from {model_id}")
    for handler in logger.handlers:
        handler.flush()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    logger.info(f"Tokenizer loaded successfully")
    for handler in logger.handlers:
        handler.flush()

    logger.info(f"Starting inference on {len(instances)} instances with concurrency={concurrency}")
    for handler in logger.handlers:
        handler.flush()

    # Queue-based writer: saves each result immediately as it completes
    q: asyncio.Queue = asyncio.Queue()

    async def writer() -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            while True:
                item = await q.get()
                if item is None:
                    q.task_done()
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                q.task_done()

    writer_task = asyncio.create_task(writer())

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    completed_lock = asyncio.Lock()

    async def process_single(idx: int, instance: Dict[str, Any]) -> None:
        nonlocal completed
        async with semaphore:
            logger.info(f"Instance {idx}: started")
            for handler in logger.handlers:
                handler.flush()
            try:
                # Extract messages and expected answer
                messages = instance.get("messages", [])
                expected_answer = ""
                user_messages = []

                for msg in messages:
                    if msg.get("role") == "user":
                        user_messages.append(msg)
                    elif msg.get("role") == "assistant":
                        expected_answer = msg.get("content", "").strip()

                if not user_messages:
                    logger.warning(f"Instance {idx}: no user message found")
                    await q.put({
                        "instance_id": idx,
                        "output": "",
                        "error": "No user message in instance",
                        "expected": expected_answer,
                    })
                    return

                # Apply chat template
                prompt = tokenizer.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # For 128k runs, left-truncate prompts that exceed the context window
                if max_input_length > 64 * 1024:
                    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    max_prompt_tokens = max_input_length - sampling_params.max_tokens
                    if len(token_ids) > max_prompt_tokens:
                        logger.warning(
                            f"Instance {idx}: prompt too long ({len(token_ids)} tokens), "
                            f"left-truncating to {max_prompt_tokens} tokens"
                        )
                        token_ids = token_ids[-max_prompt_tokens:]
                    engine_input = TokensPrompt(prompt_token_ids=token_ids)
                    prompt_len_info = f"{len(token_ids)} tokens"
                else:
                    engine_input = TextPrompt(prompt=prompt)
                    prompt_len_info = f"{len(prompt)} chars"

                # Run inference
                request_id = f"req-{idx}"
                output = None
                logger.info(f"Instance {idx}: prompt length={prompt_len_info}, sending to engine")
                for handler in logger.handlers:
                    handler.flush()
                t0 = time.perf_counter()
                async for output in engine.generate(
                    engine_input,
                    sampling_params,
                    request_id=request_id,
                    lora_request=lora_request,
                ):
                    pass  # Get the final output
                latency = time.perf_counter() - t0

                generated_text = output.outputs[0].text.strip() if output else ""

                await q.put({
                    "instance_id": idx,
                    "output": generated_text,
                    "expected": expected_answer,
                    "latency_s": round(latency, 2),
                })

                async with completed_lock:
                    completed += 1
                    if completed % 5 == 0:
                        logger.info(f"Completed {completed}/{len(instances)} instances")
                        for handler in logger.handlers:
                            handler.flush()

                logger.info(f"Instance {idx}: done in {latency:.2f}s")
                for handler in logger.handlers:
                    handler.flush()

            except Exception as e:
                logger.error(f"Error processing instance {idx}: {e}")
                await q.put({
                    "instance_id": idx,
                    "output": "",
                    "error": str(e),
                    "expected": expected_answer,
                })

    # Process all requests concurrently
    tasks = [asyncio.create_task(process_single(idx, inst)) for idx, inst in enumerate(instances)]

    try:
        await asyncio.gather(*tasks)
    finally:
        await q.join()
        await q.put(None)
        await q.join()
        await writer_task

    logger.info(f"Done. Wrote {len(instances)} results to {output_path}")
    for handler in logger.handlers:
        handler.flush()
