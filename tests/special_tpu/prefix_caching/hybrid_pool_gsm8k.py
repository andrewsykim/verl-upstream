# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Log one cold batch of 8 GSM8K prompts x 4 copies, matching the GRPO probe."""

from datasets import load_dataset


def run(llm, engine_kwargs):
    from vllm import SamplingParams

    config = llm.llm_engine.vllm_config
    tokenizer = llm.get_tokenizer()
    block_size = config.cache_config.block_size
    enabled = config.cache_config.enable_prefix_caching
    prompts = []
    data = load_dataset("openai/gsm8k", "main", split="train[:8]")
    for row in data:
        # Same instruction and chat template as the historical GRPO dataset.
        prompt = row["question"] + ' Let\'s think step by step and output the final answer after "####".'
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append((prompt, tokens))
    if not any(len(t) > block_size and len(t) % block_size == 1 for _, t in prompts):
        raise RuntimeError("No prompt leaves one token after a cache block; use Qwen3-0.6B and block_size=64")

    params = SamplingParams(temperature=1e-6, top_p=1.0, top_k=-1, max_tokens=512)
    print(
        f"GSM8K model={engine_kwargs['model']} prefix_cache={enabled} "
        f"block_size={block_size} seed={engine_kwargs['seed']} "
        f"cacheable_prompts={sum(len(t) > block_size for _, t in prompts)} "
        f"temperature={params.temperature} requested_temperature=1e-6",
        flush=True,
    )
    for index, (prompt, tokens) in enumerate(prompts):
        print(f"PROMPT row={index} tokens={len(tokens)}\n{prompt}", flush=True)

    # Submit all copies together, before any warm-up can populate the cache.
    outputs = llm.generate(
        [{"prompt_token_ids": t} for _, t in prompts for _ in range(4)],
        params,
        use_tqdm=False,
    )
    total_hits = 0
    for position, output in enumerate(outputs):
        row, sample = divmod(position, 4)
        answer = output.outputs[0]
        hits = getattr(output, "num_cached_tokens", None)
        print(
            f"OUTPUT round=0 row={row} sample={sample} cached_tokens={hits} "
            f"finish={answer.finish_reason}\n{answer.text}",
            flush=True,
        )
        if hits is None or (not enabled and hits != 0):
            raise RuntimeError("Cache statistics unavailable or inconsistent with cache-off mode")
        total_hits += hits
    # A scheduler that prevents same-step cache reuse may report zero hits.
    print(f"GSM8K_DONE cached_tokens={total_hits}", flush=True)
    return 0
