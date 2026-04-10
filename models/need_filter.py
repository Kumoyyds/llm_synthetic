import pandas as pd
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI  # Requires langchain-openai package
from dotenv import load_dotenv
load_dotenv()
lite_llm_key_all = os.getenv('LITE_LLM_KEY_ALL')

llm_model = ChatOpenAI(base_url='https://ipsos.litellm-prod.ai/',model='gpt-5', temperature=0.1, api_key=lite_llm_key_all)

from typing import Literal, Union, Dict, List


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text(value: object, *, max_chars: int | None = None) -> str:
    """Return a JSON-safe UTF-8 string.

    - Coerces non-strings to str
    - Normalizes newlines
    - Removes ASCII control characters (except \t/\n)
    - Replaces invalid unicode sequences
    - Optionally truncates to max_chars
    """
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub(" ", text)
    # Ensure the string is valid UTF-8 (replace unpaired surrogates etc.)
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]

    return text


def _normalize_system_info(system_info: object) -> dict:
    """Keep only the fields used in prompts and sanitize their content."""
    if not isinstance(system_info, dict):
        return {"insight": _sanitize_text(system_info)}

    out: dict = {}

    insight = system_info.get("insight")
    if insight is not None:
        out["insight"] = _sanitize_text(insight, max_chars=6000)

    for q in ("qneed2", "qneed3"):
        qval = system_info.get(q)
        if isinstance(qval, dict):
            cate = _sanitize_text(qval.get("cate", ""), max_chars=300)
            comment = _sanitize_text(qval.get("comment", ""), max_chars=2000)
            if cate or comment:
                out[q] = {"cate": cate, "comment": comment}

    return out

KPI_TYPE = Literal["relevance", "differentiation", "believability"]

KPI_QUESTIONS = {
    "relevance": {
        "question": "Is this product concept relevant to your needs and interests?",
        "criteria": """
        - Does this concept address needs, problems, or desires mentioned in the consumer insights?
        - Does it fit their lifestyle, preferences, or values?
        - Would they find this product useful or appealing based on their stated needs?
        """
    },
    "differentiation": {
        "question": "Does this product concept stand out as unique or different from what's currently available?",
        "criteria": """
        - Does this concept offer something new or innovative that would catch attention?
        - Is it meaningfully different from existing products the consumer might know?
        - Would this concept make the consumer want to try it over alternatives?
        """
    },
    "believability": {
        "question": "Do you find this product concept believable and credible?",
        "criteria": """
        - Are the claims and benefits stated in the concept realistic and trustworthy?
        - Does the concept align with what the consumer knows or expects from this category?
        - Would the consumer believe the product can deliver on its promises?
        """
    }
}

def ai_filter(
    new_concept: str, 
    kpi_type: KPI_TYPE, 
    system_info: dict, 
    llm = llm_model,
    return_reasoning: bool = False
) -> Union[str, dict]:
    """
    Evaluate a product concept from a consumer's perspective.
    
    Args:
        new_concept: The product concept text to evaluate
        kpi_type: One of "relevance", "differentiation", or "believability"
        system_info: Dict containing consumer insights (qneed2, qneed3, insight)
        llm: LLM model instance
        return_reasoning: If True, return dict with 'answer' and 'reasoning'
        
    Returns:
        If return_reasoning=False: "yes" or "no"
        If return_reasoning=True: {"answer": "yes/no", "reasoning": "..."}
    """
    messages = _build_messages(new_concept, kpi_type, _normalize_system_info(system_info), return_reasoning)
    ai_response = llm.invoke(messages)
    return _parse_response(ai_response.content, return_reasoning)


def _build_messages(new_concept: str, kpi_type: KPI_TYPE, system_info: dict, return_reasoning: bool) -> list:
    """Helper to build messages for LLM call (shared by sync and async versions)."""
    kpi_config = KPI_QUESTIONS[kpi_type]

    system_info = _normalize_system_info(system_info)
    new_concept = _sanitize_text(new_concept, max_chars=8000)
    
    qneeds = []
    for q in ['qneed2', 'qneed3']:
        if q in system_info and isinstance(system_info[q], dict):
            qneeds.append(system_info[q])
    qneeds = [f"{_sanitize_text(q.get('cate', ''), max_chars=300)}: {_sanitize_text(q.get('comment', ''), max_chars=2000)}" for q in qneeds]
    qneeds_text = "\n".join(qneeds) if qneeds else "No specific category needs provided."
    insight_text = _sanitize_text(system_info.get('insight', 'No general insights provided.'), max_chars=6000)
    
    system_prompt = f"""You are a digital twin mimicking a specific consumer. Your role is to respond as this consumer would, based on their interview insights.

Consumer Profile:
=================
Category-Specific Needs:
{qneeds_text}

General Insights & Preferences:
{insight_text}
=================

Respond authentically as this consumer would, based on their actual needs, values, and preferences."""

    if return_reasoning:
        user_query = f"""Evaluate this product concept:
---
{new_concept}
---

Question: {kpi_config['question']}

Evaluation Criteria:
{kpi_config['criteria']}

Think step by step:
1. Review the consumer's specific needs and general insights
2. Assess the concept against each evaluation criterion
3. Determine if this consumer would answer yes or no

Output format (use exactly this format):
REASONING: <your step-by-step reasoning here>
ANSWER: <yes or no>"""
    else:
        user_query = f"""Evaluate this product concept:
---
{new_concept}
---

Question: {kpi_config['question']}

Evaluation Criteria:
{kpi_config['criteria']}

Think step by step:
1. Review the consumer's specific needs and general insights
2. Assess the concept against each evaluation criterion
3. Determine if this consumer would answer yes or no

Output ONLY "yes" or "no" (lowercase, no punctuation or explanation)."""
    
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]


def _parse_response(response_text: str, return_reasoning: bool) -> Union[str, dict]:
    """Helper to parse LLM response (shared by sync and async versions)."""
    response_text = response_text.strip()
    
    if return_reasoning:
        reasoning = ""
        answer = ""
        if "REASONING:" in response_text and "ANSWER:" in response_text:
            parts = response_text.split("ANSWER:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip().lower()
        else:
            answer = "yes" if "yes" in response_text.lower() else "no"
            reasoning = response_text
        return {"answer": answer, "reasoning": reasoning}
    else:
        return response_text.lower()


async def ai_filter_async(
    new_concept: str, 
    kpi_type: KPI_TYPE, 
    system_info: dict, 
    llm = llm_model,
    return_reasoning: bool = False
) -> Union[str, dict]:
    """
    Async version of ai_filter for concurrent processing.
    
    Args:
        new_concept: The product concept text to evaluate
        kpi_type: One of "relevance", "differentiation", or "believability"
        system_info: Dict containing consumer insights (qneed2, qneed3, insight)
        llm: LLM model instance
        return_reasoning: If True, return dict with 'answer' and 'reasoning'
        
    Returns:
        If return_reasoning=False: "yes" or "no"
        If return_reasoning=True: {"answer": "yes/no", "reasoning": "..."}
    """
    messages = _build_messages(new_concept, kpi_type, _normalize_system_info(system_info), return_reasoning)
    ai_response = await llm.ainvoke(messages)
    return _parse_response(ai_response.content, return_reasoning)


async def ai_filter_batch_async(
    items: List[Dict],
    llm = llm_model,
    max_concurrency: int = 10,
    retry_attempts: int = 3,
    show_progress: bool = False,
    progress_desc: str = "AI filter"
) -> List[Union[str, dict]]:
    """
    Process multiple filter requests concurrently using async.
    
    Args:
        items: List of dicts, each containing:
            - new_concept: str
            - kpi_type: KPI_TYPE
            - system_info: dict
            - return_reasoning: bool (optional, default False)
        llm: LLM model instance
        max_concurrency: Maximum concurrent requests (default 10)
        
    Returns:
        List of results in same order as input items
        
    Example:
        items = [
            {"new_concept": "concept1", "kpi_type": "relevance", "system_info": info1},
            {"new_concept": "concept2", "kpi_type": "differentiation", "system_info": info2},
        ]
        results = asyncio.run(ai_filter_batch_async(items))
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    def _error_result(return_reasoning: bool, err: Exception) -> Union[str, dict]:
        msg = f"__error__: {type(err).__name__}: {err}"
        if return_reasoning:
            return {"answer": "__error__", "reasoning": msg}
        return "__error__"

    async def process_one(idx: int, item: dict) -> tuple[int, Union[str, dict]]:
        async with semaphore:
            last_err: Exception | None = None
            attempts = max(1, int(retry_attempts))
            for attempt in range(attempts):
                try:
                    result = await ai_filter_async(
                        new_concept=item["new_concept"],
                        kpi_type=item["kpi_type"],
                        system_info=item["system_info"],
                        llm=llm,
                        return_reasoning=item.get("return_reasoning", False),
                    )
                    return idx, result
                except Exception as e:
                    last_err = e
                    if attempt < attempts - 1:
                        # Lightweight backoff to reduce bursty failures
                        await asyncio.sleep(0.25 * (2 ** attempt))
                        continue
                    return idx, _error_result(item.get("return_reasoning", False), e)

            # Unreachable, but keeps type-checkers happy
            return idx, _error_result(item.get("return_reasoning", False), last_err or Exception("Unknown error"))

    tasks = [asyncio.create_task(process_one(i, item)) for i, item in enumerate(items)]
    results: List[Union[str, dict]] = [None] * len(items)  # type: ignore[assignment]

    if not show_progress:
        pairs = await asyncio.gather(*tasks)
        for idx, result in pairs:
            results[idx] = result
        return results

    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        # tqdm not available; fall back to no progress.
        pairs = await asyncio.gather(*tasks)
        for idx, result in pairs:
            results[idx] = result
        return results

    with tqdm(total=len(tasks), desc=progress_desc) as pbar:
        for fut in asyncio.as_completed(tasks):
            idx, result = await fut
            results[idx] = result
            pbar.update(1)
    return results


def ai_filter_batch(
    items: List[Dict],
    llm = llm_model,
    max_concurrency: int = 10,
    show_progress: bool = False,
    progress_desc: str = "AI filter"
) -> List[Union[str, dict]]:
    """
    Synchronous wrapper for batch processing (runs async internally).
    
    Args:
        items: List of dicts, each containing:
            - new_concept: str
            - kpi_type: KPI_TYPE
            - system_info: dict
            - return_reasoning: bool (optional, default False)
        llm: LLM model instance
        max_concurrency: Maximum concurrent requests (default 10)
        
    Returns:
        List of results in same order as input items
        
    Example:
        items = [
            {"new_concept": "concept1", "kpi_type": "relevance", "system_info": info1},
            {"new_concept": "concept2", "kpi_type": "differentiation", "system_info": info2},
        ]
        results = ai_filter_batch(items)
    """
    # In notebooks (IPython/Jupyter), an event loop is already running.
    # asyncio.run() cannot be used there; callers should use:
    #   batch_results = await ai_filter_batch_async(...)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            ai_filter_batch_async(
                items,
                llm=llm,
                max_concurrency=max_concurrency,
                show_progress=show_progress,
                progress_desc=progress_desc,
            )
        )
    raise RuntimeError(
        "ai_filter_batch() cannot be called from a running event loop (e.g., Jupyter). "
        "Use: `batch_results = await ai_filter_batch_async(items, llm=..., max_concurrency=...)`."
    )
