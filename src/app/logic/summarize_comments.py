import asyncio
import json
import re
import ast

import httpx
import json5
import demjson3 as demjson

from asyncio import Semaphore
from collections import defaultdict
from pydantic import BaseModel
from typing import List, Dict, Any


# Defines the output schema expected by the API
class CommentsOut(BaseModel):
    summary: List[Dict[str, Any]]  # ou adapte selon le contenu exact


# Mapping of category labels to descriptions
category_names = {
    0: "Bugs / technical issues",
    1: "Requested features",
    2: "Design & UX",
    3: "Performance & speed",
    4: "Login / account",
    5: "Other"
}


def build_summary_prompt(description: str, comments: list[str], nb_points: int) -> str:
    """
    Construct the prompt for summarizing a category.
    """
    comments_text = "\n".join(f"- {c}" for c in comments)
    return (
        f"System: You are given a list of user comments about “{description}”.\n"
        f"Identify up to {nb_points} distinct, concrete, and actionable improvement or correction points "
        "that are both important and frequently mentioned.\n"
        "– If fewer than that number of truly relevant points exist, list only those.\n"
        "– Do not invent or generalize: each point must clearly reference the underlying user feedback.\n"
        "– Use precise language in english so that a developer reading these points understands exactly what to fix.\n"
        "- Provide an objective description of the problem as if you were explaining it, do not phrase it as a task to"
        " be accomplished.\n"
        "– Order the points from most important to least important.\n"
        "– At the end of each point, append a frequency score in parentheses indicating how often this issue appears "
        "(e.g., “(12 mentions)”).\n\n"
        "Return ONLY a valid JSON array of strings (double-quoted), without any additional text.\n\n"
        "Comments:\n"
        f"{comments_text}\n\n"
        f"System: You are given a list of user comments about “{description}”.\n"
        f"Identify up to {nb_points} distinct, concrete, and actionable improvement or correction points "
        "that are both important and frequently mentioned.\n"
        "– If fewer than that number of truly relevant points exist, list only those.\n"
        "– Do not invent or generalize: each point must clearly reference the underlying user feedback.\n"
        "– Use precise language in english so that a developer reading these points understands exactly what to fix.\n"
        "- Provide an objective description of the problem as if you were explaining it, do not phrase it as a task to "
        "be accomplished.\n"
        "– Order the points from most important to least important.\n"
        "– At the end of each point, append a frequency score in parentheses indicating how often this issue appears "
        "(e.g., “(12 mentions)”).\n\n"
        "Return ONLY a valid JSON array of strings (double-quoted), without any additional text.\n\n"
        "Answer with a JSON array of strings."
    )


def clean_and_parse(raw: str):
    """
    Attempt to parse a raw response string into a Python list.
    Supports strict JSON, JSON5, demjson3, and Python repr fallback.
    """
    # Remove Markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    # Try strict JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try JSON5
    try:
        return json5.loads(raw)
    except Exception:
        pass

    # Try demjson3
    try:
        return demjson.decode(raw)
    except Exception:
        pass

    # Fallback to Python literal eval
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass

    # If all parsing fails, return the raw text in a single-item list
    return [raw]


async def summarize_category(label: int, comments: list[str], nb_points: int):
    """
    Send a summarization request for a given category and
    return a dict with label, description, and extracted points.
    """
    # Limit concurrent API requests
    token_semaphore = Semaphore(2)

    description = category_names[label]
    prompt = build_summary_prompt(description, comments, nb_points)

    async with token_semaphore:
        async with httpx.AsyncClient(timeout=2000.0) as client:
            response = await client.post(
                "https://ollama.kube.isc.heia-fr.ch/api/generate",
                json={
                    "model": "qwen2.5:14b-instruct",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.0,
                }
            )
        response.raise_for_status()

    raw = response.json().get("response", "").strip()
    summary = clean_and_parse(raw)

    return {
        "category": label,
        "description": description,
        "top_points": summary
    }


async def summarize_comments(raw):
    comments = raw['comments']

    # Group comments by category
    comments_by_label = defaultdict(list)
    for c in comments:
        lbl = int(c["category"])
        comments_by_label[lbl].append(c["content"])

    # Compute tasks and number of points per category
    total_comments = sum(len(lst) for lst in comments_by_label.values())
    tasks = []
    for label, lst in comments_by_label.items():
        nb_points = max(1, round((len(lst) / total_comments) * 42) + 3)
        tasks.append(summarize_category(label, lst, nb_points))

    # Run all summarization tasks in parallel
    results = await asyncio.gather(*tasks)

    # Remove internal double-quotes from points
    for cat in results:
        cat["top_points"] = [p.replace('"', '') for p in cat["top_points"]]
    
    print(results)

    return CommentsOut(summary=results)


if __name__ == "__main__":
    asyncio.run(summarize_comments())
