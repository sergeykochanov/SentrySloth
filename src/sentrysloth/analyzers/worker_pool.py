"""Bounded async worker pool for parallel LLM analysis."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def run_bounded_pool(
    items: list[T],
    worker: Callable[[int, T], Awaitable],
    max_in_flight: int,
) -> list:
    """Execute *worker(index, item)* for each item with bounded concurrency.

    Returns results sorted by original index.  Each worker must return a tuple
    whose first element is the index passed to it.
    """
    if not items:
        return []

    it = iter(enumerate(items))
    pending: set[asyncio.Task] = set()
    results: list = []

    for _ in range(min(max_in_flight, len(items))):
        idx, item = next(it)
        pending.add(asyncio.create_task(worker(idx, item)))

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            results.append(task.result())
            try:
                idx, item = next(it)
            except StopIteration:
                continue
            pending.add(asyncio.create_task(worker(idx, item)))

    results.sort(key=lambda r: r[0])
    return results
