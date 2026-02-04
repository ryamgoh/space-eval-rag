from __future__ import annotations

import asyncio
from typing import Awaitable, Iterable, List


class ParallelExecutor:
    """Run coroutines with bounded concurrency."""
    def __init__(self, max_workers: int = 4):
        """Create a semaphore to limit in-flight tasks."""
        self._semaphore = asyncio.Semaphore(max_workers)

    async def run(self, coroutines: Iterable[Awaitable]) -> List:
        """Execute coroutines with concurrency control and return their results."""
        async def _guard(coro: Awaitable):
            async with self._semaphore:
                return await coro

        tasks = [asyncio.create_task(_guard(coro)) for coro in coroutines]
        return await asyncio.gather(*tasks)
