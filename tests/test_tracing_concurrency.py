"""
Tests for the tracer's behaviour on the app's critical path: that concurrent work is attributed
to the right span, and that nothing tracing does can block the caller unboundedly.

Two properties are load-bearing for anyone tracing a request path:

  * the active-span stack is per-TASK, not per-thread. asyncio runs every task on one thread, so
    a thread-scoped stack makes concurrent handlers adopt each other as children - see
    `_active_spans` in tracer.py.
  * `flush(timeout=...)` returns within `timeout`. It is reachable from the request path
    (`trace(..., sync=True)` drains child spans through it, and the OpenAI Agents integration
    exposes it as force_flush/shutdown), so an unbounded wait there stalls a user's request for
    as long as the backend is unwell.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from agentx.tracing.ingest_client import IngestClient
from agentx.tracing.tracer import Tracer, _cached_signature


def make_tracer() -> Tracer:
    return Tracer(ingest_client=MagicMock())


def wires_by_name(tracer: Tracer) -> dict:
    return {call.args[0]["name"]: call.args[0] for call in tracer._client.enqueue.call_args_list}


# ----------------------------------------------------------------------------------
# asyncio: concurrent tasks must not adopt each other
# ----------------------------------------------------------------------------------

def test_concurrent_async_tasks_are_independent_roots():
    """Three handlers under gather() are three unrelated requests, not one nested tree."""
    tracer = make_tracer()

    async def handler(i: int) -> None:
        with tracer.trace(f"handler-{i}"):
            # Staggered so the tasks interleave and finish out of the order they started -
            # the arrangement that corrupts a shared stack.
            await asyncio.sleep(0.01 * (3 - i))

    async def main() -> None:
        await asyncio.gather(*(handler(i) for i in range(3)))

    asyncio.run(main())

    wires = wires_by_name(tracer)
    assert len(wires) == 3
    for i in range(3):
        assert wires[f"handler-{i}"].get("parent_span_id") is None
    # ...and each is its own conversation, not three views of one.
    assert len({w["session_id"] for w in wires.values()}) == 3


def test_async_task_created_inside_a_span_still_nests_under_it():
    """Isolation must not cost real nesting: a task spawned inside a span belongs to it."""
    tracer = make_tracer()

    async def child() -> None:
        with tracer.trace("child"):
            await asyncio.sleep(0)

    async def parent() -> None:
        with tracer.trace("parent"):
            await asyncio.sleep(0)          # suspend first, so nesting survives a real await
            await asyncio.create_task(child())

    asyncio.run(parent())

    wires = wires_by_name(tracer)
    assert wires["child"]["parent_span_id"] == wires["parent"]["span_id"]
    assert wires["child"]["session_id"] == wires["parent"]["session_id"]


def test_span_does_not_leak_out_of_a_completed_task():
    """A task's span must not still look active to whatever runs after it."""
    tracer = make_tracer()

    async def handler() -> None:
        with tracer.trace("handler"):
            await asyncio.sleep(0)

    async def main() -> None:
        await asyncio.gather(handler(), handler())
        assert tracer.current_span is None
        with tracer.trace("after"):
            pass

    asyncio.run(main())
    assert wires_by_name(tracer)["after"].get("parent_span_id") is None


# ----------------------------------------------------------------------------------
# Threads: behaviour must be unchanged from the threading.local() implementation
# ----------------------------------------------------------------------------------

def test_pool_worker_without_use_span_is_its_own_root():
    tracer = make_tracer()
    with tracer.trace("orchestrator"):
        def worker() -> None:
            with tracer.trace("worker"):
                pass
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(worker).result()
    assert wires_by_name(tracer)["worker"].get("parent_span_id") is None


def test_use_span_attaches_a_pool_worker_to_a_span_from_another_thread():
    tracer = make_tracer()
    with tracer.trace("orchestrator") as root:
        def worker() -> None:
            with tracer.use_span(root):
                with tracer.trace("worker"):
                    pass
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(worker).result()
    wires = wires_by_name(tracer)
    assert wires["worker"]["parent_span_id"] == wires["orchestrator"]["span_id"]


# ----------------------------------------------------------------------------------
# Stack hygiene
# ----------------------------------------------------------------------------------

def test_second_tracer_does_not_adopt_the_first_tracers_span():
    """The stack is process-wide; two AgentX clients must still keep separate trees."""
    outer, inner = make_tracer(), make_tracer()
    with outer.trace("outer-root"):
        assert inner.current_span is None
        with inner.trace("inner-root"):
            pass
    assert wires_by_name(inner)["inner-root"].get("parent_span_id") is None


def test_out_of_order_exit_leaves_the_remaining_chain_intact():
    """Closing an outer span first must not evict the inner one that is still open."""
    tracer = make_tracer()
    outer, inner = tracer.trace("outer"), tracer.trace("inner")
    outer.__enter__()
    inner.__enter__()
    outer.__exit__(None, None, None)
    assert tracer.current_span is inner
    with tracer.trace("next"):
        pass
    inner.__exit__(None, None, None)
    assert wires_by_name(tracer)["next"]["parent_span_id"] == inner.span_id


# ----------------------------------------------------------------------------------
# flush() must be bounded
# ----------------------------------------------------------------------------------

class _HangingIngest(IngestClient):
    """Real queue and drain thread, with a delivery that never finishes."""

    def __init__(self) -> None:
        self._workspace_id = None
        self._queue = queue.Queue(maxsize=500)
        self._worker = threading.Thread(target=self._drain, daemon=True)
        self._worker.start()

    def _send(self, payload: dict) -> None:
        time.sleep(30)


def test_flush_returns_within_its_timeout_when_delivery_is_stuck():
    client = _HangingIngest()
    client.enqueue({"name": "a"})
    client.enqueue({"name": "b"})

    started = time.monotonic()
    client.flush(timeout=0.5)
    elapsed = time.monotonic() - started

    # Generous upper bound - the point is that it returns at all rather than joining forever.
    assert elapsed < 5.0, f"flush(timeout=0.5) blocked for {elapsed:.1f}s"


def test_flush_returns_immediately_once_the_queue_is_drained():
    client = _HangingIngest()
    started = time.monotonic()
    client.flush(timeout=5.0)
    assert time.monotonic() - started < 1.0


# ----------------------------------------------------------------------------------
# Decorator input capture
# ----------------------------------------------------------------------------------

def test_signature_lookup_is_memoised_per_function():
    def fn(a, b=1):
        return a

    _cached_signature.cache_clear()
    first = _cached_signature(fn)
    second = _cached_signature(fn)
    assert first is second
    assert _cached_signature.cache_info().hits == 1


def test_decorated_function_still_captures_named_arguments():
    tracer = make_tracer()

    @tracer.trace("decorated")
    def run(query: str, top_k: int = 3) -> str:
        return "answer"

    assert run("hello") == "answer"
    wire = wires_by_name(tracer)["decorated"]
    assert wire["input"] == {"query": "hello", "top_k": 3}
    assert wire["output"] == "answer"


# ----------------------------------------------------------------------------------
# `async with` (documented in TRACING.md)
# ----------------------------------------------------------------------------------

def test_async_with_records_a_span():
    tracer = make_tracer()

    async def main() -> None:
        async with tracer.trace("async-agent") as span:
            span.input = "q"
            await asyncio.sleep(0)
            span.output = "a"

    asyncio.run(main())
    wire = wires_by_name(tracer)["async-agent"]
    assert wire["input"] == "q"
    assert wire["output"] == "a"
    assert wire.get("parent_span_id") is None


def test_async_with_nests_and_isolates_like_the_sync_form():
    tracer = make_tracer()

    async def handler(i: int) -> None:
        async with tracer.trace(f"outer-{i}"):
            async with tracer.trace(f"inner-{i}"):
                await asyncio.sleep(0.01 * (2 - i))

    async def main() -> None:
        await asyncio.gather(*(handler(i) for i in range(2)))

    asyncio.run(main())
    wires = wires_by_name(tracer)
    for i in range(2):
        assert wires[f"inner-{i}"]["parent_span_id"] == wires[f"outer-{i}"]["span_id"]
        assert wires[f"outer-{i}"].get("parent_span_id") is None


def test_async_with_propagates_exceptions_and_records_the_error():
    tracer = make_tracer()

    async def main() -> None:
        async with tracer.trace("boom"):
            raise ValueError("kaboom")

    try:
        asyncio.run(main())
    except ValueError as exc:
        assert str(exc) == "kaboom"
    else:
        raise AssertionError("exception was swallowed by __aexit__")

    assert wires_by_name(tracer)["boom"]["error"] == "kaboom"


def test_async_with_sync_true_does_not_block_the_event_loop():
    """sync=True waits on the network; that wait must not freeze other coroutines."""
    ticks = 0
    ticks_during_send = None

    def blocking_send(payload):
        nonlocal ticks_during_send
        before = ticks
        time.sleep(0.3)                      # stand-in for the real POST
        ticks_during_send = ticks - before
        return "trace-1"

    client = MagicMock()
    client.send_trace_sync.side_effect = blocking_send
    tracer = Tracer(ingest_client=client)

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(30):
            await asyncio.sleep(0.01)
            ticks += 1

    async def traced() -> None:
        async with tracer.trace("blocking", sync=True):
            await asyncio.sleep(0)

    async def main() -> None:
        await asyncio.gather(traced(), ticker())

    asyncio.run(main())

    assert client.send_trace_sync.called
    # The whole point: the loop kept scheduling other coroutines while the send was in flight.
    # Run it inline instead and this is 0 - the event loop is frozen for the full 0.3s.
    assert ticks_during_send and ticks_during_send >= 5, (
        f"only {ticks_during_send} tick(s) ran during a 0.3s synchronous send - "
        "the event loop was blocked"
    )
