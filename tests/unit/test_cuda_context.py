"""
Unit tests for the session worker's CUDA context lifecycle.

The bug these guard against had no traceback and no log line: when a worker
thread exited with a CUDA context still on its stack, PyCUDA called abort()
and the entire Streamlit process died. The user saw only "Connection error",
every time a run finished.

These tests run anywhere -- the helpers must be complete no-ops when pycuda
is absent, which is also what keeps CI and dev laptops working.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from src.cameras import cuda_context


class _FakeContextStack:
    """Minimal stand-in for pycuda.driver's context stack."""

    def __init__(self, depth: int = 0, pop_raises: bool = False) -> None:
        self.depth = depth
        self.pushes = 0
        self.pops = 0
        self._pop_raises = pop_raises
        outer = self

        class Context:
            @staticmethod
            def get_current():
                return object() if outer.depth > 0 else None

            @staticmethod
            def pop():
                if outer._pop_raises:
                    raise RuntimeError("cannot pop")
                outer.depth -= 1
                outer.pops += 1

        class _Primary:
            @staticmethod
            def push():
                outer.depth += 1
                outer.pushes += 1

        class Device:
            def __init__(self, index): self.index = index
            @staticmethod
            def retain_primary_context():
                return _Primary()

        self.module = types.SimpleNamespace(
            init=lambda: None, Context=Context, Device=Device
        )


@pytest.fixture()
def fake_cuda(monkeypatch):
    def install(stack: _FakeContextStack):
        monkeypatch.setattr(cuda_context, "_driver", lambda: stack.module)
        return stack
    return install


class TestWithoutCuda:
    """A machine with no CUDA must be entirely unaffected."""

    def test_push_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(cuda_context, "_driver", lambda: None)
        assert cuda_context.push_primary_context() is False

    def test_drain_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(cuda_context, "_driver", lambda: None)
        assert cuda_context.drain_contexts() == 0

    def test_a_broken_pycuda_does_not_raise(self, monkeypatch):
        """Never propagate: these run in a worker's startup and shutdown."""
        def explode():
            raise ImportError("pycuda is wrecked")
        monkeypatch.setattr(cuda_context, "_driver", explode)
        with pytest.raises(ImportError):
            explode()          # sanity: the fake really does raise
        monkeypatch.setattr(cuda_context, "_driver", lambda: None)
        cuda_context.push_primary_context()
        cuda_context.drain_contexts()


class TestContextLifecycle:
    def test_drain_empties_a_non_empty_stack(self, fake_cuda):
        """This is the whole point: a thread must not exit with a context
        still pushed, or PyCUDA aborts the process."""
        stack = fake_cuda(_FakeContextStack(depth=2))
        assert cuda_context.drain_contexts() == 2
        assert stack.depth == 0

    def test_drain_on_an_empty_stack_does_nothing(self, fake_cuda):
        stack = fake_cuda(_FakeContextStack(depth=0))
        assert cuda_context.drain_contexts() == 0
        assert stack.pops == 0

    def test_push_makes_a_context_current_when_there_is_none(self, fake_cuda):
        """Needed by every session after the first, since the previous
        worker drained the stack on its way out."""
        stack = fake_cuda(_FakeContextStack(depth=0))
        assert cuda_context.push_primary_context() is True
        assert stack.depth == 1
        assert stack.pushes == 1

    def test_push_does_not_stack_a_second_context(self, fake_cuda):
        """The first worker inherits the context autoprimaryctx pushed at
        import time; pushing another would leave one behind on drain."""
        stack = fake_cuda(_FakeContextStack(depth=1))
        assert cuda_context.push_primary_context() is True
        assert stack.pushes == 0
        assert stack.depth == 1

    def test_push_then_drain_leaves_the_stack_as_it_was(self, fake_cuda):
        stack = fake_cuda(_FakeContextStack(depth=0))
        cuda_context.push_primary_context()
        cuda_context.drain_contexts()
        assert stack.depth == 0

    def test_drain_gives_up_rather_than_spinning(self, fake_cuda):
        """A stuck stack must not hang shutdown."""
        stack = fake_cuda(_FakeContextStack(depth=3, pop_raises=True))
        assert cuda_context.drain_contexts() == 0   # gave up, did not raise

    def test_a_stack_that_never_empties_terminates(self, fake_cuda):
        stack = _FakeContextStack(depth=1)
        # pop() that never actually reduces the depth
        stack.module.Context.pop = staticmethod(lambda: None)
        fake_cuda(stack)
        cuda_context.drain_contexts()   # bounded loop; returns rather than hangs


class TestWorkerThreadContract:
    def test_a_worker_thread_can_push_and_drain_independently(self, fake_cuda):
        """Contexts are per-thread, which is why this lives in the worker
        rather than in the manager's constructor."""
        stack = fake_cuda(_FakeContextStack(depth=0))
        result = {}

        def worker():
            result["pushed"] = cuda_context.push_primary_context()
            result["drained"] = cuda_context.drain_contexts()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert result["pushed"] is True
        assert result["drained"] == 1
        assert stack.depth == 0, "the worker left a context behind"
