"""Benchmarks tracking python-som against its own git history.

Run with ``uv run --extra bench asv run`` from ``asv_benchmarks/``. These measure this package
alone; the cross-library comparisons are hand-run scripts in ``benchmarks/`` and answer a different
question with a different tool.

Nothing here gates anything. asv numbers are only meaningful as a trend on one consistent machine,
so CI runs each benchmark once to prove it still executes and discards the timings.
"""
