"""Live vLLM candidate backend for MOSAIC's extract stage (MODEL_SELECTION_PLAN.md, stage S0).

Talks to an OpenAI-compatible vLLM server over HTTP. Nothing here calls a
paid API or opens a socket to anything but a loopback mock server started by
a test, unless `--live` is set (VS6). Eric declined a live-switch
acknowledgement token (2026-09-24); `--live` is the only switch.
"""
