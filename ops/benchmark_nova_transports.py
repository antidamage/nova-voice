#!/usr/bin/env python3
"""Read-only REST/MCP latency probe used before pinning a provider route.

The script intentionally does not call mutating tools. It emits JSON so a
deployment record can retain medians/p95 without storing speech or prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx


async def measure(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> float:
    started = time.perf_counter()
    response = await client.request(method, path, **kwargs)
    response.raise_for_status()
    response.json()
    return (time.perf_counter() - started) * 1000


async def run(base_url: str, samples: int, token: str | None) -> None:
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=10) as client:
        timings: dict[str, list[float]] = {"rest_state": [], "mcp_health": []}
        mcp_error: str | None = None
        for _ in range(samples):
            timings["rest_state"].append(await measure(client, "GET", "/api/state"))
            if mcp_error:
                continue
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            try:
                timings["mcp_health"].append(
                    await measure(
                        client,
                        "POST",
                        "/api/mcp",
                        headers=headers,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {
                                "name": "nova.dashboard.health",
                                "arguments": {},
                            },
                        },
                    )
                )
            except httpx.HTTPStatusError as error:
                mcp_error = (
                    f"HTTP {error.response.status_code}; configure a Nova MCP token "
                    "for authenticated POST benchmarking"
                )
                # Continue measuring REST so an unauthenticated dashboard still
                # yields a useful hot-path result.
                continue
    result = {}
    for name, values in timings.items():
        if not values:
            continue
        ordered = sorted(values)
        result[name] = {
            "samples": len(values),
            "medianMs": round(statistics.median(values), 2),
            "p95Ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2),
        }
    if mcp_error:
        result["mcp_health"] = {"available": False, "error": mcp_error}
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    # Deployed dashboard hostname: see PRIVATEREF.md#1.4.
    parser.add_argument("--base-url", default="http://dashboard-host.local")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--mcp-token")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    asyncio.run(run(args.base_url, args.samples, args.mcp_token))


if __name__ == "__main__":
    main()
