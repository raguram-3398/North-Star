"""Connectivity spike — Himalayas MCP + Tavily, real response shapes.

NOT production code, not a pytest test (excluded from testpaths by name —
run directly: `python tests/spike_grounding_connectivity.py`). Confirms,
before any cross-validation logic is designed:

  1. Himalayas MCP is reachable via ADK's McpToolset and returns something
     usable for a job-market query.
  2. Tavily is reachable with the existing .env key for the same query.
  3. How each source fails/times out/empties, independently, so the real
     cross-validation function (research_outline_agent.py) can be designed
     around actual behavior instead of assumed behavior.

No cross-validation, no output_guard, no Postgres writes here — just raw
shapes and failure modes, printed in full.
"""

import asyncio
import json

from dotenv import load_dotenv
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

load_dotenv()

import os  # noqa: E402  (after load_dotenv, so TAVILY_API_KEY is populated)

from tavily import TavilyClient  # noqa: E402

TIMEOUT_SECONDS = 10  # existing repo convention (CLAUDE.md guardrail #14)
SEED_ROLE = "Backend Engineer"  # from PRD §7.3's seed-list example roles
HIMALAYAS_MCP_URL = "https://mcp.himalayas.app/mcp"


async def spike_himalayas() -> None:
    print("\n=== Himalayas MCP: search_jobs ===")
    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=HIMALAYAS_MCP_URL,
            timeout=TIMEOUT_SECONDS,
        ),
    )
    try:
        tools = await asyncio.wait_for(toolset.get_tools(), timeout=TIMEOUT_SECONDS)
        print(f"toolset visible to ADK: {len(tools)} tools returned")
        by_name = {t.name: t for t in tools}
        if "search_jobs" not in by_name:
            print("FAILURE: 'search_jobs' tool not present in toolset")
            return

        result = await asyncio.wait_for(
            by_name["search_jobs"].run_async(
                args={"keyword": SEED_ROLE}, tool_context=None
            ),
            timeout=TIMEOUT_SECONDS,
        )
        print("raw result type:", type(result))
        print("top-level keys:", list(result.keys()))
        print("isError:", result.get("isError"))
        print("content item count:", len(result.get("content", [])))
        print("content item types:", [c.get("type") for c in result.get("content", [])])
        text = result["content"][0]["text"] if result.get("content") else ""
        print(f"content[0].text length: {len(text)} chars")
        print("--- first 1500 chars of content[0].text ---")
        print(text[:1500])
    except TimeoutError:
        print(f"FAILURE: Himalayas MCP call exceeded {TIMEOUT_SECONDS}s timeout")
    except Exception as exc:  # spike: report every failure mode, don't narrow
        print(f"FAILURE: Himalayas MCP call raised {type(exc).__name__}: {exc}")
    finally:
        await toolset.close()


def spike_tavily() -> None:
    print("\n=== Tavily: search ===")
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("FAILURE: TAVILY_API_KEY not set in environment/.env")
        return

    client = TavilyClient(api_key=api_key)
    query = f"{SEED_ROLE} job requirements and key skills"
    try:
        result = client.search(
            query=query,
            max_results=5,
            search_depth="basic",
            timeout=TIMEOUT_SECONDS,
        )
        print("raw result type:", type(result))
        print("top-level keys:", list(result.keys()))
        print(f"'results' count: {len(result.get('results', []))}")
        if result.get("results"):
            print("--- keys on first result item ---")
            print(list(result["results"][0].keys()))
            print("--- full first result item ---")
            print(json.dumps(result["results"][0], indent=2, default=str)[:1500])
    except Exception as exc:  # spike: report every failure mode, don't narrow
        print(f"FAILURE: Tavily call raised {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(spike_himalayas())
    spike_tavily()
