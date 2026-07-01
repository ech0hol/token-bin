"""token-bin entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .engine import (
    AnthropicProvider,
    DeepSeekProvider,
    GenericOpenAIProvider,
    OpenAIProvider,
    TokenWaster,
    format_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="token-bin",
        description="🗑️  Precision token wasting — your token trash bin",
    )

    sub = parser.add_subparsers(dest="command")

    # ── tui (default) ────────────────────────────────────────────────────
    sub.add_parser("tui", help="Launch the interactive TUI (default)")

    # ── waste ────────────────────────────────────────────────────────────
    waste_p = sub.add_parser("waste", help="Waste tokens via CLI")
    waste_p.add_argument("--provider", "-p", choices=["openai", "anthropic", "deepseek", "generic"], default="openai")
    waste_p.add_argument("--model", "-m", required=True, help="Model name (e.g. gpt-4o)")
    waste_p.add_argument("--api-key", "-k", help="API key (or set env var)")
    waste_p.add_argument("--base-url", help="Base URL for generic/openai providers")
    waste_p.add_argument("--tokens", "-n", type=int, required=True, help="Target token count")
    waste_p.add_argument("--plain", action="store_true", help="Output plain report only")

    args = parser.parse_args()

    if args.command == "waste":
        asyncio.run(_cli_waste(args))
    else:
        # Lazy import: TUI requires textual, not needed for CLI
        from .ui import run_tui

        run_tui()


async def _cli_waste(args) -> None:
    """Run waste from CLI."""
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ API key required. Use --api-key or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var.")
        sys.exit(1)

    if args.provider == "openai":
        base = args.base_url or "https://api.openai.com/v1"
        provider = OpenAIProvider(api_key=api_key, model=args.model, base_url=base)
    elif args.provider == "anthropic":
        provider = AnthropicProvider(api_key=api_key, model=args.model)
    elif args.provider == "deepseek":
        base = args.base_url or "https://api.deepseek.com/v1"
        provider = DeepSeekProvider(api_key=api_key, model=args.model, base_url=base)
    else:
        base = args.base_url or "https://api.openai.com/v1"
        provider = GenericOpenAIProvider(api_key=api_key, model=args.model, base_url=base)

    waster = TokenWaster(provider)
    if not args.plain:
        print(f"🔬 Calibrating {args.model}...")
    await waster.calibrate()

    if not args.plain:
        print(f"🗑️  Wasting {args.tokens:,} tokens...")
    report = await waster.waste(args.tokens)

    print(format_report(report))


if __name__ == "__main__":
    main()
