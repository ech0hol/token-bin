"""token-bin TUI — guided token wasting with a beautiful terminal interface."""

from __future__ import annotations

import asyncio
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Center, Container
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    Static,
)

from .engine import (
    AnthropicProvider,
    DeepSeekProvider,
    GenericOpenAIProvider,
    OpenAIProvider,
    TokenWaster,
    WasteReport,
    format_report,
)

# ── Constants ────────────────────────────────────────────────────────────────

CSS = """
Screen {
    background: #0a0a0a;
    layout: vertical;
}

/* Global: all screens fill the available area */
Container > Screen {
    height: 100%;
}

/* ── Shared: centered full-area container ─────── */

.centered {
    align: center middle;
    height: 100%;
    width: 100%;
}

/* ── Welcome ─────────────────────────────────── */

#logo {
    color: #00ff88;
    text-style: bold;
    margin-bottom: 1;
}

#tagline {
    color: #888888;
    margin-bottom: 2;
}

#start-btn {
    min-width: 24;
    margin-top: 2;
}

/* ── Setup form ──────────────────────────────── */

#setup-form {
    width: 64;
    border: solid #00ff88;
    padding: 1 2;
}

#setup-title {
    color: #00ff88;
    text-style: bold;
    margin-bottom: 1;
}

.setup-label {
    color: #aaaaaa;
    margin: 1 0 0 0;
}

.setup-input {
    width: 100%;
    margin-bottom: 1;
}

#provider-radios {
    margin: 0 0 1 0;
}

RadioButton {
    margin: 0 1;
}

#advanced-section {
    display: none;
}

#advanced-section.visible {
    display: block;
}

#error-msg {
    color: #ff4444;
    margin-top: 1;
}

#setup-buttons {
    width: 100%;
    margin-top: 1;
}

#setup-buttons Button {
    margin: 0 1;
}

/* ── Waste progress ──────────────────────────── */

#waste-box {
    width: 56;
    border: solid #00ff88;
    padding: 1 2;
}

#waste-title {
    color: #00ff88;
    text-style: bold;
    margin-bottom: 1;
}

#waste-status {
    color: #cccccc;
    margin: 1 0;
}

#waste-bar {
    width: 100%;
    margin: 1 0;
}

#waste-bar > Bar {
    color: #00ff88;
}

#waste-stats {
    color: #888888;
    margin-top: 1;
}

/* ── Report ─────────────────────────────────── */

#report-box {
    width: 70;
    height: auto;
    max-height: 18;          /* limit height so it fits screen */
    border: double #00ff88;
    padding: 0 1;
    overflow-y: auto;
}

#report-text {
    color: #cccccc;
}

#report-footer-text {
    color: #888888;
    margin-top: 1;
}

#done-btn {
    min-width: 20;
    margin-top: 1;
}
"""

LOGO = r"""
╔══════════════════════════════════╗
║                                  ║
║    ████████╗ ██████╗ ██╗  ██╗   ║
║    ╚══██╔══╝██╔═══██╗██║ ██╔╝   ║
║       ██║   ██║   ██║█████╔╝    ║
║       ██║   ██║   ██║██╔═██╗    ║
║       ██║   ╚██████╔╝██║  ██╗   ║
║       ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ║
║                                  ║
║     ██████╗ ██╗███╗   ██╗        ║
║     ██╔══██╗██║████╗  ██║        ║
║     ██████╔╝██║██╔██╗ ██║        ║
║     ██╔══██╗██║██║╚██╗██║        ║
║     ██████╔╝██║██║ ╚████║        ║
║     ╚═════╝ ╚═╝╚═╝  ╚═══╝        ║
║                                  ║
╚══════════════════════════════════╝
"""

TAGLINE = "Your precision token wastebin"


# ── Welcome Screen ───────────────────────────────────────────────────────────


class WelcomeScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Container(
            Static(LOGO, id="logo"),
            Static(TAGLINE),
            Button(">> START WASTING", variant="success", id="start-btn"),
            classes="centered",
        )

    @on(Button.Pressed, "#start-btn")
    def go_setup(self) -> None:
        self.app.push_screen(SetupScreen())  # type: ignore[arg-type]


# ── Setup Screen ─────────────────────────────────────────────────────────────


class SetupScreen(Screen):
    _provider: str = "openai"

    def compose(self) -> ComposeResult:
        yield Container(
            Container(
                Static("<< CONFIGURE YOUR WASTE >>", id="setup-title"),
                Label("Provider", classes="setup-label"),
                RadioSet(
                    RadioButton("OpenAI / Azure", value="openai", id="rb-openai"),
                    RadioButton("Anthropic Claude", value="anthropic", id="rb-anthropic"),
                    RadioButton("DeepSeek", value="deepseek", id="rb-deepseek"),
                    RadioButton("Generic (OpenRouter / Custom)", value="generic", id="rb-generic"),
                    id="provider-radios",
                ),
                Label("API Key", classes="setup-label"),
                Input(placeholder="sk-... or your API key", password=True, id="api-key", classes="setup-input"),
                Label("Model Name", classes="setup-label"),
                Input(placeholder="gpt-4o / claude-3-5-sonnet / deepseek-chat / ...", id="model-name", classes="setup-input"),
                Container(
                    Label("Base URL", classes="setup-label"),
                    Input(placeholder="https://api.openai.com/v1", id="base-url", classes="setup-input"),
                    id="advanced-section",
                ),
                Label("Target Tokens", classes="setup-label"),
                Input(placeholder="10000", id="target-tokens", classes="setup-input"),
                Static("", id="error-msg"),
                Container(
                    Button("Back", variant="default", id="back-btn"),
                    Button("WASTE!", variant="success", id="waste-btn"),
                    id="setup-buttons",
                ),
                id="setup-form",
            ),
            classes="centered",
        )

    def on_mount(self) -> None:
        import os

        if key := os.environ.get("OPENAI_API_KEY"):
            self.query_one("#api-key", Input).value = key
        if model := os.environ.get("TOKEN_BIN_MODEL"):
            self.query_one("#model-name", Input).value = model

    @on(RadioSet.Changed, "#provider-radios")
    def provider_changed(self, event: RadioSet.Changed) -> None:
        _PROVIDER_BY_BUTTON = {
            "rb-openai": "openai",
            "rb-anthropic": "anthropic",
            "rb-deepseek": "deepseek",
            "rb-generic": "generic",
        }
        self._provider = _PROVIDER_BY_BUTTON.get(event.pressed.id, "openai")

        advanced = self.query_one("#advanced-section", Container)
        advanced.set_class(self._provider == "generic", "visible")

        model_input = self.query_one("#model-name", Input)
        base_url_input = self.query_one("#base-url", Input)
        _PLACEHOLDERS = {
            "openai": ("gpt-4o", "https://api.openai.com/v1"),
            "anthropic": ("claude-3-5-sonnet-20241022", ""),
            "deepseek": ("deepseek-chat", "https://api.deepseek.com/v1"),
            "generic": ("gpt-4o", "https://api.openai.com/v1"),
        }
        model_ph, url_ph = _PLACEHOLDERS[self._provider]
        model_input.placeholder = model_ph
        base_url_input.placeholder = url_ph

    @on(Button.Pressed, "#back-btn")
    def go_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#waste-btn")
    def start_waste(self) -> None:
        api_key = self.query_one("#api-key", Input).value.strip()
        model = self.query_one("#model-name", Input).value.strip()
        target_str = self.query_one("#target-tokens", Input).value.strip()
        base_url = self.query_one("#base-url", Input).value.strip()

        errors = []
        if not api_key:
            errors.append("API Key is required")
        if not model:
            errors.append("Model name is required")
        if not target_str:
            errors.append("Target tokens is required")
        elif not target_str.isdigit() or int(target_str) <= 0:
            errors.append("Target tokens must be a positive integer")
        if self._provider == "generic" and not base_url:
            base_url = "https://api.openai.com/v1"

        err_widget = self.query_one("#error-msg", Static)
        if errors:
            err_widget.update("\n".join(f"X {e}" for e in errors))
            return
        err_widget.update("")

        target = int(target_str)

        if self._provider == "openai":
            provider = OpenAIProvider(api_key=api_key, model=model, base_url=base_url or "https://api.openai.com/v1")
        elif self._provider == "anthropic":
            provider = AnthropicProvider(api_key=api_key, model=model)
        elif self._provider == "deepseek":
            provider = DeepSeekProvider(api_key=api_key, model=model, base_url=base_url or "https://api.deepseek.com/v1")
        else:
            provider = GenericOpenAIProvider(api_key=api_key, model=model, base_url=base_url)

        self.app.push_screen(WasteScreen(provider, target))  # type: ignore[arg-type]


# ── Waste (Progress) Screen ──────────────────────────────────────────────────


class WasteScreen(Screen):
    def __init__(self, provider, target: int):
        super().__init__()
        self.provider = provider
        self.target = target
        self.waster = TokenWaster(provider)
        self.report: Optional[WasteReport] = None

    def compose(self) -> ComposeResult:
        yield Container(
            Container(
                Static("* WASTING TOKENS...", id="waste-title"),
                Static("Calibrating...", id="waste-status"),
                ProgressBar(total=100, show_eta=False, id="waste-bar"),
                Static("", id="waste-stats"),
                id="waste-box",
            ),
            classes="centered",
        )

    def on_mount(self) -> None:
        self._run_waste()

    @work(exclusive=True)
    async def _run_waste(self) -> None:
        status = self.query_one("#waste-status", Static)
        bar = self.query_one("#waste-bar", ProgressBar)
        stats = self.query_one("#waste-stats", Static)

        def update_progress(round_num: int, target: int, consumed: int) -> None:
            pct = min(100, int(consumed / target * 100)) if target else 100
            bar.update(progress=pct)
            status.update(f"Round {round_num}: consumed {consumed:,} / {target:,} tokens")
            stats.update(f"Error so far: {consumed - target:+d} tokens")

        try:
            status.update("[*] Calibrating...")
            bar.update(progress=5)
            stats.update("Sending sample request...")
            await self.waster.calibrate()

            status.update("[*] Wasting tokens...")
            self.report = await self.waster.waste(self.target, progress_callback=update_progress)

            bar.update(progress=100)
            status.update("[+] Done!")
            await asyncio.sleep(1)
            self.app.push_screen(ReportScreen(self.report))
        except Exception as exc:
            status.update(f"[!] Error: {exc}")
            stats.update("Check API key, model name, network.")


# ── Report Screen ────────────────────────────────────────────────────────────


class ReportScreen(Screen):
    def __init__(self, report: WasteReport):
        super().__init__()
        self.report = report

    def compose(self) -> ComposeResult:
        report_text = format_report(self.report)

        yield Container(
            Container(
                Static(report_text, id="report-text"),
                id="report-box",
            ),
            Container(
                Static("Waste complete. Mother Earth disapproves.", id="report-footer-text"),
                Button("> WASTE MORE", variant="success", id="done-btn"),
            ),
            classes="centered",
        )

    @on(Button.Pressed, "#done-btn")
    def restart(self) -> None:
        self.app.pop_screen()
        self.app.pop_screen()


# ── Main App ─────────────────────────────────────────────────────────────────


class TokenBinApp(App[None]):
    CSS = CSS
    TITLE = "Token-Bin"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(id="main-area")
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())


def run_tui() -> None:
    app = TokenBinApp()
    app.run()


if __name__ == "__main__":
    run_tui()
