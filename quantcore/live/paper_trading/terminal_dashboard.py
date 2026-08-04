"""Live Terminal Risk Dashboard for Delta Neutral VRP.

Renders a real-time portfolio health monitor in the terminal using ANSI escape
codes (pure stdlib, no external TUI dependency required).  An optional Textual
mode provides a fully interactive TUI with blinking cells and keyboard shortcuts.

Philosophy: the dashboard is a *read-only consumer* of shared state.  It never
modifies execution_state, risk_controller, or portfolio objects.  Data flows
from the runner -> shared state objects -> dashboard via Copy-on-Write snapshots.

Thread Safety
------------
All shared data is read through ``threading.RLock``-protected snapshots created
by the data producer (runner).  Dashboard reads never hold a lock — they
operate on a pre-copied snapshot.  This eliminates blocking on every display
refresh tick.

Layout (80-column terminal)
----------------------------

  ┌══ CAPITAL HEALTH ════════════════════════════════════════════╗
  ║  Starting  $100,000    Peak Equity $100,432    Drawdown -0.0% ║
  ║  Current   $100,203    Margin Used  23.4%     Day P&L +$203  ║
  ├── POSITIONS ──────────────────────────────────────────────────┤
  ║  Open: 4  │  Net Δ  -2.3   │  Largest Δ  AAPL   +3.1         ║
  ├── GREEKS ────────────────────────────────────────────────────┤
  ║  Δ  -2.3  │  Γ  +0.04  │  ν  +12.3  │  Θ  -$4.21/day         ║
  ├── SAFETY ────────────────────────────────────────────────────┤
  ║  MaxDD Kill  ●GREEN●   │  Sharpe    ●GREEN●  │  Macro  ●GREEN● ║
  ║  Feed Health ●GREEN●    │  PID  12345  │  Run 04:23:31       ║
  └── TCA ────────────────────────────────────────────────────────┤
  ║  Avg Slip  4.2bps  │  WC  32.1bps │ Sweeps  2  │  rejections 0 ║
  └──────────────────────────────────────────────────────────────┘

ANSI Color Codes
----------------
  GREEN  = safe / no action needed
  YELLOW = warning -- approaching limit
  RED    = critical -- breach or halt
  BLUE   = informational (no action)
  WHITE  = default terminal text

Usage
-----
    # In runner.py:
    dashboard = TerminalDashboard()
    dashboard.start()

    # After each fill:
    dashboard.update_portfolio(equity=..., positions=..., margin_used=...)
    dashboard.update_risk(state="ACTIVE", maxdd_pct=-0.042, sharpe_3m=0.89)
    dashboard.update_greeks(delta=-2.1, gamma=+0.04, vega=+8.7, theta=-3.2)
    dashboard.update_tca(avg_slippage_bps=4.1, worst_bps=21.3, sweeps=1)
    dashboard.update_connection(status="CONNECTED")

    # At end of session:
    dashboard.stop()

    # For Textual interactive mode:
    dashboard = TerminalDashboard(mode="textual")
"""

from __future__ import annotations

import os
import sys
import threading
import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

_log = __import__("logging").getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI Color Codes (Windows-compatible via colorama or stdlib)
# ---------------------------------------------------------------------------

# Try to get the terminal width
try:
    _TERMINAL_WIDTH = os.get_terminal_size().columns
except OSError:
    _TERMINAL_WIDTH = 80

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_BLINK = "\033[5m"
ANSI_RESET_BLINK = "\033[25m"

# Foreground
F_black  = "\033[30m"
F_red    = "\033[31m"
F_green  = "\033[32m"
F_yellow = "\033[33m"
F_blue   = "\033[34m"
F_magenta= "\033[35m"
F_cyan   = "\033[36m"
F_white  = "\033[37m"
F_default= "\033[39m"

# Background
B_red    = "\033[41m"
B_green  = "\033[42m"
B_yellow = "\033[43m"
B_blue   = "\033[44m"

# Style combinators
def _color(text: str, fg: str, bg: str = "", bold: bool = False, blink: bool = False) -> str:
    parts = [fg]
    if bg:
        parts.append(bg)
    if bold:
        parts.append(ANSI_BOLD)
    if blink:
        parts.append(ANSI_BLINK)
    parts.append(text)
    parts.append(ANSI_RESET)
    return "".join(parts)


# Convenience helpers
def green(text: str, bold: bool = False) -> str:
    return _color(text, F_green, bold=bold)

def yellow(text: str, bold: bool = False) -> str:
    return _color(text, F_yellow, bold=bold)

def red(text: str, bold: bool = False, blink: bool = False) -> str:
    return _color(text, F_red, B_red if blink else "", bold=bold, blink=blink)

def blue(text: str, bold: bool = False) -> str:
    return _color(text, F_blue, bold=bold)

def bold_white(text: str) -> str:
    return _color(text, F_white, bold=True)

def status_indicator(level: str) -> str:
    """Return a color-dot string for status level: 'green', 'yellow', 'red'."""
    icons = {"green": "●", "yellow": "◐", "red": "●", "grey": "○", "unknown": "?"}
    icon = icons.get(level, "?")
    if level == "green":
        return _color(icon, F_green, bold=True)
    elif level == "yellow":
        return _color(icon, F_yellow, bold=True)
    elif level == "red":
        return _color(icon, F_red, B_red, bold=True, blink=True)
    elif level == "grey":
        return _color(icon, F_default)
    return _color(icon, F_default)


# ---------------------------------------------------------------------------
# Status classification helpers
# ---------------------------------------------------------------------------

def _delta_level(delta: float) -> tuple[str, str]:
    """Return (color_level, display_str)."""
    abs_d = abs(delta)
    if abs_d < 5:
        return "green", f"{delta:+.1f}"
    if abs_d < 15:
        return "yellow", f"{delta:+.1f}"
    return "red", f"{delta:+.1f}"


def _drawdown_level(dd_pct: float) -> tuple[str, str]:
    """Return (color_level, display_str)."""
    abs_dd = abs(dd_pct)
    if abs_dd < 0.05:
        return "green", f"{dd_pct:+.1%}"
    if abs_dd < 0.10:
        return "yellow", f"{dd_pct:+.1%}"
    return "red", f"{dd_pct:+.1%}"


def _sharpe_level(sharpe: float) -> tuple[str, str]:
    if sharpe >= 0.5:
        return "green", f"{sharpe:.2f}"
    if sharpe >= 0.0:
        return "yellow", f"{sharpe:.2f}"
    return "red", f"{sharpe:.2f}"


def _slippage_level(bps: float) -> tuple[str, str]:
    if bps < 10:
        return "green", f"{bps:.1f}bps"
    if bps < 25:
        return "yellow", f"{bps:.1f}bps"
    return "red", f"{bps:.1f}bps"


# ---------------------------------------------------------------------------
# Dashboard state dataclasses (CoW snapshots)
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    starting_capital: float = 100_000.0
    current_equity: float = 100_000.0
    peak_equity: float = 100_000.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    margin_used_pct: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0
    net_delta: float = 0.0
    largest_position_delta: float = 0.0
    largest_position_symbol: str = ""
    day_pnl: str = ""   # formatted


@dataclass
class GreeksState:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0      # daily theta in $
    delta_dollar: float = 0.0

    @property
    def delta_status(self) -> str:
        return _delta_level(self.delta)[0]
    @property
    def delta_display(self) -> str:
        return _delta_level(self.delta)[1]


@dataclass
class RiskState:
    state: str = "UNKNOWN"       # ACTIVE / PAUSED / HALTED / UNKNOWN
    maxdd_pct: float = 0.0
    sharpe_3m: float = 0.0
    macro_triggered: bool = False
    manual_reset_count: int = 0

    @property
    def state_display(self) -> tuple[str, str]:
        s = self.state.upper()
        if s == "ACTIVE":
            return "green", "ACTIVE"
        if s == "PAUSED":
            return "yellow", "PAUSED"
        if s in {"HALTED", "FLOW审"}:
            return "red", "HALTED"
        return "grey", s

    @property
    def kill_status(self) -> list[tuple[str, str, str]]:
        """List of (name, level, status_str) for each kill switch."""
        items = [
            ("MaxDD", _drawdown_level(self.maxdd_pct)[0], _drawdown_level(self.maxdd_pct)[1] + " DD"),
        ]
        sharpe_lvl, sharpe_str = _sharpe_level(self.sharpe_3m)
        items.append(("Sharpe", sharpe_lvl, f"3m {sharpe_str}"))
        macro_lvl = "red" if self.macro_triggered else "green"
        macro_str = "TRIGGERED" if self.macro_triggered else "ARMED"
        items.append(("Macro", macro_lvl, macro_str))
        return items


@dataclass
class SystemState:
    feed_status: str = "UNKNOWN"   # CONNECTED / DEGRADED / DISCONNECTED
    pid: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    runtime_seconds: float = 0.0
    # Kill switch status for each of the 3 controller switches
    kill_switch_1: str = "ARMED"     # MaxDD
    kill_switch_2: str = "ARMED"     # Sharpe
    kill_switch_3: str = "ARMED"      # Macro
    # Connection health per feed
    feed_health: dict[str, str] = field(default_factory=dict)  # feedname → status

    @property
    def feed_status_level(self) -> str:
        s = self.feed_status.upper()
        if "CONNECT" in s:
            return "green"
        if "DEGRADED" in s:
            return "yellow"
        if "DISCONNECT" in s:
            return "red"
        return "grey"

    def runtime_display(self) -> str:
        """HH:MM:SS from start_time."""
        elapsed = (datetime.now(UTC) - self.start_time).total_seconds()
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class TCAState:
    avg_slippage_bps: float = 0.0
    worst_slippage_bps: float = 0.0
    worst_order_id: str = ""
    total_sweeps: int = 0
    rejection_count: int = 0
    grade_distribution: dict[str, int] = field(default_factory=dict)
    fill_count: int = 0

    @property
    def slip_status(self) -> str:
        return _slippage_level(self.avg_slippage_bps)[0]

    @property
    def slip_display(self) -> str:
        return _slippage_level(self.avg_slippage_bps)[1]


@dataclass
class DashboardSnapshot:
    """Aggregate snapshot containing all panels."""
    portfolio: PortfolioState = field(default_factory=PortfolioState)
    greeks: GreeksState = field(default_factory=GreeksState)
    risk: RiskState = field(default_factory=RiskState)
    system: SystemState = field(default_factory=SystemState)
    tca: TCAState = field(default_factory=TCAState)


# ---------------------------------------------------------------------------
# ANSI Dashboard Renderer (Pure stdlib — no external dependencies)
# ---------------------------------------------------------------------------

class ANSIDashboardRenderer:
    """Renders the dashboard using ANSI escape codes.

    Lightweight, no external dependencies.  Overwrites the terminal with
    ANSI sequences — compatible with cmd.exe, PowerShell, and most terminals.
    """

    _MIN_WIDTH = 78

    def __init__(self, width: int | None = None) -> None:
        try:
            self._w = width or os.get_terminal_size().columns
        except OSError:
            self._w = self._MIN_WIDTH
        self._w = max(self._MIN_WIDTH, self._w)
        self._hide_cursor = "\033[?25l"
        self._show_cursor = "\033[?25h"
        self._home = "\033[H"
        # Ensure stdout can handle Unicode box-drawing chars (Windows cp1252 fallback)
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass  # Python < 3.7 or non-terminal stdout

    def _line(self, text: str) -> str:
        """Pad text to terminal width."""
        return text[: self._w - 1].ljust(self._w)

    def _header(self, title: str) -> str:
        pad = self._w - len(title) - 4
        left = pad // 2
        right = pad - left
        sep = "═" * left
        sep2 = "═" * right
        return self._line(f"┌{sep} {title} {sep2}┐")

    def _row(self, label: str, value: str, label_color: str = "", val_color: str = "") -> str:
        lbl = (label_color + label + ANSI_RESET).ljust(28)
        val = val_color + value + ANSI_RESET
        return f"║  {lbl}{val.ljust(self._w - 32)}  ║"

    def _section(self, title: str) -> str:
        items = []
        sep_len = max(self._w - len(title) - 6, 0)
        separator = "_" * sep_len
        items.append(self._line("+- " + title + " -" + separator + "+"))
        return "\n".join(items)

    def _footer(self) -> str:
        return self._line("└" + "─" * (self._w - 2) + "┘")

    def render(self, snap: DashboardSnapshot) -> str:
        """Render a complete dashboard frame as a multiline string."""
        W = self._w
        sep = "─" * (W - 4)

        # ---- Header ------------------------------------------------------------
        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S ET")
        header = self._line(
            f"╔{sep}╗\n"
            f"║ {bold_white('DELTA NEUTRAL VRP — LIVE MONITOR').ljust(W - 4)} ║\n"
            f"║ {blue(now_str).ljust(W - 4)} ║\n"
            f"╚{sep}╝"
        )

        # ---- Portfolio --------------------------------------------------------
        pf = snap.portfolio
        rr = snap.risk
        dd_lvl, dd_str = _drawdown_level(pf.current_drawdown_pct)
        dd_display = _color(f"{pf.current_drawdown_pct:+.2%}", F_red if dd_lvl == "red" else F_yellow if dd_lvl == "yellow" else F_green)

        row = []
        row.append(f"║  Starting Capital  {green(f'${pf.starting_capital:,.0f}').ljust(W-36)}   ║")
        row.append(f"║  Peak Equity       {green(f'${pf.peak_equity:,.2f}').ljust(W-36)}   ║")
        row.append(f"║  Current Equity    {bold_white(f'${pf.current_equity:,.2f}').ljust(W-36)}   ║")
        row.append(f"║  Drawdown         {dd_display.ljust(W-36)}   ║")
        row.append(f"║  Day P&L          {green(f'${pf.daily_pnl:+,.2f}').ljust(W-36)}   ║")
        row.append(f"║  Margin Used      {green(f'{pf.margin_used_pct:.1%}').ljust(W-36)}   ║")
        row.append(f"║  Open Positions   {bold_white(str(pf.open_positions)).ljust(W-36)}   ║")

        portfolio_block = (
            self._line("╔══ CAP HEALTH ═════════════════════════════════════════════╗")
            + "\n" + "\n".join(self._line(r) for r in row)
            + "\n" + self._line("╚════════════════════════════════════════════════════════╝")
        )

        # ---- Risk Status -------------------------------------------------------
        state_lvl, state_str = snap.risk.state_display
        state_color = F_green if state_lvl == "green" else F_yellow if state_lvl == "yellow" else F_red
        state_display = _color(state_str, state_color, bold=True)

        maxdd_lvl, maxdd_str = _drawdown_level(snap.risk.maxdd_pct)
        maxdd_color = F_red if maxdd_lvl == "red" else F_yellow if maxdd_lvl == "yellow" else F_green
        maxdd_display = _color(maxdd_str, maxdd_color)

        sharpe_lvl, sharpe_str = _sharpe_level(snap.risk.sharpe_3m)
        sharpe_color = F_green if sharpe_lvl == "green" else F_yellow if sharpe_lvl == "yellow" else F_red
        sharpe_display = _color(f"3m {sharpe_str}", sharpe_color)

        macro_color = F_red if snap.risk.macro_triggered else F_green
        macro_lbl = F_red if snap.risk.macro_triggered else F_green
        macro_display = _color("TRIGGERED" if snap.risk.macro_triggered else "ok", macro_lbl)
        con_lvl = snap.system.feed_status_level
        con_color = F_green if con_lvl == "green" else F_yellow if con_lvl == "yellow" else F_red
        con_display = _color(snap.system.feed_status, con_color)

        runtime_str = snap.system.runtime_display()
        pid_str = str(snap.system.pid or os.getpid())

        risk_block = (
            self._line("╔══ SAFETY ══════════════════════════════════════════════════╗")
            + "\n"
            + self._line(f"║  Strategy State  {state_display.ljust(W-4)}   ║")
            + "\n"
            + self._line(f"║  MaxDD Kill   {status_indicator(maxdd_lvl).ljust(8)}  {maxdd_display.ljust(W-28)}   ║")
            + "\n"
            + self._line(f"║  Sharpe Halt  {status_indicator(sharpe_lvl).ljust(8)}  {sharpe_display.ljust(W-28)}   ║")
            + "\n"
            + self._line(f"║  Macro Exit   {status_indicator('red' if snap.risk.macro_triggered else 'green').ljust(8)}  {macro_display.ljust(W-28)}   ║")
            + "\n"
            + self._line(f"║  Feed Status  {con_display.ljust(W-4)}   ║")
            + "\n"
            + self._line(f"║  PID  {bold_white(pid_str).ljust(8)}  Runtime  {bold_white(runtime_str).ljust(W-30)}   ║")
            + "\n"
            + self._line("╚════════════════════════════════════════════════════════════╝")
        )

        # ---- Greeks -----------------------------------------------------------
        gk = snap.greeks
        def greeks_color(v: float, pos_ok: bool = True) -> str:
            if abs(v) < 2.0:
                return F_green
            if abs(v) < 8.0:
                return F_yellow
            return F_red

        delta_str = _color(f"{gk.delta:+.2f}", greeks_color(abs(gk.delta)))
        gamma_str = _color(f"{gk.gamma:+.4f}", greeks_color(gk.gamma))
        vega_str  = _color(f"{gk.vega:+.2f}", greeks_color(gk.vega))
        theta_str = _color(f"${gk.theta:+.2f}/day", F_yellow if abs(gk.theta) < 5 else F_red)

        greeks_block = (
            self._line("╔══ GREEKS ════════════════════════════════════════════════╗")
            + "\n"
            + self._line(f"║  Δ  {delta_str}   │  Γ  {gamma_str}   │  ν  {vega_str}   │  Θ  {theta_str.ljust(W-52)}   ║")
            + "\n"
            + self._line(f"║  Net Delta  {Greek_indicator(gk.delta, 5, 15).ljust(W-36)}   ║")
            + "\n"
            + self._line("╚══════════════════════════════════════════════════════════╝")
        )

        # ---- TCA --------------------------------------------------------------
        tc = snap.tca
        slip_color = F_green if tc.slip_status == "green" else F_yellow if tc.slip_status == "yellow" else F_red
        slip_display = _color(
            f"{tc.avg_slippage_bps:.1f}bps", slip_color
        )
        worst_display = _color(f"wc {tc.worst_slippage_bps:.1f}bps", F_red if tc.worst_slippage_bps > 25 else F_yellow)
        sweep_color = F_red if tc.total_sweeps > 0 else F_green
        sweep_display = _color(str(tc.total_sweeps), sweep_color, bold=tc.total_sweeps > 0)
        rej_color = F_red if tc.rejection_count > 0 else F_green
        rej_display = _color(str(tc.rejection_count), rej_color)
        total_fills = _color(str(tc.fill_count), F_blue)
        grades = ", ".join(f"{g}:{c}" for g, c in sorted(tc.grade_distribution.items()))

        tca_block = (
            self._line("╔══ TCA ════════════════════════════════════════════════════╗")
            + "\n"
            + self._line(f"║  Avg Slip  {slip_display}   │  Worst  {worst_display}   │  Sweeps  {sweep_display}   │  Rejs  {rej_display.ljust(W-57)}   ║")
            + "\n"
            + self._line(f"║  Fills  {total_fills}  │  Grades  {blue(grades).ljust(W-42)}   ║")
            + "\n"
            + self._line("╚═══════════════════════════════════════════════════════════╝")
        )

        return f"{self._hide_cursor}{self._home}\n{header}\n{portfolio_block}\n{risk_block}\n{greeks_block}\n{tca_block}\n{self._show_cursor}"


class ANSIDashboardRendererTextual(ANSIDashboardRenderer):
    """Extended renderer with blinking red for critical states (Textual compat)."""
    pass


def Greek_indicator(delta: float, warn_threshold: float, critical_threshold: float) -> str:
    abs_d = abs(delta)
    if abs_d < warn_threshold:
        return green(f"{delta:+.1f}", bold=True)
    if abs_d < critical_threshold:
        return yellow(f"{delta:+.1f}", bold=True)
    return red(f"{delta:+.1f}", bold=True, blink=True)


# ---------------------------------------------------------------------------
# Optional Textual Dashboard (rich, interactive)
# ---------------------------------------------------------------------------

def _try_import_textual():
    """Try to import Textual. Returns (module_or_None, is_available)."""
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Static
        from textual.reactive import reactive
        return (True,)
    except ImportError:
        return (False,)


_TEXTUAL_AVAILABLE = _try_import_textual()[0]


# ---------------------------------------------------------------------------
# TerminalDashboard — main orchestrator
# ---------------------------------------------------------------------------

class TerminalDashboard:
    r"""Real-time terminal dashboard for the PaperTradingRunner.

    Runs a background refresh thread that polls shared state (via
    Copy-on-Write snapshots) and updates the terminal.  Does NOT block
    the main execution thread.

    The runner calls ``update_*`` methods to push new data into the shared
    snapshot dict.  The background thread reads it and re-renders.

    Panel Layout
    ------------
    ┌═══════════════╗  ┌═══════════════════════════════════════╗
    ║  CAP HEALTH    ║  ║  PORTFOLIO GREEKS                    ║
    ║  Equity: $X    ║  ║  Δ  Γ  ν  Θ                         ║
    ║  DD: Y%        ║  ╠═════════════════════════════════════╣
    ║  P&L: Z        ║  ║  RISK KILL SWITCHES                  ║
    ╠═══════════════╣  ║  MAXDD ●  SHARPE ●  MACRO ●           ║
    ║  TCA METRICS   ║  ╠═════════════════════════════════════╣
    ║  Slip: N bps  ║  ║  SYSTEM STATUS                       ║
    ║  Sweep: M     ║  ║  Feed ● | PID | Runtime HH:MM:SS    ║
    ╚═══════════════╝  ╚═════════════════════════════════════╝

    Parameters
    ----------
    mode : str
        "ansi" (default, no dependencies) or "textual" (interactive TUI).
    refresh_interval_sec : float
        How often to redraw.  Default 1.0 s.
    starting_capital : float
        For the capital health panel.  Default $100,000.

    Thread Safety
    -------------
    ``update_*`` methods acquire ``self._lock`` and swap a shared dict.
    The background thread reads the latest dict value atomically.
    """

    def __init__(
        self,
        mode: str = "ansi",
        refresh_interval_sec: float = 1.0,
        starting_capital: float = 100_000.0,
        risk_controller: "RiskController | None" = None,
        tca_engine: "TCAEngine | None" = None,
        hedger: "AdaptiveHedger | None" = None,
    ) -> None:
        self._mode = mode
        self._interval = refresh_interval_sec
        self._starting_capital = starting_capital
        self._rc = risk_controller
        self._tca = tca_engine
        self._hedger = hedger
        self._lock = threading.Lock()
        self._running = False
        self._render_thread: threading.Thread | None = None

        # Shared snapshot written by update_* calls, read by render thread
        self._snap = DashboardSnapshot(
            system=SystemState(pid=os.getpid()),
        )

        self._renderer = ANSIDashboardRenderer()

        _log.info(
            "TerminalDashboard: mode=%s refresh=%.1fs starting_capital=$%.0f",
            mode,
            refresh_interval_sec,
            starting_capital,
        )

    # ------------------------------------------------------------------
    # Public API — called by the runner (may be called from any thread)
    # ------------------------------------------------------------------

    def update_portfolio(
        self,
        equity: float,
        peak_equity: float,
        current_drawdown_pct: float,
        max_drawdown_pct: float,
        margin_used_pct: float,
        daily_pnl: float,
        open_positions: int,
        net_delta: float,
        largest_position_delta: float | None = None,
        largest_position_symbol: str = "",
    ) -> None:
        with self._lock:
            self._snap.portfolio = PortfolioState(
                starting_capital=self._starting_capital,
                current_equity=equity,
                peak_equity=peak_equity,
                current_drawdown_pct=current_drawdown_pct,
                max_drawdown_pct=max_drawdown_pct,
                margin_used_pct=margin_used_pct,
                daily_pnl=daily_pnl,
                open_positions=open_positions,
                net_delta=net_delta,
                largest_position_delta=largest_position_delta or net_delta,
                largest_position_symbol=largest_position_symbol,
                day_pnl=f"${daily_pnl:+,.2f}",
            )

    def update_greeks(
        self,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        delta_dollar: float | None = None,
    ) -> None:
        with self._lock:
            self._snap.greeks = GreeksState(
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                delta_dollar=delta_dollar if delta_dollar is not None else delta * 450,
            )

    def update_risk(
        self,
        state: str,
        maxdd_pct: float,
        sharpe_3m: float,
        macro_triggered: bool,
        manual_reset_count: int = 0,
    ) -> None:
        with self._lock:
            self._snap.risk = RiskState(
                state=state,
                maxdd_pct=maxdd_pct,
                sharpe_3m=sharpe_3m,
                macro_triggered=macro_triggered,
                manual_reset_count=manual_reset_count,
            )

    def update_system(
        self,
        feed_status: str,
        feed_health: dict[str, str] | None = None,
        kill_switch_1: str | None = None,
        kill_switch_2: str | None = None,
        kill_switch_3: str | None = None,
        start_time: datetime | None = None,
    ) -> None:
        with self._lock:
            cur = self._snap.system
            self._snap.system = SystemState(
                feed_status=feed_status,
                pid=cur.pid or os.getpid(),
                start_time=start_time or cur.start_time,
                runtime_seconds=cur.runtime_seconds,
                kill_switch_1=kill_switch_1 or cur.kill_switch_1,
                kill_switch_2=kill_switch_2 or cur.kill_switch_2,
                kill_switch_3=kill_switch_3 or cur.kill_switch_3,
                feed_health=feed_health or cur.feed_health,
            )

    def update_tca(
        self,
        avg_slippage_bps: float,
        worst_slippage_bps: float,
        worst_order_id: str = "",
        total_sweeps: int = 0,
        rejection_count: int = 0,
        grade_distribution: dict[str, int] | None = None,
        fill_count: int = 0,
    ) -> None:
        with self._lock:
            self._snap.tca = TCAState(
                avg_slippage_bps=avg_slippage_bps,
                worst_slippage_bps=worst_slippage_bps,
                worst_order_id=worst_order_id,
                total_sweeps=total_sweeps,
                rejection_count=rejection_count,
                grade_distribution=grade_distribution or {},
                fill_count=fill_count,
            )

    def update_connection(self, status: str, feed_health: dict[str, str] | None = None) -> None:
        self.update_system(feed_status=status, feed_health=feed_health)

    def update_hedger_state(self, hedger_state: dict[str, Any]) -> None:
        """Update system state from AdaptiveHedger.current_state()."""
        with self._lock:
            cur = self._snap.system
            threshold = hedger_state.get("threshold", 5.0)
            self._snap.system = SystemState(
                feed_status=cur.feed_status,
                pid=cur.pid,
                start_time=cur.start_time,
                runtime_seconds=(datetime.now(UTC) - cur.start_time).total_seconds(),
                kill_switch_1=cur.kill_switch_1,
                kill_switch_2=cur.kill_switch_2,
                kill_switch_3=cur.kill_switch_3,
                feed_health={**cur.feed_health, **hedger_state},
            )

    # Pull live TCA snapshot from TCAEngine
    def sync_from_tca(self) -> None:
        if self._tca is None:
            return
        snap_tca = self._tca.get_snapshot()
        self.update_tca(
            avg_slippage_bps=snap_tca.avg_slippage_bps,
            worst_slippage_bps=snap_tca.worst_slippage_bps,
            worst_order_id=snap_tca.worst_order_id,
            total_sweeps=snap_tca.sweeeps_triggered,
            rejection_count=snap_tca.rejection_count,
            grade_distribution=snap_tca.grade_distribution,
            fill_count=snap_tca.fill_count,
        )

    # Pull live RiskController snapshot
    def sync_from_risk_controller(self) -> None:
        if self._rc is None:
            return
        summary = self._rc.get_summary()
        self.update_risk(
            state=summary.get("state", "UNKNOWN"),
            maxdd_pct=summary.get("current_drawdown_pct", 0.0),
            sharpe_3m=summary.get("sharpe_3m", 0.0),
            macro_triggered=summary.get("macro_triggered", False),
            manual_reset_count=summary.get("reset_count", 0),
        )

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="TerminalDashboard"
        )
        self._render_thread.start()
        _log.info("TerminalDashboard started (mode=%s interval=%.1fs)", self._mode, self._interval)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._render_thread:
            self._render_thread.join(timeout=self._interval + 2.0)
        # Restore cursor
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        _log.info("TerminalDashboard stopped")

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def _render_loop(self) -> None:
        while self._running:
            try:
                with self._lock:
                    snap = DashboardSnapshot(
                        portfolio=DashboardSnapshot.portfolio.__class__(**self._snap.portfolio.__dict__),
                        greeks=GreeksState(**self._snap.greeks.__dict__),
                        risk=RiskState(**self._snap.risk.__dict__),
                        system=SystemState(**self._snap.system.__dict__),
                        tca=TCAState(**self._snap.tca.__dict__),
                    )
                frame = self._renderer.render(snap)
                sys.stdout.write(frame)
                sys.stdout.flush()
            except UnicodeEncodeError:
                # Fallback: ASCII-only render (no Unicode box chars)
                try:
                    sys.stdout.write("\033[2J\033[H")  # clear screen
                    sys.stdout.write(
                        f"[{datetime.now(UTC).strftime('%H:%M:%S')} "
                        f"P: ${self._snap.portfolio.current_equity:,.0f}  "
                        f"D: {self._snap.portfolio.current_drawdown_pct:+.1%}  "
                        f"Δ: {self._snap.greeks.delta:+.1f}  "
                        f"S: {self._snap.risk.state!s}  "
                        f"TCA: {self._snap.tca.avg_slippage_bps:.1f}bps\n"
                    )
                    sys.stdout.flush()
                except Exception:
                    pass
            except Exception as exc:
                _log.error("Dashboard render error: %s", exc)
            _time.sleep(self._interval)

    def refresh(self) -> None:
        """Force a synchronous refresh (call from runner if needed)."""
        with self._lock:
            snap = DashboardSnapshot(
                portfolio=DashboardSnapshot.portfolio.__class__(**self._snap.portfolio.__dict__),
                greeks=GreeksState(**self._snap.greeks.__dict__),
                risk=RiskState(**self._snap.risk.__dict__),
                system=SystemState(**self._snap.system.__dict__),
                tca=TCAState(**self._snap.tca.__dict__),
            )
        frame = self._renderer.render(snap)
        sys.stdout.write(frame)
        sys.stdout.flush()