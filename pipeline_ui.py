# pipeline_ui.py — FineHero ensemble pipeline with a live tkinter dashboard.
#
#   python pipeline_ui.py                 # everything except cleanlab
#   python pipeline_ui.py --cleanlab      # + label-noise scan
#   python pipeline_ui.py --fetch         # also refetch raw NYC data
#   python pipeline_ui.py --skip-engineer # use existing features.csv
#
# Pops up a dark-themed window with:
#   - step cards (spinner, elapsed, ETA per step)
#   - a Gantt-style timeline with live "now" cursor
#   - a colored live log tail
#   - overall progress bar + total ETA in minutes/hours
#
# Step-duration estimates live in models/pipeline_timings.json and learn from
# every completed run (EMA, alpha=0.5). First run uses DEFAULT_TIMINGS.

import argparse
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import font, ttk


ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
TIMINGS_PATH = os.path.join(MODELS_DIR, "pipeline_timings.json")

# Conservative defaults — EMA will refine them after the first successful run.
DEFAULT_TIMINGS = {
    "fetch":     600,      # 10 min
    "engineer":  240,      # 4 min
    "train_cb":  7200,     # 2 hr
    "evaluate":  45,
    "train_lgb": 1500,     # 25 min
    "train_xgb": 1800,     # 30 min
    "ensemble":  45,
    "cleanlab":  3600,     # 1 hr
}


def load_timings():
    try:
        with open(TIMINGS_PATH, "r") as f:
            disk = json.load(f)
        return {**DEFAULT_TIMINGS, **disk}
    except (OSError, ValueError):
        return dict(DEFAULT_TIMINGS)


def save_timings(timings):
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(TIMINGS_PATH, "w") as f:
            json.dump(timings, f, indent=2)
    except OSError:
        pass


# ----- stdout capture -------------------------------------------------------

class TeeStream:
    """Duplicate writes to the original stream AND push full lines to a queue."""
    def __init__(self, original, log_queue):
        self.original = original
        self.queue = log_queue
        self._buf = ""

    def write(self, s):
        try:
            self.original.write(s)
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.queue.put(("log", line))

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self.original.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self.original, name)


# ----- pipeline UI ----------------------------------------------------------

class PipelineUI:
    BG      = "#0d1117"
    PANEL   = "#161b22"
    PANEL2  = "#010409"
    BORDER  = "#30363d"
    TEXT    = "#c9d1d9"
    MUTED   = "#8b949e"
    FAINT   = "#30363d"
    ACCENT  = "#58a6ff"
    OK      = "#3fb950"
    ERR     = "#f85149"
    RUN     = "#d29922"

    SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    def __init__(self, steps):
        self.steps = steps                       # [(key, title, fn), ...]
        self.n = len(steps)
        self.timings = load_timings()
        self.status     = ["pending"] * self.n
        self.starts     = [None] * self.n        # wall-clock start per step
        self.durations  = [None] * self.n        # actual duration on completion
        self.current    = -1
        self.spinner_i  = 0
        self.queue      = queue.Queue()
        self.t0         = None
        self.finished   = False
        self._gantt_spans = None                 # [(start, end), ...] from last refresh

        self.root = tk.Tk()
        self.root.title("FineHero Pipeline")
        self.root.geometry("1320x880")
        self.root.configure(bg=self.BG)
        self.root.minsize(1050, 720)

        self._build()
        self.root.after(100, self._tick)
        self.root.after(100, self._pump_queue)

    # ---- expected durations ------------------------------------------------
    @property
    def expected(self):
        return [float(self.timings.get(k, 60)) for k, _, _ in self.steps]

    # ---- layout ------------------------------------------------------------
    def _build(self):
        self.mono   = font.Font(family="Consolas", size=10)
        self.mono_b = font.Font(family="Consolas", size=10, weight="bold")
        self.small  = font.Font(family="Consolas", size=9)
        title_f     = font.Font(family="Segoe UI", size=22, weight="bold")
        sub_f       = font.Font(family="Segoe UI", size=10)
        eta_big     = font.Font(family="Segoe UI", size=28, weight="bold")

        # --- header ---
        header = tk.Frame(self.root, bg=self.BG)
        header.pack(fill="x", padx=24, pady=(18, 10))
        lh = tk.Frame(header, bg=self.BG)
        lh.pack(side="left")
        tk.Label(lh, text="FineHero ML", font=title_f,
                 bg=self.BG, fg=self.ACCENT).pack(anchor="w")
        tk.Label(lh,
                 text="Ensemble Pipeline \u00b7 CatBoost + LightGBM + XGBoost \u00b7 rank-averaged blend",
                 font=sub_f, bg=self.BG, fg=self.MUTED).pack(anchor="w")
        rh = tk.Frame(header, bg=self.BG)
        rh.pack(side="right")
        tk.Label(rh, text="TOTAL ETA", font=self.small,
                 bg=self.BG, fg=self.MUTED).pack(anchor="e")
        self.eta_label = tk.Label(rh, text="--", font=eta_big,
                                   bg=self.BG, fg=self.ACCENT)
        self.eta_label.pack(anchor="e")
        self.elapsed_label = tk.Label(rh, text="0s elapsed", font=self.small,
                                       bg=self.BG, fg=self.MUTED)
        self.elapsed_label.pack(anchor="e")

        # --- step cards row ---
        cards = tk.Frame(self.root, bg=self.BG)
        cards.pack(fill="x", padx=24, pady=(6, 6))
        self.step_cards = []
        for i, (_, title, _) in enumerate(self.steps):
            card = self._make_card(cards, i, title)
            card["frame"].pack(side="left", fill="both", expand=True, padx=4)
            self.step_cards.append(card)

        # --- middle body: gantt (top) + log (bottom) ---
        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=24, pady=(8, 6))

        # Gantt timeline
        gpanel = tk.Frame(body, bg=self.PANEL,
                          highlightbackground=self.BORDER, highlightthickness=1)
        gpanel.pack(fill="x", pady=(0, 8))
        tk.Label(gpanel, text="  TIMELINE", font=self.mono_b,
                 bg=self.PANEL, fg=self.MUTED).pack(anchor="w", padx=12, pady=(10, 4))
        self.gantt = tk.Canvas(gpanel, bg=self.PANEL2,
                                highlightbackground=self.BORDER, highlightthickness=1,
                                height=max(28 * self.n + 50, 190))
        self.gantt.pack(fill="x", padx=12, pady=(0, 12))
        self.gantt.bind("<Configure>", lambda e: self._draw_gantt())

        # Log
        lpanel = tk.Frame(body, bg=self.PANEL,
                          highlightbackground=self.BORDER, highlightthickness=1)
        lpanel.pack(fill="both", expand=True)
        tk.Label(lpanel, text="  LIVE LOG", font=self.mono_b,
                 bg=self.PANEL, fg=self.MUTED).pack(anchor="w", padx=12, pady=(10, 4))
        lf = tk.Frame(lpanel, bg=self.PANEL)
        lf.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log = tk.Text(lf, bg=self.PANEL2, fg=self.TEXT, font=self.mono,
                           wrap="none", relief="flat", padx=10, pady=10,
                           insertbackground=self.ACCENT,
                           highlightbackground=self.BORDER, highlightthickness=1)
        self.log.tag_configure("err", foreground=self.ERR)
        self.log.tag_configure("ok",  foreground=self.OK)
        self.log.tag_configure("dim", foreground=self.MUTED)
        self.log.tag_configure("acc", foreground=self.ACCENT)
        sby = tk.Scrollbar(lf, command=self.log.yview, bg=self.PANEL)
        self.log.config(yscrollcommand=sby.set)
        sby.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.config(state="disabled")

        # --- footer ---
        footer = tk.Frame(self.root, bg=self.BG)
        footer.pack(fill="x", padx=24, pady=(4, 18))
        self.status_label = tk.Label(footer, text="Ready", font=self.mono_b,
                                     bg=self.BG, fg=self.ACCENT)
        self.status_label.pack(anchor="w")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("H.Horizontal.TProgressbar",
                        troughcolor=self.PANEL, background=self.ACCENT,
                        bordercolor=self.BORDER, lightcolor=self.ACCENT,
                        darkcolor=self.ACCENT, thickness=8)
        self.pbar = ttk.Progressbar(footer, style="H.Horizontal.TProgressbar",
                                    length=100, mode="determinate",
                                    maximum=self.n)
        self.pbar.pack(fill="x", pady=(10, 0))

    def _make_card(self, parent, i, title):
        card = tk.Frame(parent, bg=self.PANEL,
                        highlightbackground=self.BORDER, highlightthickness=1)
        inner = tk.Frame(card, bg=self.PANEL)
        inner.pack(fill="both", expand=True, padx=12, pady=10)
        top = tk.Frame(inner, bg=self.PANEL)
        top.pack(fill="x")
        num = tk.Label(top, text=f"{i+1:02d}", font=self.small,
                        bg=self.PANEL, fg=self.MUTED)
        num.pack(side="left")
        icon = tk.Label(top, text="\u25cb", font=self.mono_b,
                         bg=self.PANEL, fg=self.MUTED)
        icon.pack(side="right")
        name = tk.Label(inner, text=title, font=self.mono_b, wraplength=150,
                         bg=self.PANEL, fg=self.TEXT, anchor="w", justify="left")
        name.pack(anchor="w", pady=(6, 8), fill="x")
        dur = tk.Label(inner, text="", font=self.small,
                        bg=self.PANEL, fg=self.MUTED, anchor="w")
        dur.pack(anchor="w", fill="x")
        eta = tk.Label(inner, text="", font=self.small,
                        bg=self.PANEL, fg=self.MUTED, anchor="w")
        eta.pack(anchor="w", fill="x")
        return {"frame": card, "icon": icon, "num": num, "name": name,
                "dur": dur, "eta": eta}

    # ---- tick loops --------------------------------------------------------
    def _tick(self):
        self.spinner_i = (self.spinner_i + 1) % len(self.SPINNER)
        if 0 <= self.current < self.n and self.status[self.current] == "running":
            card = self.step_cards[self.current]
            card["icon"].config(text=self.SPINNER[self.spinner_i], fg=self.RUN)
        self._refresh_times()
        self._draw_gantt()
        self.root.after(150, self._tick)

    def _pump_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._append_log(msg[1])
                elif kind == "start":
                    self.t0 = time.time()
                elif kind == "step_start":
                    self._on_step_start(msg[1])
                elif kind == "step_done":
                    self._on_step_done(msg[1], msg[2])
                elif kind == "step_fail":
                    self._on_step_fail(msg[1], msg[2])
                elif kind == "done":
                    self._on_done(msg[1])
        except queue.Empty:
            pass
        self.root.after(100, self._pump_queue)

    # ---- state changes -----------------------------------------------------
    def _on_step_start(self, i):
        self.current = i
        self.status[i] = "running"
        self.starts[i] = time.time()
        card = self.step_cards[i]
        card["frame"].config(highlightbackground=self.RUN, highlightthickness=2)
        card["num"].config(fg=self.RUN)
        self.status_label.config(
            text=f"[{i+1}/{self.n}]  {self.steps[i][1]}", fg=self.RUN)

    def _on_step_done(self, i, dur):
        self.status[i] = "done"
        self.durations[i] = dur
        card = self.step_cards[i]
        card["icon"].config(text="\u2713", fg=self.OK)
        card["frame"].config(highlightbackground=self.OK, highlightthickness=2)
        card["num"].config(fg=self.OK)
        card["dur"].config(text=f"Took {self._fmt_dur(dur)}", fg=self.OK)
        card["eta"].config(text="")
        self.pbar["value"] = i + 1
        key = self.steps[i][0]
        prev = self.timings.get(key, dur)
        self.timings[key] = 0.5 * prev + 0.5 * dur
        save_timings(self.timings)

    def _on_step_fail(self, i, err):
        self.status[i] = "failed"
        self.finished = True
        card = self.step_cards[i]
        card["icon"].config(text="\u2717", fg=self.ERR)
        card["frame"].config(highlightbackground=self.ERR, highlightthickness=2)
        card["num"].config(fg=self.ERR)
        card["dur"].config(text="Failed", fg=self.ERR)
        card["eta"].config(text="")
        short = err[:80] + ("..." if len(err) > 80 else "")
        self.status_label.config(text=f"FAILED: {short}", fg=self.ERR)
        self.eta_label.config(text="--", fg=self.ERR)

    def _on_done(self, total):
        self.finished = True
        self.pbar["value"] = self.n
        self.status_label.config(
            text=f"\u2713  Pipeline complete \u00b7 {self._fmt_dur(total)}",
            fg=self.OK)
        self.eta_label.config(text="Done", fg=self.OK)

    # ---- time & ETA maths --------------------------------------------------
    def _speed_factor(self):
        """actual/expected ratio across completed steps. None if no data yet."""
        actual = 0.0
        expected = 0.0
        for i in range(self.n):
            if self.status[i] == "done":
                actual   += self.durations[i]
                expected += self.expected[i]
        if expected <= 0:
            return None
        # Clamp to a reasonable band so a single weird fast/slow step doesn't
        # dominate. Blend with 1.0 when we have limited data (shrinkage).
        raw = actual / expected
        n_done = sum(1 for s in self.status if s == "done")
        shrink = min(1.0, n_done / 3.0)        # full weight after ~3 steps
        blended = shrink * raw + (1.0 - shrink) * 1.0
        return max(0.25, min(6.0, blended))

    def _refresh_times(self):
        if self.t0 is None:
            for i in range(self.n):
                c = self.step_cards[i]
                c["dur"].config(text=f"Est {self._fmt_dur(self.expected[i])}",
                                 fg=self.MUTED)
                c["eta"].config(text="", fg=self.MUTED)
            total_expected = sum(self.expected)
            self.eta_label.config(text=self._fmt_dur_long(total_expected),
                                   fg=self.ACCENT)
            return

        now = time.time()
        el = now - self.t0
        self.elapsed_label.config(text=f"{self._fmt_dur(el)} elapsed")

        sf = self._speed_factor()        # may be None until a step completes
        # Live running-step signal: if elapsed / expected is already > sf,
        # prefer the observation (step is slower than pre-completed steps).
        running_sf = sf
        for i in range(self.n):
            if self.status[i] == "running":
                exp_i = self.expected[i]
                if exp_i > 0:
                    live = (now - self.starts[i]) / exp_i
                    # Only bump *up* — don't let a short elapsed pull us low.
                    running_sf = max(running_sf or 1.0, live)
                break

        def adj(exp_i):
            # projected duration for a not-yet-run step
            k = running_sf if running_sf is not None else 1.0
            return max(10.0, exp_i * k)

        # Build (start, end) spans relative to t0 for every step.
        spans = []
        cursor = 0.0
        for i in range(self.n):
            if self.status[i] == "done":
                s = self.starts[i] - self.t0
                e = s + self.durations[i]
            elif self.status[i] == "running":
                s = self.starts[i] - self.t0
                elapsed_here = now - self.starts[i]
                # projected total: scale expected by running_sf, but never
                # below elapsed_here (+ a small buffer so ETA isn't 0)
                proj = adj(self.expected[i])
                proj = max(proj, elapsed_here + 5.0)
                e = s + proj
            elif self.status[i] == "failed":
                s = self.starts[i] - self.t0
                e = s + (now - self.starts[i])
            else:  # pending
                s = cursor
                e = s + adj(self.expected[i])
            cursor = e
            spans.append((s, e))
        self._gantt_spans = spans

        # Update cards
        for i, (s, e) in enumerate(spans):
            c = self.step_cards[i]
            if self.status[i] in ("done", "failed"):
                continue
            if self.status[i] == "running":
                elapsed_here = now - self.starts[i]
                remaining = max(1, e - el)
                c["dur"].config(text=f"Running {self._fmt_dur(elapsed_here)}",
                                 fg=self.RUN)
                c["eta"].config(
                    text=f"~{self._fmt_dur(remaining)} left", fg=self.RUN)
            else:
                starts_in = max(0, s - el)
                c["dur"].config(
                    text=f"Est {self._fmt_dur(e - s)}", fg=self.MUTED)
                c["eta"].config(
                    text=f"Starts in ~{self._fmt_dur(starts_in)}", fg=self.MUTED)

        if not self.finished:
            total_end = spans[-1][1]
            remaining = max(0, total_end - el)
            self.eta_label.config(text=self._fmt_dur_long(remaining),
                                   fg=self.ACCENT)

    # ---- Gantt draw --------------------------------------------------------
    def _draw_gantt(self):
        c = self.gantt
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 60 or H < 60:
            return

        # Build spans: either live (from _refresh_times) or pure expected.
        if self._gantt_spans is not None:
            spans = list(self._gantt_spans)
        else:
            cursor = 0.0
            spans = []
            for i in range(self.n):
                spans.append((cursor, cursor + self.expected[i]))
                cursor += self.expected[i]

        total = max(0.001, spans[-1][1])
        if self.t0 is not None:
            total = max(total, time.time() - self.t0)

        pad_l = 150
        pad_r = 20
        pad_t = 34
        pad_b = 16
        chart_w = max(10, W - pad_l - pad_r)
        chart_h = max(10, H - pad_t - pad_b)
        row_h = max(18, int(chart_h / max(1, self.n)))

        # Pick a nice tick interval
        tick_sec = self._pick_tick(total)
        t = 0.0
        while t <= total + 1e-6:
            x = pad_l + (t / total) * chart_w
            c.create_line(x, pad_t - 8, x, pad_t + self.n * row_h,
                           fill=self.FAINT, dash=(2, 3))
            c.create_text(x, pad_t - 16, text=self._fmt_dur(t),
                           fill=self.MUTED, font=self.small)
            t += tick_sec

        # now-line
        if self.t0 is not None and not self.finished:
            x_now = pad_l + ((time.time() - self.t0) / total) * chart_w
            c.create_line(x_now, pad_t - 8, x_now, pad_t + self.n * row_h,
                           fill=self.ACCENT, width=2)
            c.create_text(x_now, pad_t + self.n * row_h + 6,
                           text="now", fill=self.ACCENT, font=self.small)

        # bars
        for i, (s, e) in enumerate(spans):
            y = pad_t + i * row_h + 4
            bar_h = row_h - 10
            x1 = pad_l + (s / total) * chart_w
            x2 = pad_l + (e / total) * chart_w
            # step label on left
            label = self.steps[i][1]
            if len(label) > 20:
                label = label[:18] + "\u2026"
            c.create_text(pad_l - 10, y + bar_h / 2, text=label, anchor="e",
                           fill=self.TEXT, font=self.small)
            # faint background (expected range)
            c.create_rectangle(x1, y, max(x1 + 1, x2), y + bar_h,
                                fill=self.FAINT, outline="")
            # filled overlay by status
            if self.status[i] == "done":
                c.create_rectangle(x1, y, x2, y + bar_h,
                                    fill=self.OK, outline="")
                fg = self.BG
            elif self.status[i] == "running":
                now = time.time() - self.t0
                x_n = min(x2, pad_l + (now / total) * chart_w)
                c.create_rectangle(x1, y, x_n, y + bar_h,
                                    fill=self.RUN, outline="")
                # dotted projection
                c.create_rectangle(x_n, y, x2, y + bar_h,
                                    outline=self.RUN, dash=(3, 3))
                fg = self.BG
            elif self.status[i] == "failed":
                c.create_rectangle(x1, y, x2, y + bar_h,
                                    fill=self.ERR, outline="")
                fg = self.BG
            else:
                fg = self.TEXT
            # duration text
            mid = (x1 + x2) / 2
            dur_txt = self._fmt_dur(e - s)
            if (x2 - x1) > 40:
                c.create_text(mid, y + bar_h / 2, text=dur_txt,
                               fill=fg, font=self.small)
            else:
                c.create_text(x2 + 4, y + bar_h / 2, text=dur_txt,
                               anchor="w", fill=self.MUTED, font=self.small)

    @staticmethod
    def _pick_tick(total_sec):
        """Pick a human-friendly axis tick interval."""
        candidates = [30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 10800, 21600]
        target = total_sec / 8.0
        for t in candidates:
            if t >= target:
                return t
        return candidates[-1]

    # ---- log ---------------------------------------------------------------
    def _append_log(self, line):
        tag = None
        ll = line.lower()
        if "[error]" in ll or "traceback" in ll or "error:" in ll or "valueerror" in ll:
            tag = "err"
        elif "auc" in ll and (":" in line or "=" in line):
            tag = "ok"
        elif " done in " in ll or "saved ->" in line or "saved to" in ll:
            tag = "ok"
        elif "step " in ll and ":" in line:
            tag = "acc"
        elif line.startswith("  "):
            tag = "dim"
        self.log.config(state="normal")
        if tag:
            self.log.insert("end", line + "\n", tag)
        else:
            self.log.insert("end", line + "\n")
        num = int(self.log.index("end-1c").split(".")[0])
        if num > 3000:
            self.log.delete("1.0", f"{num-3000}.0")
        self.log.see("end")
        self.log.config(state="disabled")

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _fmt_dur(sec):
        sec = float(max(0, sec))
        if sec < 60:
            return f"{sec:.0f}s"
        m, s = divmod(int(sec), 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    @staticmethod
    def _fmt_dur_long(sec):
        sec = float(max(0, sec))
        if sec < 60:
            return f"{sec:.0f} sec"
        m = sec / 60.0
        if m < 60:
            return f"{m:.0f} min"
        h = m / 60.0
        return f"{h:.1f} hr"

    # ---- worker ------------------------------------------------------------
    def _run_pipeline(self):
        self.queue.put(("start",))
        t_pipeline = time.time()
        for i, (_, title, fn) in enumerate(self.steps):
            self.queue.put(("step_start", i))
            print(f"\n{'-'*60}\n  STEP {i+1}: {title}\n{'-'*60}")
            t = time.time()
            try:
                fn()
            except Exception as exc:
                print(f"  [ERROR] {title} failed: {exc}", file=sys.stderr)
                traceback.print_exc()
                self.queue.put(("step_fail", i, str(exc)))
                return
            dur = time.time() - t
            print(f"  {title} done in {dur:.1f}s ({dur/60:.1f} min)")
            self.queue.put(("step_done", i, dur))
        self.queue.put(("done", time.time() - t_pipeline))

    def run(self):
        sys.stdout = TeeStream(sys.stdout, self.queue)
        sys.stderr = TeeStream(sys.stderr, self.queue)
        threading.Thread(target=self._run_pipeline, daemon=True).start()
        self.root.mainloop()


# ----- step assembly --------------------------------------------------------

def build_steps(args):
    steps = []

    if args.fetch:
        def fetch_step():
            from src.fetch_data import fetch_all
            fetch_all(args.rows)
        steps.append(("fetch", "Fetch raw NYC data", fetch_step))

    if not args.skip_engineer:
        def eng_step():
            from src.engineer import engineer_features
            engineer_features()
        steps.append(("engineer", "Engineer features", eng_step))

    if not args.skip_catboost:
        def cb_step():
            from src.train import train_models
            train_models()
        steps.append(("train_cb", "Train CatBoost", cb_step))

    if not args.skip_evaluate:
        def eval_step():
            from src.evaluate import evaluate
            evaluate()
        steps.append(("evaluate", "Evaluate \u2192 metadata", eval_step))

    if not args.skip_lgb:
        def lgb_step():
            import src.train_lgb as train_lgb
            train_lgb.main()
        steps.append(("train_lgb", "Train LightGBM", lgb_step))

    if not args.skip_xgb:
        def xgb_step():
            import src.train_xgb as train_xgb
            train_xgb.main()
        steps.append(("train_xgb", "Train XGBoost", xgb_step))

    if not args.skip_ensemble:
        def ens_step():
            import src.ensemble as ensemble
            ensemble.main()
        steps.append(("ensemble", "Rank-avg blend", ens_step))

    if args.cleanlab:
        def cl_step():
            import src.cleanlab_scan as cleanlab_scan
            if args.cleanlab_subsample:
                cleanlab_scan.SUBSAMPLE_ROWS = args.cleanlab_subsample
            cleanlab_scan.main()
        steps.append(("cleanlab", "Cleanlab scan", cl_step))

    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--skip-engineer", action="store_true")
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-lgb", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--only-cleanlab", action="store_true",
                        help="Shortcut for --skip-engineer --skip-catboost --skip-evaluate "
                             "--skip-lgb --skip-xgb --skip-ensemble --cleanlab")
    parser.add_argument("--cleanlab", action="store_true")
    parser.add_argument("--cleanlab-subsample", type=int, default=None)
    args = parser.parse_args()

    if args.only_cleanlab:
        args.skip_engineer = args.skip_catboost = args.skip_evaluate = True
        args.skip_lgb = args.skip_xgb = args.skip_ensemble = True
        args.cleanlab = True

    steps = build_steps(args)
    PipelineUI(steps).run()


if __name__ == "__main__":
    main()
