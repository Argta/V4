"""Claude Code Session Viewer — desktop GUI for browsing conversation history.
Features: browse, search, star, sort, annotate, export sessions."""

import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SESSIONS_DIR = Path.home() / ".claude" / "projects" / "C--Users-LENOVO"
STARS_FILE = Path.home() / ".claude" / "starred_sessions.json"
NOTES_FILE = Path.home() / ".claude" / "session_notes.json"

SORT_OPTIONS = {
    "日期最新 ↓": ("first_ts", True),
    "日期最早 ↑": ("first_ts", False),
    "消息最多 ↓": ("message_count", True),
    "消息最少 ↑": ("message_count", False),
    "时长最长 ↓": ("duration_min", True),
    "时长最短 ↑": ("duration_min", False),
    "星标优先": ("starred", True),
    "有备注优先": ("has_notes", True),
}


class SessionViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Claude Code 会话管理器")
        self.root.geometry("1160x750")
        self.root.minsize(900, 550)

        # Data
        self.sessions = []
        self.session_meta = {}
        self.current_uuid = None
        self.search_var = tk.StringVar()
        self.sort_var = tk.StringVar(value="日期最新 ↓")
        self.starred = self._load_stars()
        self.notes = self._load_notes()
        self._notes_dirty = False
        self._notes_save_job = None
        self._notes_suppress = False  # flag to suppress <<Modified>> during programmatic changes

        # Build UI
        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()

        # Bindings
        self.root.bind("<Control-s>", lambda e: self._toggle_star_current())
        self.root.bind("<Control-f>", lambda e: self.search_entry.focus_set())
        self.root.bind("<Control-n>", lambda e: self._focus_notes())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load
        self._load_sessions()

    # ── Persistence helpers ──────────────────────────────────

    def _load_json(self, path, default):
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return default()

    def _save_json(self, path, data):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_stars(self):
        loaded = self._load_json(STARS_FILE, lambda: [])
        return set(loaded) if isinstance(loaded, list) else set()

    def _save_stars(self):
        self._save_json(STARS_FILE, sorted(self.starred))

    def _load_notes(self):
        loaded = self._load_json(NOTES_FILE, lambda: {})
        return loaded if isinstance(loaded, dict) else {}

    def _save_notes(self):
        self._save_json(NOTES_FILE, self.notes)
        self._notes_dirty = False
        # Update marker indicators for all visible rows
        for item_id in self.tree.get_children():
            try:
                self.tree.set(item_id, "markers", self._markers_str(item_id))
            except Exception:
                pass

    # ── Layout ──────────────────────────────────────────────

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill=tk.X)

        # Search
        ttk.Label(bar, text="搜索:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(bar, textvariable=self.search_var, width=28,
                                      font=("Microsoft YaHei", 10))
        self.search_entry.pack(side=tk.LEFT, padx=(4, 4))
        self.search_entry.bind("<Return>", lambda e: self._do_search())
        self.search_entry.bind("<KeyRelease>", lambda e: self._do_search())
        ttk.Button(bar, text="搜索", command=self._do_search, width=5).pack(
            side=tk.LEFT, padx=(0, 8))

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # Sort
        ttk.Label(bar, text="排序:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=(0, 3))
        self.sort_combo = ttk.Combobox(bar, textvariable=self.sort_var,
                                       values=list(SORT_OPTIONS.keys()),
                                       state="readonly", width=13, font=("Microsoft YaHei", 10))
        self.sort_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.sort_combo.bind("<<ComboboxSelected>>", lambda e: self._resort())

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # Star button
        self.star_btn = ttk.Button(bar, text="☆ 星标", command=self._toggle_star_current, width=9)
        self.star_btn.pack(side=tk.LEFT, padx=(8, 4))

        # Action buttons (right side)
        ttk.Button(bar, text="导出选中", command=self._export_current, width=9).pack(
            side=tk.RIGHT, padx=(4, 0))
        ttk.Button(bar, text="统计", command=self._show_stats, width=5).pack(
            side=tk.RIGHT, padx=(4, 0))
        ttk.Button(bar, text="刷新", command=self._load_sessions, width=5).pack(
            side=tk.RIGHT, padx=(4, 0))

    def _build_main_area(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # ── Left: session list ───────────────────────────────
        left = ttk.Frame(paned)
        paned.add(left, weight=3)

        self.tree = ttk.Treeview(left, columns=("markers", "date", "msgs", "dur"),
                                 show="headings", selectmode="browse")
        self.tree.heading("markers", text="★/📝", anchor=tk.CENTER)
        self.tree.heading("date", text="日期", anchor=tk.W,
                          command=lambda: self._sort_by_column("date"))
        self.tree.heading("msgs", text="轮数", anchor=tk.CENTER,
                          command=lambda: self._sort_by_column("msgs"))
        self.tree.heading("dur", text="时长", anchor=tk.CENTER,
                          command=lambda: self._sort_by_column("dur"))
        self.tree.column("markers", width=40, anchor=tk.CENTER, stretch=False)
        self.tree.column("date", width=90, minwidth=80)
        self.tree.column("msgs", width=50, anchor=tk.CENTER)
        self.tree.column("dur", width=55, anchor=tk.CENTER)
        self.tree.column("#0", width=400, minwidth=200)

        scroll_v = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_v.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_v.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda e: self._show_full())

        # ── Right: detail + notes ────────────────────────────
        right = ttk.Frame(paned)
        paned.add(right, weight=5)

        right_pane = ttk.PanedWindow(right, orient=tk.VERTICAL)
        right_pane.pack(fill=tk.BOTH, expand=True)

        # Top: conversation preview
        top_frame = ttk.Frame(right_pane)
        right_pane.add(top_frame, weight=3)

        self.detail_text = tk.Text(top_frame, wrap=tk.WORD, font=("Microsoft YaHei", 10),
                                   padx=12, pady=10, state=tk.DISABLED,
                                   bg="#fafafa", fg="#222222",
                                   selectbackground="#0078d4")
        detail_scroll = ttk.Scrollbar(top_frame, orient=tk.VERTICAL,
                                      command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom: notes editor
        notes_frame = ttk.LabelFrame(right_pane, text=" 备注 ",
                                     padding=(4, 4))
        right_pane.add(notes_frame, weight=1)

        notes_inner = ttk.Frame(notes_frame)
        notes_inner.pack(fill=tk.BOTH, expand=True)

        self.notes_text = tk.Text(notes_inner, wrap=tk.WORD,
                                  font=("Microsoft YaHei", 10),
                                  padx=10, pady=8,
                                  bg="#fffef5", fg="#333333",
                                  insertbackground="#0078d4",
                                  undo=True, maxundo=50,
                                  height=4)
        notes_scroll = ttk.Scrollbar(notes_inner, orient=tk.VERTICAL,
                                     command=self.notes_text.yview)
        self.notes_text.configure(yscrollcommand=notes_scroll.set)
        self.notes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        notes_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.notes_text.bind("<FocusIn>", self._on_notes_focus_in)
        self.notes_text.bind("<FocusOut>", self._on_notes_focus_out)
        # Use KeyRelease for reliable save-on-type
        self.notes_text.bind("<KeyRelease>", self._on_notes_key)
        # Prevent tab from leaving the widget
        self.notes_text.bind("<Tab>",
            lambda e: self.notes_text.insert(tk.INSERT, "    ") or "break")

        self._notes_placeholder = "在此添加备注… (Ctrl+N 聚焦)"

    def _build_statusbar(self):
        self.status = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W,
                                padding=(8, 2), font=("Microsoft YaHei", 9))
        self.status.pack(fill=tk.X)

    # ── Data loading ────────────────────────────────────────

    def _load_sessions(self):
        selected_uuid = self.current_uuid
        self.tree.delete(*self.tree.get_children())
        self.sessions = []
        self.session_meta = {}

        for f in sorted(SESSIONS_DIR.glob("*.jsonl")):
            uuid = f.stem
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    events = [json.loads(line) for line in fh if line.strip()]
                if events:
                    self.sessions.append((uuid, events))
                    meta = self._extract_meta(uuid, events)
                    self.session_meta[uuid] = meta
            except Exception:
                continue

        self._populate_tree()

        count = len(self.sessions)
        sc = len(self.starred)
        nc = len([n for n in self.notes.values() if n.strip()])
        parts = [f"已加载 {count} 次会话"]
        if sc:
            parts.append(f"{sc} ★")
        if nc:
            parts.append(f"{nc} 📝")
        self.status.config(text="，".join(parts))

        # Restore selection or select first
        if selected_uuid and selected_uuid in self.session_meta:
            try:
                self.tree.selection_set(selected_uuid)
                self.tree.see(selected_uuid)
                self._on_select()
                return
            except Exception:
                pass

        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self._on_select()

    def _markers_str(self, uuid):
        """Return marker icons combining star and notes status."""
        parts = []
        if uuid in self.starred:
            parts.append("★")
        if uuid in self.notes and self.notes[uuid].strip():
            parts.append("📝")
        return " ".join(parts) if parts else ""

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())

        sort_key, reverse = SORT_OPTIONS.get(self.sort_var.get(),
                                             ("first_ts", True))

        items = []
        for uuid, _ in self.sessions:
            meta = self.session_meta[uuid]
            v = meta.get(sort_key, 0)
            if sort_key == "starred":
                v = (1 if uuid in self.starred else 0, meta.get("first_ts", ""))
                reverse = True
            elif sort_key == "has_notes":
                has = 1 if (uuid in self.notes and self.notes[uuid].strip()) else 0
                v = (has, meta.get("first_ts", ""))
                reverse = True
            items.append((v, uuid))

        items.sort(key=lambda x: x[0], reverse=reverse)

        query = self.search_var.get().strip().lower()
        for _, uuid in items:
            meta = self.session_meta[uuid]
            if query:
                note_text = self.notes.get(uuid, "")
                if query not in meta["first_message"].lower() and \
                   query not in meta.get("cwd", "").lower() and \
                   query not in note_text.lower():
                    continue

            title = meta["first_message"][:80] or "(无文本)"
            display_title = title.replace("\n", " ")
            self.tree.insert("", tk.END, iid=uuid, text=display_title,
                             values=(self._markers_str(uuid), meta["date"],
                                     meta["message_count"], meta["duration"]))

    def _resort(self):
        self._populate_tree()
        q = self.search_var.get().strip().lower()
        c = len(self.tree.get_children())
        t = len(self.sessions)
        self.status.config(text=f"搜索 '{q}' — {c}/{t}" if q else
                           f"排序: {self.sort_var.get()}  |  共 {c} 次会话")

    def _sort_by_column(self, col):
        mapping = {"date": "日期最新 ↓", "msgs": "消息最多 ↓", "dur": "时长最长 ↓"}
        target = mapping.get(col)
        if target:
            cur = self.sort_var.get()
            if cur == target:
                self.sort_var.set(target.replace(" ↓", " ↑"))
            else:
                self.sort_var.set(target)
            self._resort()

    def _extract_meta(self, uuid, events):
        first_ts = None
        last_ts = None
        first_user_msg = ""
        cwd = ""
        model = ""
        message_count = 0
        files_touched = set()

        for ev in events:
            ts = ev.get("timestamp", "")
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            if ev.get("type") == "user":
                msg = ev.get("message", {}).get("content", [])
                if isinstance(msg, list):
                    for block in msg:
                        text = block.get("text", "") if isinstance(block, dict) else str(block)
                        if text and not text.startswith("<ide_"):
                            if not first_user_msg:
                                first_user_msg = text
                cwd = ev.get("cwd", cwd)
            if ev.get("type") == "assistant":
                message_count += 1
                m = ev.get("model", "") or ev.get("message", {}).get("model", "")
                if m:
                    model = m
            if ev.get("type") == "file-history-snapshot":
                snap = ev.get("snapshot", {})
                for fp in snap.get("trackedFileBackups", {}):
                    files_touched.add(fp)

        duration_str, duration_min = self._compute_duration(first_ts, last_ts)
        return {
            "uuid": uuid,
            "date": first_ts[:10] if first_ts else "?",
            "time": first_ts[11:19] if first_ts else "",
            "first_message": first_user_msg,
            "cwd": cwd,
            "model": model,
            "message_count": message_count,
            "duration": duration_str,
            "duration_min": duration_min,
            "first_ts": first_ts or "",
            "last_ts": last_ts or "",
            "files_touched": sorted(files_touched),
        }

    @staticmethod
    def _compute_duration(start, end):
        dur_str, dur_min = "?", 0
        if start and end:
            try:
                s = datetime.fromisoformat(start.replace("Z", "+00:00"))
                e = datetime.fromisoformat(end.replace("Z", "+00:00"))
                mins = (e - s).total_seconds() / 60
                dur_min = round(mins, 1)
                if mins < 1:
                    dur_str = "<1m"
                elif mins < 60:
                    dur_str = f"{int(mins)}m"
                else:
                    dur_str = f"{mins/60:.1f}h"
            except Exception:
                pass
        return dur_str, dur_min

    # ── Star toggle ─────────────────────────────────────────

    def _toggle_star_current(self):
        uuid = self.current_uuid
        if not uuid:
            return
        if uuid in self.starred:
            self.starred.discard(uuid)
        else:
            self.starred.add(uuid)
        self._save_stars()
        # Update tree row marker
        try:
            self.tree.set(uuid, "markers", self._markers_str(uuid))
        except Exception:
            pass
        self._update_star_btn()
        self.status.config(
            text=f"{'★ 已星标' if uuid in self.starred else '☆ 已取消星标'} — {uuid[:8]}")
        # Refresh preview header
        self._render_preview(uuid)

    def _update_star_btn(self):
        uuid = self.current_uuid
        if uuid and uuid in self.starred:
            self.star_btn.config(text="★ 取消星标")
        else:
            self.star_btn.config(text="☆ 星标")

    # ── Notes ────────────────────────────────────────────────

    def _load_notes_for(self, uuid):
        """Load notes into the editor (programmatic — suppress events)."""
        self._notes_suppress = True
        self.notes_text.delete(1.0, tk.END)
        note = self.notes.get(uuid, "")
        if note:
            self.notes_text.insert(1.0, note)
            self.notes_text.config(fg="#333333")
        else:
            self.notes_text.insert(1.0, self._notes_placeholder)
            self.notes_text.config(fg="#aaaaaa")
        self.notes_text.edit_modified = False
        self._notes_suppress = False

    def _on_notes_key(self, event=None):
        """Handle key release — save content for real edits."""
        if self._notes_suppress:
            return
        # Skip modifier-only keys
        if event and event.keysym in (
            "Shift_L", "Shift_R", "Control_L", "Control_R",
            "Alt_L", "Alt_R", "Caps_Lock", "Num_Lock",
            "Left", "Right", "Up", "Down", "Home", "End",
            "Page_Up", "Page_Down", "Escape"
        ):
            return
        self._commit_notes()

    def _commit_notes(self):
        """Read notes text and persist if changed."""
        uuid = self.current_uuid
        if not uuid:
            return
        text = self.notes_text.get(1.0, tk.END).strip()
        placeholder = self._notes_placeholder.strip()

        # Ignore placeholder text
        if not text or text == placeholder:
            if uuid in self.notes:
                del self.notes[uuid]
                self._notes_dirty = True
                self._schedule_save_notes()
            return

        # Check if actually different from stored
        current = self.notes.get(uuid, "")
        if text == current:
            return

        self.notes[uuid] = text
        self._notes_dirty = True
        self.notes_text.config(fg="#333333")
        self._schedule_save_notes()

    def _schedule_save_notes(self):
        """Debounced save: persist 1.5s after last edit."""
        if self._notes_save_job:
            self.root.after_cancel(self._notes_save_job)
        self._notes_save_job = self.root.after(1500, self._do_auto_save)

    def _do_auto_save(self):
        self._notes_save_job = None
        if self._notes_dirty:
            self._save_notes()
            uuid = self.current_uuid
            if uuid:
                try:
                    self.tree.set(uuid, "markers", self._markers_str(uuid))
                except Exception:
                    pass

    def _on_close(self):
        if self._notes_dirty:
            self._save_notes()
        self.root.destroy()

    def _on_notes_focus_in(self, event=None):
        """Clear placeholder when user clicks into notes."""
        if self._notes_suppress:
            return
        text = self.notes_text.get(1.0, tk.END).strip()
        if text == self._notes_placeholder.strip():
            self._notes_suppress = True
            self.notes_text.delete(1.0, tk.END)
            self.notes_text.config(fg="#333333")
            self.notes_text.edit_modified = False
            self._notes_suppress = False

    def _on_notes_focus_out(self, event=None):
        """Show placeholder if notes is empty when focus leaves."""
        if self._notes_suppress:
            return
        self._commit_notes()  # Ensure latest edit is saved
        text = self.notes_text.get(1.0, tk.END).strip()
        if not text or text == self._notes_placeholder.strip():
            self._notes_suppress = True
            self.notes_text.delete(1.0, tk.END)
            self.notes_text.insert(1.0, self._notes_placeholder)
            self.notes_text.config(fg="#aaaaaa")
            self.notes_text.edit_modified = False
            self._notes_suppress = False

    def _focus_notes(self):
        """Keyboard shortcut: focus notes, clear placeholder."""
        uuid = self.current_uuid
        if not uuid:
            return "break"
        text = self.notes_text.get(1.0, tk.END).strip()
        if not text or text == self._notes_placeholder.strip():
            self._notes_suppress = True
            self.notes_text.delete(1.0, tk.END)
            self.notes_text.config(fg="#333333")
            self.notes_text.edit_modified = False
            self._notes_suppress = False
        self.notes_text.focus_set()
        return "break"

    # ── Event handlers ──────────────────────────────────────

    def _do_search(self):
        self._populate_tree()
        q = self.search_var.get().strip().lower()
        c = len(self.tree.get_children())
        t = len(self.sessions)
        self.status.config(
            text=f"显示全部 {t} 次会话" if not q else f"搜索 '{q}' — 匹配 {c}/{t} 次会话")

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        uuid = sel[0]

        # Save notes before switching
        if self.current_uuid and self.current_uuid != uuid:
            self._commit_notes()
            if self._notes_dirty:
                self._do_auto_save()

        self.current_uuid = uuid
        self._update_star_btn()
        self._render_preview(uuid)
        self._load_notes_for(uuid)

    def _render_preview(self, uuid):
        """Render the top preview panel."""
        meta = self.session_meta.get(uuid, {})
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)

        markers = self._markers_str(uuid)
        if markers:
            markers += " "

        self.detail_text.insert(tk.END, f"{markers}会话概览\n", "h1")
        self.detail_text.insert(tk.END, f"UUID:  {uuid}\n", "dim")
        self.detail_text.insert(tk.END,
            f"日期:  {meta.get('date', '?')}  {meta.get('time', '')}"
            f"    时长:  {meta.get('duration', '?')}"
            f"    轮数:  {meta.get('message_count', 0)}\n")
        self.detail_text.insert(tk.END, f"模型:  {meta.get('model', '?')}\n")
        self.detail_text.insert(tk.END, f"目录:  {meta.get('cwd', '?')}\n")

        files = meta.get("files_touched", [])
        if files:
            self.detail_text.insert(tk.END,
                f"文件:  {', '.join(files[:6])}\n")
        self.detail_text.insert(tk.END, "\n")

        events = None
        for u, evts in self.sessions:
            if u == uuid:
                events = evts
                break

        if events:
            self.detail_text.insert(tk.END, "对话摘要\n", "h2")
            for ev in events:
                if ev.get("type") == "user":
                    msg = ev.get("message", {}).get("content", [])
                    for block in msg if isinstance(msg, list) else []:
                        text = block.get("text", "") if isinstance(block, dict) else ""
                        if text and len(text) > 5 and not text.startswith("<ide_"):
                            ts = ev.get("timestamp", "")[11:19]
                            short = text[:150].replace("\n", " ")
                            if len(text) > 150:
                                short += "…"
                            self.detail_text.insert(tk.END, f"\n[{ts}] 你:\n", "user_label")
                            self.detail_text.insert(tk.END, f"{short}\n", "user_text")
                elif ev.get("type") == "assistant":
                    content = ev.get("message", {}).get("content", [])
                    for block in content if isinstance(content, list) else []:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text and len(text) > 30:
                                ts = ev.get("timestamp", "")[11:19]
                                short = text[:200].replace("\n", " ")
                                if len(text) > 200:
                                    short += "…"
                                self.detail_text.insert(tk.END, f"\n[{ts}] Claude:\n", "ai_label")
                                self.detail_text.insert(tk.END, f"{short}\n", "ai_text")

        self.detail_text.config(state=tk.DISABLED)

    def _show_full(self):
        uuid = self.current_uuid
        if not uuid:
            return
        events = None
        for u, evts in self.sessions:
            if u == uuid:
                events = evts
                break
        if not events:
            return

        win = tk.Toplevel(self.root)
        win.title(f"完整对话 — {uuid[:8]}")
        win.geometry("900x650")

        text = tk.Text(win, wrap=tk.WORD, font=("Microsoft YaHei", 10), padx=12, pady=10)
        scroll = ttk.Scrollbar(win, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for ev in events:
            if ev.get("type") == "user":
                msg = ev.get("message", {}).get("content", [])
                for block in msg if isinstance(msg, list) else []:
                    txt = block.get("text", "") if isinstance(block, dict) else ""
                    if txt:
                        ts = ev.get("timestamp", "")[:19]
                        text.insert(tk.END, f"\n{'─'*60}\n[用户] {ts}\n", "label")
                        text.insert(tk.END, txt + "\n")
            elif ev.get("type") == "assistant":
                content = ev.get("message", {}).get("content", [])
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        ts = ev.get("timestamp", "")[:19]
                        body = block.get("text", "")
                        text.insert(tk.END, f"\n[Claude] {ts}\n", "label")
                        text.insert(tk.END, body + "\n")

        # Append notes
        note = self.notes.get(uuid, "").strip()
        if note:
            text.insert(tk.END, f"\n{'='*60}\n📝 备注:\n", "notes_label")
            text.insert(tk.END, note + "\n")

        text.config(state=tk.DISABLED)
        text.tag_config("label", foreground="#888888", font=("Microsoft YaHei", 9))
        text.tag_config("notes_label", foreground="#d4a800",
                        font=("Microsoft YaHei", 10, "bold"))

    def _export_current(self):
        uuid = self.current_uuid
        if not uuid:
            messagebox.showinfo("提示", "请先选中一个会话")
            return
        events = None
        for u, evts in self.sessions:
            if u == uuid:
                events = evts
                break
        meta = self.session_meta.get(uuid, {})
        date = meta.get("date", "unknown")
        out_path = SESSIONS_DIR / f"{date}_{uuid[:8]}.md"

        lines = [f"# 会话: {uuid[:8]}", "",
                 f"- 日期: {meta.get('date', '?')} {meta.get('time', '')}",
                 f"- 时长: {meta.get('duration', '?')}",
                 f"- 模型: {meta.get('model', '?')}",
                 f"- 目录: {meta.get('cwd', '?')}",
                 f"- 星标: {'是' if uuid in self.starred else '否'}",
                 f"- 备注: {'是' if uuid in self.notes and self.notes[uuid].strip() else '否'}",
                 "", "---", ""]

        note = self.notes.get(uuid, "").strip()
        if note:
            lines.append("> 📝 备注")
            lines.append("> " + note.replace("\n", "\n> "))
            lines.append("")

        for ev in events:
            if ev.get("type") == "user":
                msg = ev.get("message", {}).get("content", [])
                for block in msg if isinstance(msg, list) else []:
                    txt = block.get("text", "") if isinstance(block, dict) else ""
                    if txt:
                        lines.append(f"## 用户 ({ev.get('timestamp', '')[:19]})")
                        lines.append("")
                        lines.append(txt)
                        lines.append("")
            elif ev.get("type") == "assistant":
                content = ev.get("message", {}).get("content", [])
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        lines.append(f"### Claude ({ev.get('timestamp', '')[:19]})")
                        lines.append("")
                        lines.append(block.get("text", ""))
                        lines.append("")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.status.config(text=f"已导出: {out_path}")
        messagebox.showinfo("导出成功", f"已保存到:\n{out_path}")

    def _show_stats(self):
        if not self.sessions:
            return
        total_msgs = sum(m["message_count"] for m in self.session_meta.values())
        dates = defaultdict(int)
        models = defaultdict(int)
        for m in self.session_meta.values():
            if m["date"] != "?":
                dates[m["date"]] += 1
            if m["model"]:
                models[m["model"]] += 1

        win = tk.Toplevel(self.root)
        win.title("使用统计")
        win.geometry("550x500")

        text = tk.Text(win, wrap=tk.WORD, font=("Microsoft YaHei", 11), padx=16, pady=12)
        text.pack(fill=tk.BOTH, expand=True)

        note_count = len([n for n in self.notes.values() if n.strip()])
        text.insert(tk.END, "Claude Code 使用统计\n", "h")
        text.insert(tk.END, f"\n总会话数:      {len(self.sessions)}\n")
        text.insert(tk.END, f"星标会话:      {len(self.starred)}\n")
        text.insert(tk.END, f"有备注会话:    {note_count}\n")
        text.insert(tk.END, f"总消息数:      {total_msgs}\n")
        text.insert(tk.END, f"日期范围:      {min(dates.keys())} ~ {max(dates.keys())}\n")

        text.insert(tk.END, f"\n模型使用:\n", "h")
        for m, c in sorted(models.items(), key=lambda x: -x[1]):
            bar = "█" * min(c, 50)
            text.insert(tk.END, f"  {m}:  {bar} {c}\n")

        text.insert(tk.END, f"\n最活跃日期:\n", "h")
        for d, c in sorted(dates.items(), key=lambda x: -x[1])[:12]:
            bar = "█" * c
            text.insert(tk.END, f"  {d}:  {bar} {c}\n")

        text.config(state=tk.DISABLED)
        text.tag_config("h", font=("Microsoft YaHei", 12, "bold"), foreground="#0078d4")

    # ── Text tags ───────────────────────────────────────────

    def _configure_tags(self):
        t = self.detail_text
        t.tag_config("h1", font=("Microsoft YaHei", 14, "bold"),
                     foreground="#0078d4", spacing3=6)
        t.tag_config("h2", font=("Microsoft YaHei", 11, "bold"),
                     foreground="#333333", spacing3=4)
        t.tag_config("dim", font=("Consolas", 9), foreground="#888888")
        t.tag_config("user_label", font=("Microsoft YaHei", 9, "bold"),
                     foreground="#d4730a")
        t.tag_config("user_text", font=("Microsoft YaHei", 9),
                     foreground="#5a3a00", lmargin1=10, lmargin2=10)
        t.tag_config("ai_label", font=("Microsoft YaHei", 9, "bold"),
                     foreground="#0078d4")
        t.tag_config("ai_text", font=("Microsoft YaHei", 9),
                     foreground="#1a3a5c", lmargin1=10, lmargin2=10)
        t.tag_config("starred_tag", font=("Microsoft YaHei", 10, "bold"),
                     foreground="#d4a800")


def main():
    root = tk.Tk()
    app = SessionViewer(root)
    app._configure_tags()
    root.mainloop()


if __name__ == "__main__":
    main()
