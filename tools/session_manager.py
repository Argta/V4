#!/usr/bin/env python
"""Claude Code session organizer — browse, search, and archive conversations.

Usage:
  python session_manager.py list              # Show all sessions with titles
  python session_manager.py list -n 20        # Last 20 sessions
  python session_manager.py show <uuid>       # Print full conversation
  python session_manager.py show <uuid> -s    # Summary only (key decisions)
  python session_manager.py search "ITD"      # Search across all sessions
  python session_manager.py export <uuid>     # Export as readable markdown
  python session_manager.py stats             # Usage statistics
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SESSIONS_DIR = Path.home() / ".claude" / "projects" / "C--Users-LENOVO"


def load_sessions():
    """Load all session files and return sorted list of (uuid, events)."""
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.jsonl")):
        uuid = f.stem
        try:
            with open(f, "r", encoding="utf-8") as fh:
                events = [json.loads(line) for line in fh if line.strip()]
            if events:
                sessions.append((uuid, events))
        except (json.JSONDecodeError, OSError):
            continue
    return sessions


def extract_meta(uuid, events):
    """Extract metadata from a session."""
    first_ts = None
    last_ts = None
    first_user_msg = ""
    cwd = ""
    model = ""
    files_touched = set()
    message_count = 0

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
                            first_user_msg = text[:120]
            cwd = ev.get("cwd", cwd)

        if ev.get("type") == "assistant":
            message_count += 1
            model = ev.get("model", model) or ev.get("message", {}).get("model", "") or model

        if ev.get("type") == "file-history-snapshot":
            snap = ev.get("snapshot", {})
            for fpath in snap.get("trackedFileBackups", {}):
                files_touched.add(fpath)

    return {
        "uuid": uuid,
        "date": first_ts[:10] if first_ts else "unknown",
        "time": first_ts[11:19] if first_ts else "",
        "first_message": first_user_msg,
        "cwd": cwd,
        "model": model,
        "files_touched": sorted(files_touched),
        "message_count": message_count,
        "duration": _duration_str(first_ts, last_ts),
    }


def _duration_str(start, end):
    if not start or not end:
        return "?"
    try:
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        mins = (e - s).total_seconds() / 60
        if mins < 1:
            return "<1m"
        if mins < 60:
            return f"{int(mins)}m"
        return f"{mins/60:.1f}h"
    except Exception:
        return "?"


def extract_summary(events):
    """Extract a structured summary: key decisions, file changes, conclusions."""
    user_messages = []
    assistant_decisions = []
    files_modified = set()
    tool_calls = defaultdict(int)

    for ev in events:
        if ev.get("type") == "user":
            msg = ev.get("message", {}).get("content", [])
            for block in msg if isinstance(msg, list) else []:
                text = block.get("text", "") if isinstance(block, dict) else ""
                if text and len(text) > 10 and not text.startswith("<ide_"):
                    user_messages.append(text[:200])

        if ev.get("type") == "assistant":
            content = ev.get("message", {}).get("content", [])
            for block in content if isinstance(content, list) else []:
                text = block.get("text", "") if isinstance(block, dict) else ""
                if text and len(text) > 50:
                    # Look for decision/conclusion keywords
                    for kw in [
                        "done", "fixed", "resolved", "completed", "implemented",
                        "created", "modified", "changed", "added", "removed",
                        "结论", "完成", "修复", "实现", "创建", "修改"
                    ]:
                        if kw in text.lower()[:200]:
                            assistant_decisions.append(text[:200])
                            break

        if ev.get("type") == "assistant":
            for tb in ev.get("message", {}).get("content", []) if isinstance(ev.get("message", {}).get("content"), list) else []:
                if isinstance(tb, dict) and tb.get("type") == "tool_use":
                    name = tb.get("name", "unknown")
                    if name in ("Edit", "Write"):
                        inp = tb.get("input", {})
                        fpath = inp.get("file_path", "")
                        if fpath:
                            files_modified.add(fpath)
                    tool_calls[name] += 1

    return {
        "user_requests": user_messages,
        "key_outputs": assistant_decisions[:5],
        "files_modified": sorted(files_modified),
        "tools_used": dict(tool_calls),
    }


def cmd_list(sessions, n=50):
    """Print a table of recent sessions."""
    print(f"{'#':>3}  {'Date':>10}  {'Dur':>5}  {'Msgs':>4}  {'Model':<18}  {'First message'}")
    print("-" * 120)
    for i, (uuid, events) in enumerate(sessions[-n:], 1):
        m = extract_meta(uuid, events)
        msg = m["first_message"][:70].replace("\n", " ")
        model_short = m["model"].split("/")[-1][:17] if m["model"] else "?"
        print(f"{i:>3}  {m['date']}  {m['duration']:>5}  {m['message_count']:>4}  {model_short:<18}  {msg}")


def cmd_show(sessions, uuid_short, summary_only=False):
    """Show a session in detail."""
    matches = [(u, e) for u, e in sessions if u.startswith(uuid_short)]
    if not matches:
        print(f"No session matching '{uuid_short}'")
        return

    uuid, events = matches[0]
    meta = extract_meta(uuid, events)

    print(f"\n{'='*70}")
    print(f"  Session: {uuid}")
    print(f"  Date:    {meta['date']} {meta['time']}  Duration: {meta['duration']}")
    print(f"  Model:   {meta['model']}")
    print(f"  CWD:     {meta['cwd']}")
    print(f"  Files:   {', '.join(meta['files_touched'][:8]) or 'none'}")
    print(f"{'='*70}\n")

    if summary_only:
        summary = extract_summary(events)
        print("--- USER REQUESTS ---")
        for r in summary["user_requests"]:
            print(f"  * {r}")
        print("\n--- KEY OUTPUTS ---")
        for o in summary["key_outputs"]:
            print(f"  * {o}")
        print(f"\n--- FILES MODIFIED ({len(summary['files_modified'])}) ---")
        for f in summary["files_modified"]:
            print(f"  {f}")
        return

    # Full transcript
    for ev in events:
        if ev.get("type") == "user":
            msg = ev.get("message", {}).get("content", [])
            for block in msg if isinstance(msg, list) else []:
                text = block.get("text", "") if isinstance(block, dict) else ""
                if text:
                    print(f"\n{'─'*60}")
                    print(f"[USER] {ev['timestamp'][:19]}")
                    print(f"{'─'*60}")
                    print(text[:500])

        elif ev.get("type") == "assistant":
            content = ev.get("message", {}).get("content", [])
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        print(f"\n[ASSISTANT] {ev['timestamp'][:19]}")
                        print(block["text"][:500])


def cmd_search(sessions, query):
    """Search across all sessions for a keyword."""
    q = query.lower()
    results = []
    for uuid, events in sessions:
        hits = 0
        snippet = ""
        for ev in events:
            text = json.dumps(ev, ensure_ascii=False).lower()
            if q in text:
                hits += 1
                if not snippet:
                    # Extract surrounding context
                    idx = text.find(q)
                    snippet = text[max(0, idx - 40):idx + len(q) + 40]
        if hits > 0:
            meta = extract_meta(uuid, events)
            meta["hits"] = hits
            meta["snippet"] = snippet
            results.append(meta)

    print(f"\nSearch: '{query}' — {len(results)} session(s) matched\n")
    for r in results:
        print(f"  [{r['date']}] {r['uuid'][:8]}... ({r['hits']} hits) — {r['first_message'][:60]}")


def cmd_export(sessions, uuid_short):
    """Export a session as markdown."""
    matches = [(u, e) for u, e in sessions if u.startswith(uuid_short)]
    if not matches:
        print(f"No session matching '{uuid_short}'")
        return

    uuid, events = matches[0]
    meta = extract_meta(uuid, events)
    out_path = SESSIONS_DIR / f"{meta['date']}_{uuid[:8]}.md"

    lines = [
        f"# Session: {uuid[:8]}...",
        f"",
        f"- **Date:** {meta['date']} {meta['time']}",
        f"- **Duration:** {meta['duration']}",
        f"- **Model:** {meta['model']}",
        f"- **Working Dir:** {meta['cwd']}",
        f"- **Topic:** {meta['first_message']}",
        f"",
        f"---",
        f"",
    ]

    for ev in events:
        if ev.get("type") == "user":
            msg = ev.get("message", {}).get("content", [])
            for block in msg if isinstance(msg, list) else []:
                text = block.get("text", "") if isinstance(block, dict) else ""
                if text:
                    lines.append(f"## 🧑 User ({ev['timestamp'][:19]})")
                    lines.append("")
                    lines.append(text)
                    lines.append("")

        elif ev.get("type") == "assistant":
            content = ev.get("message", {}).get("content", [])
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "text":
                    lines.append(f"### 🤖 Assistant ({ev['timestamp'][:19]})")
                    lines.append("")
                    lines.append(block["text"])
                    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Exported to: {out_path}")


def cmd_stats(sessions):
    """Print usage statistics."""
    if not sessions:
        print("No sessions found.")
        return

    total_msgs = 0
    models = defaultdict(int)
    dates = defaultdict(int)
    files_touched = defaultdict(int)
    total_duration_min = 0

    for uuid, events in sessions:
        meta = extract_meta(uuid, events)
        total_msgs += meta["message_count"]
        models[meta["model"]] += 1
        if meta["date"] != "unknown":
            dates[meta["date"]] += 1
        for f in meta["files_touched"]:
            files_touched[f] += 1

    print(f"\n{'='*50}")
    print(f"  Claude Code Session Statistics")
    print(f"{'='*50}")
    print(f"  Total sessions:      {len(sessions)}")
    print(f"  Total messages:      {total_msgs}")
    print(f"  Date range:          {min(dates.keys())} ~ {max(dates.keys())}")
    print(f"\n  Models used:")
    for m, c in models.items():
        print(f"    {m}: {c} sessions")
    print(f"\n  Most active days:")
    for d, c in sorted(dates.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(c, 40)
        print(f"    {d}: {bar} {c}")
    print(f"\n  Most-touched files:")
    for f, c in sorted(files_touched.items(), key=lambda x: -x[1])[:10]:
        print(f"    [{c}x] {f}")


def interactive(sessions):
    """Interactive loop — type commands, no need to remember flags."""
    print("\n" + "=" * 55)
    print("  Claude Code Session Manager")
    print("=" * 55)
    print("  Commands: list, show <id>, search <kw>, export <id>, stats, help, quit")
    print("-" * 55)
    cmd_list(sessions, n=10)
    print()

    while True:
        try:
            line = input("session> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "quit" or cmd == "exit" or cmd == "q":
            print("Bye.")
            break
        elif cmd == "list" or cmd == "ls":
            n = 50
            if arg:
                try:
                    n = int(arg)
                except ValueError:
                    pass
            cmd_list(sessions, n)
        elif cmd == "show" or cmd == "s":
            if not arg:
                print("  Usage: show <id_prefix>  (e.g. show 5b6614fc)")
                continue
            summary = "-s" in arg
            uuid = arg.replace(" -s", "").strip()
            cmd_show(sessions, uuid, summary)
        elif cmd == "search" or cmd == "find":
            if not arg:
                print("  Usage: search <keyword>")
                continue
            cmd_search(sessions, arg)
        elif cmd == "export" or cmd == "exp":
            if not arg:
                print("  Usage: export <id_prefix>")
                continue
            cmd_export(sessions, arg)
        elif cmd == "stats" or cmd == "stat":
            cmd_stats(sessions)
        elif cmd == "help" or cmd == "h" or cmd == "?":
            print("""
  list [N]        Show recent sessions (default 50, use 'list 10' for 10)
  show <id>       Print full conversation (use first 4+ chars of UUID)
  show <id> -s    Summary only — key decisions and files changed
  search <kw>     Search all sessions for a keyword
  export <id>     Save session as markdown file
  stats           Usage statistics
  quit            Exit
            """)
        else:
            print(f"  Unknown command: {cmd}  (type 'help' for commands)")


def main():
    sessions = load_sessions()

    if not sessions:
        print("No sessions found.")
        return

    if len(sys.argv) < 2:
        interactive(sessions)
        return

    cmd = sys.argv[1]

    if cmd == "list":
        n = 50
        if "-n" in sys.argv:
            try:
                n = int(sys.argv[sys.argv.index("-n") + 1])
            except (ValueError, IndexError):
                pass
        cmd_list(sessions, n)

    elif cmd == "show":
        if len(sys.argv) < 3:
            print("Usage: session_manager.py show <uuid_prefix> [-s]")
            return
        uuid_short = sys.argv[2]
        summary_only = "-s" in sys.argv
        cmd_show(sessions, uuid_short, summary_only)

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: session_manager.py search <keyword>")
            return
        cmd_search(sessions, sys.argv[2])

    elif cmd == "export":
        if len(sys.argv) < 3:
            print("Usage: session_manager.py export <uuid_prefix>")
            return
        cmd_export(sessions, sys.argv[2])

    elif cmd == "stats":
        cmd_stats(sessions)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
