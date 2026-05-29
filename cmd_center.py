#!/usr/bin/env python3
"""
Hermes Command Center — real-time dashboard with live chat.
"""

import asyncio
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/opt/data"))
DB_PATH = HERMES_HOME / "state.db"
HERMES_BIN = "/opt/hermes/.venv/bin/hermes"

app = FastAPI(title="Hermes Command Center")


# ── WebSocket Hub ────────────────────────────────────────────────────────────
class Hub:
    def __init__(self):
        self.clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.remove(ws)


hub = Hub()


# ── DB Helpers ───────────────────────────────────────────────────────────────
def _row_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


async def query(sql: str, params: tuple = ()) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = _row_factory
        async with db.execute(sql, params) as cur:
            return await cur.fetchall()


async def query_one(sql: str, params: tuple = ()) -> dict | None:
    rows = await query(sql, params)
    return rows[0] if rows else None


# ── API: Stats ───────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def get_stats():
    row = await query_one("""
        SELECT
            COUNT(*) as total_sessions,
            SUM(CASE WHEN ended_at IS NULL THEN 1 ELSE 0 END) as active_sessions,
            SUM(message_count) as total_messages,
            SUM(tool_call_count) as total_tool_calls,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(cache_read_tokens) as total_cache_tokens,
            SUM(reasoning_tokens) as total_reasoning_tokens,
            SUM(estimated_cost_usd) as total_cost,
            SUM(api_call_count) as total_api_calls
        FROM sessions
    """)
    return {"stats": row}


# ── API: Sessions ────────────────────────────────────────────────────────────
@app.get("/api/sessions")
async def get_sessions(limit: int = 50):
    rows = await query(
        """SELECT id, source, title, model, started_at, ended_at,
                  message_count, tool_call_count,
                  input_tokens, output_tokens, cache_read_tokens,
                  reasoning_tokens, estimated_cost_usd, api_call_count,
                  parent_session_id
           FROM sessions ORDER BY started_at DESC LIMIT ?""",
        (limit,),
    )
    now = time.time()
    for r in rows:
        r["is_active"] = r["ended_at"] is None and (now - (r["started_at"] or 0)) < 600
        r["started_human"] = (
            datetime.fromtimestamp(r["started_at"], tz=timezone.utc).strftime("%H:%M · %b %d")
            if r["started_at"]
            else ""
        )
        r["total_tokens"] = (r["input_tokens"] or 0) + (r["output_tokens"] or 0)
    return {"sessions": rows}


# ── API: Session Messages ────────────────────────────────────────────────────
@app.get("/api/sessions/{sid}/messages")
async def get_messages(sid: str, limit: int = 80):
    rows = await query(
        "SELECT role, content, tool_name, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
        (sid, limit),
    )
    for r in rows:
        r["time"] = (
            datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
            if r["timestamp"]
            else ""
        )
        if r.get("content") and len(r["content"]) > 1500:
            r["content"] = r["content"][:1500] + "…"
    return {"messages": rows}


# ── API: Profiles ────────────────────────────────────────────────────────────
@app.get("/api/profiles")
async def get_profiles():
    try:
        r = subprocess.run(
            [HERMES_BIN, "profile", "list"],
            capture_output=True, text=True, timeout=10,
        )
        lines = r.stdout.strip().split("\n")[2:]  # skip header
        profiles = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].lstrip("◆").strip()
                model = parts[1] if len(parts) > 1 else "—"
                status = "running" if "running" in line else "stopped"
                profiles.append({"name": name, "model": model, "status": status})
        return {"profiles": profiles}
    except Exception as e:
        return {"profiles": [], "error": str(e)}


# ── API: Costs by Model ──────────────────────────────────────────────────────
@app.get("/api/costs")
async def get_costs(days: int = 7):
    cutoff = time.time() - days * 86400
    rows = await query(
        """SELECT model, COUNT(*) as sessions,
                  SUM(input_tokens) as in_tok, SUM(output_tokens) as out_tok,
                  SUM(estimated_cost_usd) as cost
           FROM sessions WHERE started_at > ? GROUP BY model ORDER BY cost DESC""",
        (cutoff,),
    )
    return {"models": rows}


# ── API: Daily Activity ──────────────────────────────────────────────────────
@app.get("/api/activity")
async def get_activity(days: int = 7):
    cutoff = time.time() - days * 86400
    rows = await query(
        """SELECT date(started_at, 'unixepoch') as day,
                  COUNT(*) as sessions, SUM(message_count) as msgs,
                  SUM(input_tokens + output_tokens) as tokens,
                  SUM(estimated_cost_usd) as cost
           FROM sessions WHERE started_at > ? GROUP BY day ORDER BY day""",
        (cutoff,),
    )
    return {"daily": rows}


# ── WebSocket: Chat + Live Updates ───────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "chat":
                text = msg.get("text", "").strip()
                if not text:
                    continue
                profile = msg.get("profile", "")
                response = await chat_with_hermes(text, profile)
                await ws.send_json({"type": "chat_response", "text": response})

            elif msg.get("type") == "refresh":
                stats = await get_stats()
                sessions = await get_sessions()
                await ws.send_json({"type": "full_update", "stats": stats["stats"], "sessions": sessions["sessions"]})

    except WebSocketDisconnect:
        hub.disconnect(ws)


async def chat_with_hermes(text: str, profile: str = "") -> str:
    cmd = [HERMES_BIN, "chat", "-q", text, "-Q"]
    if profile:
        cmd = [HERMES_BIN, "-p", profile, "chat", "-q", text, "-Q"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "HERMES_HOME": str(HERMES_HOME)},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        return stdout.decode().strip() or stderr.decode().strip() or "(no response)"
    except asyncio.TimeoutError:
        return "(timeout — agent took too long)"
    except Exception as e:
        return f"(error: {e})"


# ── Background Poller ────────────────────────────────────────────────────────
async def poller():
    while True:
        await asyncio.sleep(4)
        if hub.clients:
            try:
                stats = await get_stats()
                sessions = await get_sessions(limit=30)
                await hub.broadcast({
                    "type": "live_update",
                    "stats": stats["stats"],
                    "sessions": sessions["sessions"],
                    "ts": time.time(),
                })
            except Exception:
                pass


@app.on_event("startup")
async def startup():
    asyncio.create_task(poller())


# ── Serve Frontend ───────────────────────────────────────────────────────────
@app.get("/")
async def index():
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard.html not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9900, log_level="warning")
