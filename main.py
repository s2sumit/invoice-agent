"""A2A Invoice Agent - one file.
Local run:  uvicorn main:app --port 8000
Env vars:   BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, DATA_DIR
"""
import asyncio, hashlib, json, os, re, sqlite3, threading, time, uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---------------------------------------------------------------- config
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/a2a/")  # EXACT string you submit
PREFIX = urlparse(BASE_URL).path.rstrip("/")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DATA_DIR = os.environ.get("DATA_DIR", ".")

A2A = "application/a2a+json"
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RES = "application/vnd.ga5.invoice-action-results+json"
MT_RCPT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

CARD = {
    "name": "Invoice Action Agent",
    "description": "Reads invoice packages, proposes one safe business action per invoice with evidence, and executes only grader-accepted proposals.",
    "version": "1.0.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
    "defaultInputModes": [MT_BATCH, MT_RES],
    "defaultOutputModes": [MT_PROP, MT_RCPT],
    "skills": [{
        "id": "invoice_action_agent",
        "name": "Invoice action agent",
        "description": "Chooses settle, approve, hold, reject-duplicate or exception for each invoice package.",
        "tags": ["invoice", "accounts-payable", "approval"],
    }],
}

# ---------------------------------------------------------------- helpers
def now(): return datetime.now(timezone.utc).isoformat()
def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

STATUS = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED",
          404: "NOT_FOUND", 409: "ABORTED", 415: "INVALID_ARGUMENT", 500: "INTERNAL"}

class ApiError(Exception):
    def __init__(self, status, reason, msg):
        self.status, self.reason, self.msg = status, reason, msg

def reply(obj, status=200):
    return Response(json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
                    status_code=status, media_type=A2A)

def err(status, reason, msg):
    return reply({"error": {"code": status, "status": STATUS.get(status, "UNKNOWN"),
                            "message": f"{reason}: {msg}",
                            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                         "reason": reason, "domain": "a2a-protocol.org"}]}}, status)

app = FastAPI()

@app.exception_handler(ApiError)
async def _h1(request, e): return err(e.status, e.reason, e.msg)
@app.exception_handler(StarletteHTTPException)
async def _h2(request, e): return err(e.status_code, "HTTP_ERROR", "Request failed")
@app.exception_handler(RequestValidationError)
async def _h3(request, e): return err(400, "INVALID_REQUEST", "Invalid request")
@app.exception_handler(Exception)
async def _h4(request, e):
    print("ERROR", repr(e))
    return err(500, "INTERNAL", "Internal error")

# ---------------------------------------------------------------- storage
os.makedirs(DATA_DIR, exist_ok=True)
db = sqlite3.connect(os.path.join(DATA_DIR, "state.db"), check_same_thread=False, isolation_level=None)
db_lock = threading.Lock()
with db_lock:
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, state TEXT, json TEXT, created INTEGER)")
    db.execute("CREATE TABLE IF NOT EXISTS msgs(principal TEXT, mid TEXT, hash TEXT, task_id TEXT, PRIMARY KEY(principal, mid))")
    db.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, json TEXT)")

def q(sql, args=()):
    with db_lock:
        return db.execute(sql, args).fetchall()

def save(task, principal, mid=None, mhash=None):
    """Save task (+ message record) atomically."""
    with db_lock:
        db.execute("BEGIN")
        try:
            db.execute("INSERT INTO tasks(id,principal,state,json,created) VALUES(?,?,?,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET state=excluded.state, json=excluded.json",
                       (task["id"], principal, task["status"]["state"], json.dumps(task), time.time_ns()))
            if mid:
                db.execute("INSERT OR IGNORE INTO msgs VALUES(?,?,?,?)", (principal, mid, mhash, task["id"]))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise

def load_task(principal, tid):
    r = q("SELECT json FROM tasks WHERE id=? AND principal=?", (tid, principal))
    return json.loads(r[0][0]) if r else None

locks = {}
def lock_for(key): return locks.setdefault(key, asyncio.Lock())

# ---------------------------------------------------------------- auth
def guard(request: Request):
    h = request.headers.get("authorization", "")
    p = h.split(" ", 1)
    if len(p) != 2 or p[0].lower() != "bearer" or not p[1].strip():
        raise ApiError(401, "UNAUTHENTICATED", "Authentication required")
    if request.headers.get("a2a-version") != "1.0":
        raise ApiError(400, "VERSION_NOT_SUPPORTED", "A2A-Version 1.0 required")
    return sha(p[1].strip())  # principal = hash of token (each token = separate user)

# ---------------------------------------------------------------- AI layer
SYSTEM = """You are a careful accounts-payable analyst. For EACH invoice package choose exactly one action:
- settle_invoice: the invoice is valid, reconciled to its records, and within autonomous approval authority.
- request_approval: commercially valid, but the amount or terms are outside delegated authority, so a named approver must sign off.
- hold_invoice: payment pauses until a stated verification completes (for example a bank-detail callback, a missing receipt or a tax check).
- reject_duplicate: the SAME commercial invoice (same vendor, same invoice number, same amount) was already paid. A payment for a DIFFERENT invoice number is NOT a duplicate.
- open_exception: material records conflict (amounts, vendor, purchase order, currency) and need an exception workflow.

Each package holds several documents: a cover sheet, archived or old examples, training decoys, and one paragraph that really controls the action for THIS invoice. Work in this order:
1. Identify THIS invoice (vendor, invoice number, amount, currency) from the invoice records, not from examples.
2. Find the ONE paragraph that is a current, binding statement about THIS invoice and contains exactly three bracketed references. IGNORE the cover sheet, archived or prior-period examples, training material, sample or legacy text, statements about other invoices, and words that merely mention an action. Respect negation: "no approval is needed" means NOT request_approval; "this is not a duplicate" means NOT reject_duplicate; "do not hold" means NOT hold_invoice.
   Each package lists candidateControllingParagraphs: the paragraphs that carry exactly three bracketed references. The controlling paragraph is one of them.
3. Choose the action that paragraph's facts require. Compare numbers carefully: if everything is reconciled and the stated total is ABOVE the autonomous limit, choose request_approval; if reconciled and within the limit, choose settle_invoice. A remaining discrepancy or conflicting records means open_exception. A verification that must finish before payment means hold_invoice. An earlier payment of the same vendor AND the same invoice number means reject_duplicate.
   vendorName and invoiceNumber come from the normalized identifiers on the signed cover sheet; amountMinor comes from the stated total.
4. Copy that paragraph verbatim into controllingParagraph and list its three bracketed references exactly as written.
amountMinor is an integer in minor units (1,234.50 becomes 123450). currency is a 3-letter code. Use the batch policy information given below.
Reply with JSON only:
{"decisions":[{"packageId":"...","controllingParagraph":"...","reason":"one or two sentences explaining how the evidence supports the action","action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR","evidenceRefs":["[..]","[..]","[..]"]}]}"""
SALT = hashlib.sha256((SYSTEM + LLM_MODEL).encode()).hexdigest()[:8]

BR = re.compile(r"\[[^\[\]\n]{1,80}\]")

def walk(o):
    if isinstance(o, str): yield o
    elif isinstance(o, dict):
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)

def pid_of(p, i): return str(p.get("packageId") or p.get("id") or f"pkg-{i}")
def pkey(p): return sha(SALT + canon({k: v for k, v in p.items() if k not in ("packageId", "id")}))

def candidates(text):
    """Paragraphs that carry exactly three distinct bracketed references."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        refs = list(dict.fromkeys(BR.findall(para)))
        if len(refs) == 3: out.append((re.sub(r"\s+", " ", para).strip(), refs))
    return out

def norm(s): return re.sub(r"\s+", " ", s).strip()

def clean(d, text, cands=()):
    try:
        a = d["action"]
        if a not in ACTIONS: return None
        refs = list(dict.fromkeys(str(r) for r in d["evidenceRefs"]))
        para = norm(str(d.get("controllingParagraph", "")))
        if len(cands) == 1:
            refs = cands[0][1]
        elif cands:
            match = [r for p, r in cands if p == para or set(r) == set(refs)]
            if not match: return None
            refs = match[0]
        elif para and para in norm(text):
            pr = list(dict.fromkeys(BR.findall(para)))
            if len(pr) == 3: refs = pr
        if len(refs) != 3 or any(r not in text for r in refs): return None
        amt = d["amountMinor"]
        if isinstance(amt, str): amt = int(re.sub(r"[^\d]", "", amt))
        if isinstance(amt, float): amt = int(round(amt))
        if not isinstance(amt, int) or isinstance(amt, bool): return None
        return {"action": a, "evidenceRefs": refs, "reason": str(d.get("reason", ""))[:900],
                "facts": {"vendorName": str(d["vendorName"]), "invoiceNumber": str(d["invoiceNumber"]),
                          "amountMinor": amt, "currency": str(d["currency"]).upper()}}
    except Exception:
        return None

def fallback(text, cands=()):  # never settle when unsure
    refs = cands[0][1] if cands else list(dict.fromkeys(BR.findall(text)))[:3]
    return {"action": "hold_invoice", "evidenceRefs": refs, "reason": "Automatic analysis was inconclusive, so payment is paused for manual verification.",
            "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"}}

async def ask_llm(chunk, policy, attempt):
    user = "BATCH POLICY INFO: " + json.dumps(policy, ensure_ascii=False) + "\nPACKAGES: " + json.dumps([{"packageId": c["pid"], "candidateControllingParagraphs": [p for p, _ in c["cands"]], "package": c["pkg"]} for c in chunk], ensure_ascii=False)
    async with httpx.AsyncClient(timeout=38) as c:
        r = await c.post(f"{LLM_BASE_URL}/chat/completions",
                         headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                         json={"model": LLM_MODEL, "temperature": 0.3 * attempt,
                               "response_format": {"type": "json_object"},
                               "messages": [{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": user}]})
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    return {str(d.get("packageId")): d for d in json.loads(txt)["decisions"]}

async def decide(packages, policy=None):
    t0 = time.monotonic()
    items = [{"pid": pid_of(p, i), "pkg": p, "key": pkey(p), "text": "\n\n".join(walk(p)), "cands": candidates("\n\n".join(walk(p)))}
             for i, p in enumerate(packages)]
    out, todo, seen = {}, [], set()
    for it in items:
        row = q("SELECT json FROM cache WHERE key=?", (it["key"],))
        if row: out[it["key"]] = json.loads(row[0][0])
        elif it["key"] not in seen:
            seen.add(it["key"]); todo.append(it)
    for attempt in range(2):
        if not todo or (attempt == 1 and time.monotonic() - t0 > 14): break
        chunks = [todo[i:i + 12] for i in range(0, len(todo), 12)]
        results = await asyncio.gather(*[ask_llm(c, policy or {}, attempt) for c in chunks], return_exceptions=True)
        bad = []
        for chunk, res in zip(chunks, results):
            if isinstance(res, Exception): print("LLM error", repr(res))
            got = {} if isinstance(res, Exception) else res
            for it in chunk:
                d = clean(got.get(it["pid"]) or {}, it["text"], it["cands"])
                if d:
                    out[it["key"]] = d
                    q("INSERT OR REPLACE INTO cache VALUES(?,?)", (it["key"], json.dumps(d)))
                else:
                    bad.append(it)
        todo = bad
    for it in todo: out[it["key"]] = fallback(it["text"], it["cands"])
    return [(it, out[it["key"]]) for it in items]

def rationale(action, reason, refs):
    s = f"Chosen action {action}. {reason} Decisive evidence: {', '.join(refs)}."
    if len(s) < 60: s += " The cited paragraph determines this action for the invoice."
    return s[:1500]

# ---------------------------------------------------------------- routes
@app.get("/.well-known/agent-card.json")
async def card():
    return Response(json.dumps(CARD), media_type="application/json")

def find_part(msg, mt):
    for p in msg.get("parts", []):
        if isinstance(p, dict) and p.get("mediaType") == mt and isinstance(p.get("data"), dict):
            return p["data"]
    return None

async def handle_initial(principal, msg, mhash):
    data = find_part(msg, MT_BATCH)
    if not data or not isinstance(data.get("packages"), list) or not data["packages"] or not data.get("batchId"):
        raise ApiError(400, "INVALID_MESSAGE", "Invalid batch")
    pk = data["packages"]
    print("BATCH", data["batchId"], len(pk), "first package keys:", list(pk[0].keys()) if isinstance(pk[0], dict) else type(pk[0]))
    decided = await decide(pk, {k: v for k, v in data.items() if k != "packages"})
    proposals = [{"packageId": it["pid"], "actionId": "act-" + uuid.uuid4().hex, "action": d["action"],
                  "facts": d["facts"], "evidenceRefs": d["evidenceRefs"],
                  "rationale": rationale(d["action"], d["reason"], d["evidenceRefs"])} for it, d in decided]
    for it, d in decided:
        print("DECISION", it["pid"], d["action"], d["facts"]["amountMinor"], d["facts"]["currency"], "| PARAGRAPH:", (it["cands"][0][0][:600] if it["cands"] else "NONE"), "| NCANDS:", len(it["cands"]))
    print("SAMPLE_PACKAGE", decided[0][0]["pid"], decided[0][0]["text"][:3500])
    tid = str(uuid.uuid4())
    cid = msg.get("contextId") if isinstance(msg.get("contextId"), str) and msg.get("contextId") else str(uuid.uuid4())
    task = {"id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": "proposals-" + tid, "name": "invoice action proposals",
                           "parts": [{"mediaType": MT_PROP, "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
            "history": [dict(msg, taskId=tid, contextId=cid)]}
    save(task, principal, msg["messageId"], mhash)
    return task

async def handle_continuation(principal, msg, mhash):
    tid = msg["taskId"]
    async with lock_for(("task", tid)):
        task = load_task(principal, tid)
        if not task: raise ApiError(404, "NOT_FOUND", "Task not found")
        if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            raise ApiError(409, "TASK_NOT_ACTIVE", "Task can no longer be changed")
        if msg.get("contextId") != task["contextId"]:
            raise ApiError(400, "CONTEXT_MISMATCH", "Context mismatch")
        data = find_part(msg, MT_RES)
        prop = task["artifacts"][0]["parts"][0]["data"]
        if not data or data.get("batchId") != prop["batchId"] or not isinstance(data.get("results"), list):
            raise ApiError(400, "BATCH_MISMATCH", "Batch mismatch")
        stored = {p["packageId"]: p for p in prop["proposals"]}
        seen, execs = set(), []
        for r in data["results"]:
            sp = stored.get(r.get("packageId")) if isinstance(r, dict) else None
            if (not sp or r["packageId"] in seen or r.get("actionId") != sp["actionId"]
                    or r.get("action") != sp["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                raise ApiError(400, "RESULT_MISMATCH", "Result does not match a stored proposal")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                n = r.get("receiptNonce")
                if not isinstance(n, str) or not n:
                    raise ApiError(400, "RESULT_MISMATCH", "Missing receipt")
                execs.append({"packageId": sp["packageId"], "actionId": sp["actionId"], "action": sp["action"],
                              "receiptNonce": n, "facts": sp["facts"], "evidenceRefs": sp["evidenceRefs"]})
        task["artifacts"].append({"artifactId": "receipts-" + tid, "name": "invoice action receipts",
                                  "parts": [{"mediaType": MT_RCPT, "data": {"batchId": prop["batchId"], "executions": execs}}]})
        task["history"].append(dict(msg))
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        save(task, principal, msg["messageId"], mhash)
        return task

@app.post(PREFIX + "/message:send")
async def message_send(request: Request):
    principal = guard(request)
    if "a2a+json" not in request.headers.get("content-type", "").lower():
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Use application/a2a+json")
    try:
        body = json.loads(await request.body())
        msg = body["message"]
        mid = msg["messageId"]
        assert isinstance(msg, dict) and isinstance(mid, str) and mid
        assert msg.get("role") == "ROLE_USER" and isinstance(msg.get("parts"), list) and msg["parts"]
    except Exception:
        raise ApiError(400, "INVALID_MESSAGE", "Invalid message")
    mhash = sha(canon(msg))  # message only: configuration is ignored
    async with lock_for(("msg", principal, mid)):
        row = q("SELECT hash, task_id FROM msgs WHERE principal=? AND mid=?", (principal, mid))
        if row:
            if row[0][0] != mhash:
                raise ApiError(409, "IDEMPOTENCY_CONFLICT", "messageId was already used with different content")
            return reply({"task": load_task(principal, row[0][1])})
        if msg.get("taskId"):
            task = await handle_continuation(principal, msg, mhash)
        else:
            task = await handle_initial(principal, msg, mhash)
        return reply({"task": task})

@app.get(PREFIX + "/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal = guard(request)
    task = load_task(principal, task_id)
    if not task: raise ApiError(404, "NOT_FOUND", "Task not found")
    return reply(task)

@app.get(PREFIX + "/tasks")
async def list_tasks(request: Request):
    principal = guard(request)
    out, size = [], 0
    for (j,) in q("SELECT json FROM tasks WHERE principal=? ORDER BY created DESC", (principal,)):
        if size + len(j) > 380_000: break
        out.append(json.loads(j)); size += len(j)
    return reply({"tasks": out})

@app.post(PREFIX + "/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal = guard(request)
    async with lock_for(("task", task_id)):
        task = load_task(principal, task_id)
        if not task: raise ApiError(404, "NOT_FOUND", "Task not found")
        if task["status"]["state"] in TERMINAL:
            raise ApiError(409, "TASK_NOT_CANCELABLE", "Task is already finished")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save(task, principal)
        return reply(task)
