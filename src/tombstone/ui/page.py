"""The single page ``tombstone ui`` serves. Self-contained: no network, no build step."""

from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tombstone</title>
<style>
:root{
  --bg:#fbfbfa; --panel:#fff; --ink:#1a1a19; --dim:#6b6b66; --line:#e4e4e0;
  --accent:#3a5a8c; --ok:#2f6b46; --warn:#8a5a12; --bad:#9b2c2c; --human:#5b4a8a;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 8px 24px rgba(0,0,0,.04);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#16171a; --panel:#1d1f23; --ink:#e8e8e6; --dim:#9a9a95; --line:#2e3036;
    --accent:#8fb0e0; --ok:#6cc08c; --warn:#d9a441; --bad:#e07a7a; --human:#a596d8;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25);
  }
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.wrap{max-width:860px; margin:0 auto; padding:40px 24px 96px}
header{display:flex; align-items:baseline; gap:12px; margin-bottom:6px}
h1{font-size:20px; margin:0; letter-spacing:-.01em}
.sub{color:var(--dim); font-size:13px}
.panel{
  background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:20px; margin-top:20px; box-shadow:var(--shadow);
}
label{display:block; font-size:12px; color:var(--dim); margin-bottom:6px;
  text-transform:uppercase; letter-spacing:.06em}
input[type=text]{
  width:100%; padding:11px 13px; font-size:15px; font-family:inherit;
  background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:7px;
}
input[type=text]:focus{outline:2px solid var(--accent); outline-offset:-1px; border-color:transparent}
.row{display:flex; gap:10px; align-items:flex-end}
.row > div{flex:1}
button{
  font:inherit; font-weight:500; padding:11px 18px; border-radius:7px; cursor:pointer;
  border:1px solid var(--line); background:var(--panel); color:var(--ink);
}
button:hover:not(:disabled){border-color:var(--dim)}
button:disabled{opacity:.5; cursor:default}
button.primary{background:var(--accent); border-color:var(--accent); color:#fff}
@media (prefers-color-scheme:dark){button.primary{color:#16171a}}
button.danger{background:var(--bad); border-color:var(--bad); color:#fff}
@media (prefers-color-scheme:dark){button.danger{color:#16171a}}
table{width:100%; border-collapse:collapse; font-size:14px; margin-top:4px}
th{text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.06em;
  color:var(--dim); font-weight:600; padding:7px 0; border-bottom:1px solid var(--line)}
td{padding:9px 0; border-bottom:1px solid var(--line)}
td.num,th.num{text-align:right; font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px}
.big{font-size:28px; font-weight:600; letter-spacing:-.02em}
.counts{display:flex; gap:22px; flex-wrap:wrap; margin:2px 0 14px}
.count .n{font-size:24px; font-weight:600; font-variant-numeric:tabular-nums}
.count .l{font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim)}
.verified .n{color:var(--ok)} .unverified .n{color:var(--warn)}
.residual .n{color:var(--bad)} .needs_human .n{color:var(--human)}
.note{
  border-left:3px solid var(--line); padding:10px 0 10px 14px; margin:14px 0;
  font-size:14px; color:var(--dim);
}
.note.warn{border-color:var(--warn); color:var(--ink)}
.note.bad{border-color:var(--bad); color:var(--ink)}
.note b{color:var(--ink)}
pre{
  background:var(--bg); border:1px solid var(--line); border-radius:7px; padding:14px;
  overflow-x:auto; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:12.5px; line-height:1.5; white-space:pre; margin:0;
}
.muted{color:var(--dim)}
/* a utility class has to beat whatever it is sitting on, whatever the source order: .check is
   declared below this line and would otherwise win on equal specificity and stay visible. */
.hide{display:none !important}
.spin{display:inline-block; width:13px; height:13px; border:2px solid var(--line);
  border-top-color:var(--accent); border-radius:50%; animation:s .7s linear infinite;
  vertical-align:-2px; margin-right:7px}
@keyframes s{to{transform:rotate(360deg)}}
.check{display:flex; gap:8px; align-items:flex-start; margin:14px 0; font-size:14px}
.check input{margin:3px 0 0}
footer{margin-top:28px; font-size:12.5px; color:var(--dim); line-height:1.6}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Tombstone</h1>
    <span class="sub">what do we hold on this person, and can we prove we removed it</span>
  </header>

  <div class="panel">
    <form id="find">
      <div class="row">
        <div>
          <label for="subject">Person</label>
          <input type="text" id="subject" placeholder="customer id, account id, email"
                 autocomplete="off" autofocus>
        </div>
        <button class="primary" type="submit" id="findBtn">Look</button>
      </div>
    </form>
    <div class="muted" style="font-size:12.5px;margin-top:10px">
      The id is hashed before it reaches the database. It is never stored or logged in the clear.
    </div>
  </div>

  <div id="err" class="panel hide"><div class="note bad" id="errText"></div></div>

  <div id="held" class="panel hide">
    <div class="big"><span id="heldCount"></span> <span class="muted" style="font-size:15px;font-weight:400"
      >things exist because of <span class="mono" id="heldSubject"></span></span></div>
    <table>
      <thead><tr><th>Where it lives</th><th class="num">Artifacts</th></tr></thead>
      <tbody id="heldRows"></tbody>
    </table>
    <div class="muted" style="font-size:12.5px;margin-top:10px" id="heldKinds"></div>

    <div id="heldHuman" class="note warn hide"></div>
    <div id="heldGaps" class="note bad hide"></div>

    <div id="eraseBox" style="margin-top:22px;border-top:1px solid var(--line);padding-top:20px">
      <div class="row">
        <div>
          <label for="reason">Reason (goes on the receipt)</label>
          <input type="text" id="reason" placeholder="dsr-2026-0912" autocomplete="off">
        </div>
        <button class="danger" id="eraseBtn" disabled>Erase everything</button>
      </div>
      <div class="check hide" id="gapCheckWrap">
        <input type="checkbox" id="gapCheck">
        <label for="gapCheck" style="text-transform:none;letter-spacing:0;font-size:14px;color:var(--ink);margin:0">
          Accept the lineage gaps and record <span class="mono">UNVERIFIED(lineage-gap)</span> on
          the receipt. The tool cannot follow data it never saw arrive, and will not claim it did.
        </label>
      </div>
      <div class="muted" style="font-size:12.5px;margin-top:10px">
        This is not reversible. It suppresses, reclaims and then checks every layer.
      </div>
    </div>
  </div>

  <div id="receipt" class="panel hide">
    <div style="font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin-bottom:12px">
      Receipt <span class="mono" id="rid"></span>
    </div>
    <div class="counts" id="rcounts"></div>
    <pre id="rreport"></pre>
    <div class="muted" style="margin-top:12px;font-size:12.5px">
      A receipt is a record of what was done and checked. It is not a legal instrument.
    </div>
  </div>

  <div id="past" class="panel hide">
    <label>Earlier erasures</label>
    <table>
      <thead><tr><th>Receipt</th><th>Person</th><th>Reason</th>
        <th class="num">Verified</th><th class="num">Unverified</th></tr></thead>
      <tbody id="pastRows"></tbody>
    </table>
  </div>

  <footer>
    Tombstone can only erase what it saw arrive. Data ingested before it was wrapped around your
    stores has no lineage, is reported as a gap, and is never counted as erased. Backups,
    replicas and your providers' own logs are outside what any application-level tool can reach,
    and every receipt says so.
  </footer>
</div>

<script>
const TOKEN = new URLSearchParams(location.search).get("t") || "";
history.replaceState(null, "", location.pathname);   // keep it out of the address bar and history

const $ = (id) => document.getElementById(id);
let current = null;

async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-Tombstone-Token": TOKEN},
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({error: "the server sent something unreadable"}));
  if (!r.ok) throw new Error(data.error || ("request failed: " + r.status));
  return data;
}

function showError(msg) {
  $("errText").textContent = msg;
  $("err").classList.remove("hide");
}
function clearError() { $("err").classList.add("hide"); }

function busy(btn, on, label) {
  btn.disabled = on;
  btn.innerHTML = on ? '<span class="spin"></span>' + label : label;
}

$("find").addEventListener("submit", async (e) => {
  e.preventDefault();
  const subject = $("subject").value.trim();
  if (!subject) return;
  clearError();
  $("held").classList.add("hide");
  $("receipt").classList.add("hide");
  busy($("findBtn"), true, "Looking");
  try {
    current = await api("/api/trace", {subject});
    renderHeld(current);
  } catch (err) {
    showError(err.message);
  } finally {
    busy($("findBtn"), false, "Look");
  }
});

function renderHeld(h) {
  $("heldCount").textContent = h.count;
  $("heldSubject").textContent = h.subject;
  $("heldRows").innerHTML = Object.keys(h.by_store).sort().map((s) =>
    `<tr><td class="mono">${esc(s)}</td><td class="num">${h.by_store[s]}</td></tr>`).join("");
  $("heldKinds").textContent = "kinds: " + Object.keys(h.by_kind).sort()
    .map((k) => k + "×" + h.by_kind[k]).join(", ");

  const human = $("heldHuman");
  if (h.needs_human > 0) {
    human.innerHTML = `<b>${h.needs_human} document(s) belong to other people</b> and mention
      this person. They are listed for human review and are never erased automatically —
      deleting another person's record would be a different violation.`;
    human.classList.remove("hide");
  } else human.classList.add("hide");

  const gaps = $("heldGaps");
  if (h.gaps && h.gaps.length) {
    gaps.innerHTML = `<b>${h.gaps.length} lineage gap(s).</b> Some data in these stores arrived
      before Tombstone was watching, so there is no trail to follow:<br>` +
      h.gaps.map((g) => `<span class="mono">${esc(g)}</span>`).join("<br>");
    gaps.classList.remove("hide");
    $("gapCheckWrap").classList.remove("hide");
  } else {
    gaps.classList.add("hide");
    $("gapCheckWrap").classList.add("hide");
    $("gapCheck").checked = false;
  }
  $("held").classList.remove("hide");
  updateEraseBtn();
}

$("reason").addEventListener("input", updateEraseBtn);
function updateEraseBtn() {
  $("eraseBtn").disabled = !(current && current.count > 0 && $("reason").value.trim());
}

$("eraseBtn").addEventListener("click", async () => {
  if (!current) return;
  const reason = $("reason").value.trim();
  const n = current.count;
  if (!confirm(`Erase all ${n} artifacts belonging to ${current.subject}?\\n\\n` +
               `Reason: ${reason}\\n\\nThis cannot be undone.`)) return;
  clearError();
  busy($("eraseBtn"), true, "Erasing");
  try {
    const res = await api("/api/forget", {
      subject: $("subject").value.trim(),
      reason,
      accept_gaps: $("gapCheck").checked,
      expect_trace_id: current.trace_id,
    });
    renderReceipt(res);
    $("held").classList.add("hide");
    current = null;
    loadPast();
  } catch (err) {
    showError(err.message);
  } finally {
    busy($("eraseBtn"), false, "Erase everything");
  }
});

function renderReceipt(r) {
  $("rid").textContent = r.receipt_id;
  const order = ["verified", "unverified", "residual", "out_of_scope", "needs_human"];
  $("rcounts").innerHTML = order.filter((k) => k in r.counts).map((k) =>
    `<div class="count ${k}"><div class="n">${r.counts[k]}</div>
     <div class="l">${k.replace("_", " ")}</div></div>`).join("");
  $("rreport").textContent = r.report;
  $("receipt").classList.remove("hide");
  $("receipt").scrollIntoView({behavior: "smooth", block: "start"});
}

async function loadPast() {
  try {
    const d = await api("/api/receipts", {});
    if (!d.receipts.length) { $("past").classList.add("hide"); return; }
    $("pastRows").innerHTML = d.receipts.map((r) =>
      `<tr><td class="mono">${esc(r.receipt_id.slice(0, 10))}…</td>
       <td class="mono">${esc(r.subject)}</td><td>${esc(r.reason)}</td>
       <td class="num">${r.counts.verified || 0}</td>
       <td class="num">${r.counts.unverified || 0}</td></tr>`).join("");
    $("past").classList.remove("hide");
  } catch (e) { /* the history panel is a nicety; never let it break the page */ }
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
}

loadPast();
</script>
</body>
</html>
"""
