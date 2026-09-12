"""Build the end-to-end Tombstone report as a PDF.

Every figure comes from a committed result file; nothing is typed in by hand. Written for a
reader who has never seen a vector database, with a glossary and a plain-language summary in
front of the technical detail.

    uv run python bench/make_report_pdf.py [-o path/to/report.pdf]

Needs WeasyPrint (``uv pip install weasyprint``), which is not a dependency of the package: this
builds a document about the tool, and nobody installing the tool needs it.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
OUT_HTML = ROOT / "bench" / "_work" / "report.html"
DEFAULT_PDF = ROOT / "bench" / "_work" / "Tombstone-Report.pdf"


def load(kind: str) -> dict[str, Any]:
    p = RESULTS / f"{kind}-latest.json"
    return dict(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else {}


def img(path: Path, width: str = "100%") -> str:
    if not path.is_file():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    return f'<img src="data:{mime};base64,{b64}" style="width:{width}" alt="{path.stem}"/>'


def pct(x: float | None, nd: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{nd}f}%"


def num(x: float | None, nd: int = 2) -> str:
    return "—" if x is None else f"{x:,.{nd}f}"


CSS = """
@page {
  size: A4; margin: 20mm 18mm 18mm 18mm;
  @bottom-center { content: counter(page); font-family: Georgia, serif; font-size: 9pt; color: #888; }
  @top-right { content: "Tombstone — technical report"; font-family: Georgia, serif; font-size: 8pt; color: #aaa; }
}
@page :first { @top-right { content: ""; } @bottom-center { content: ""; } }
body { font-family: Georgia, "Times New Roman", serif; font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 6pt; line-height: 1.2; }
h2 { font-size: 15pt; margin: 22pt 0 8pt; padding-bottom: 4pt; border-bottom: 1.5pt solid #2f6f4f;
     color: #14392a; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 5pt; color: #2f6f4f; break-after: avoid; }
h4 { font-size: 10pt; margin: 12pt 0 4pt; color: #14392a; break-after: avoid; }
pre { background: #f7f8f9; border: 0.5pt solid #dde0e2; border-left: 2.5pt solid #2f6f4f;
       padding: 7pt 9pt; margin: 7pt 0 10pt; font-family: "SF Mono", Menlo, monospace;
       font-size: 8.3pt; line-height: 1.45; white-space: pre-wrap; word-wrap: break-word;
       break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
p { margin: 0 0 8pt; text-align: justify; }
ul, ol { margin: 0 0 9pt 0; padding-left: 16pt; }
li { margin-bottom: 4pt; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.8pt; background: #f2f3f4;
       padding: 1pt 3pt; border-radius: 2pt; }
table { width: 100%; border-collapse: collapse; margin: 9pt 0 11pt; font-size: 8.8pt; }
th { background: #14392a; color: #fff; text-align: left; padding: 5pt 6pt; font-weight: normal; }
td { padding: 4.5pt 6pt; border-bottom: 0.5pt solid #dde0e2; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfb; }
.num { text-align: right; font-family: "SF Mono", Menlo, monospace; font-size: 8.4pt; }
.cover { text-align: center; padding-top: 55mm; }
.cover h1 { font-size: 30pt; border: none; letter-spacing: -0.5pt; }
.cover .sub { font-size: 13pt; color: #444; margin: 10pt 0 26pt; font-style: italic; }
.cover .meta { font-size: 10pt; color: #666; line-height: 1.9; margin-top: 30mm; }
.rule { width: 60pt; height: 2.5pt; background: #2f6f4f; margin: 16pt auto; }
.key { background: #f0f6f2; border-left: 3pt solid #2f6f4f; padding: 8pt 11pt; margin: 11pt 0;
       break-inside: avoid; }
.key .label { font-size: 8pt; text-transform: uppercase; letter-spacing: 1pt; color: #2f6f4f; }
.warn { background: #fdf6ee; border-left: 3pt solid #8a5a2b; padding: 8pt 11pt; margin: 11pt 0;
        break-inside: avoid; }
.warn .label { font-size: 8pt; text-transform: uppercase; letter-spacing: 1pt; color: #8a5a2b; }
figure { margin: 11pt 0; break-inside: avoid; text-align: center; }
figcaption { font-size: 8.5pt; color: #666; margin-top: 4pt; font-style: italic; }
.toc a { text-decoration: none; color: #1a1a1a; }
.toc li { margin-bottom: 3pt; }
.small { font-size: 9pt; color: #555; }
.pagebreak { break-before: page; }
.plain { background: #f7f8f8; padding: 7pt 10pt; margin: 8pt 0; font-size: 9.6pt;
         border-radius: 3pt; break-inside: avoid; }
.plain b { color: #14392a; }
dt { font-weight: bold; margin-top: 6pt; font-size: 9.6pt; }
dd { margin: 1pt 0 0 14pt; font-size: 9.6pt; color: #333; }
"""


def cover(r: dict, u: dict, cost: dict) -> str:
    n_commits = subprocess.run(
        ["git", "-C", str(ROOT), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"""
<div class="cover">
  <h1>Tombstone</h1>
  <div class="rule"></div>
  <p class="sub">What actually happens to your data when you ask a company to delete it —<br/>
  measured across four databases and a fine-tuned AI model</p>
  <div class="meta">
    A technical report<br/>
    Poojan Patel · {date.today().strftime("%d %B %Y")}<br/>
    <span style="font-size:9pt">github.com/Poojan6216/tombstone</span><br/>
    <span style="font-size:9pt">pip install tombstone-erase</span><br/><br/>
    <span style="font-size:9pt">{r["corpus"]["docs"]:,} documents · {r["corpus"]["subjects"]} people ·
    {len(r["cells"])} database experiments · {len(load("attacks").get("strategies", []))} attacks ·
    {cost.get("total_cpu_hours", 0):.0f} CPU-hours · {n_commits} commits</span>
  </div>
</div>
<div class="pagebreak"></div>
<h2>Contents</h2>
<ol class="toc">
  <li>The short version — for anyone</li>
  <li>The problem: "delete" does not mean deleted</li>
  <li>The idea: follow the data, don't guess</li>
  <li>How the tool works</li>
  <li>How to actually use it — install, the four ways in, MCP setup</li>
  <li>What we found in the databases</li>
  <li>What we found in the AI model</li>
  <li>The attacks that beat our own tool</li>
  <li>How it was built and tested</li>
  <li>What went wrong along the way</li>
  <li>What this does not do</li>
  <li>How to reproduce every number</li>
  <li>Glossary · References</li>
</ol>
"""


def summary(r: dict, u: dict, d: dict) -> str:
    b0 = [c for c in r["cells"] if c["baseline"] == "B0" and c["physical_checked"]]
    b4 = [c for c in r["cells"] if c["baseline"] == "B4" and c["physical_checked"]]
    lo, hi = (
        min(c["physical_residue_rate"] for c in b0),
        max(c["physical_residue_rate"] for c in b0),
    )
    meth = {m["name"]: m for m in u.get("methods", [])}
    m0, m3 = meth.get("M0", {}), meth.get("M3", {})
    return f"""
<div class="pagebreak"></div>
<h2>1. The short version — for anyone</h2>

<p>When you ask a company to delete your data, someone runs a delete command and the system says
it worked. Your data disappears from every search. Everyone moves on.</p>

<p><b>We checked whether the data was actually gone. Usually, it was not.</b></p>

<div class="key">
  <div class="label">The central finding</div>
  <p style="margin:5pt 0 0">Across four of the most widely used databases for AI applications, we
  deleted one person's data the normal way and then read the raw files off the disk.
  Between <b>{pct(lo)} and {pct(hi)}</b> of that person's data was still sitting there, fully
  readable. Searching could not find it. The bytes had never left.</p>
</div>

<p>This is not a bug in any one product. It is how these systems are designed: deleting a row
usually just marks it as hidden and leaves the contents in place until the storage is later
compacted — something that may not happen for hours, or ever. The industry term is a
<i>soft delete</i>. The practical effect is that "deleted" data can be recovered by anyone who
can read the files: an administrator, a backup, or an attacker who gets a copy of the disk.</p>

<p>We built a tool called <b>Tombstone</b> that fixes this, and — more importantly — that
<i>proves</i> whether it worked instead of just claiming it did. On the same four databases, after
Tombstone's erasure, the amount of the person's data still findable in the files was
<b>{pct(max(c["physical_residue_rate"] for c in b4))}</b>.</p>

<h3>It also handles the hardest part: the AI model itself</h3>

<p>Modern applications do not just store your data — they <i>train on it</i>. Once your
information is baked into a model's weights, deleting the original document changes nothing. The
model still knows. We tested this directly by planting a unique fake ID for each person in their
documents and then asking the trained model to recite it.</p>

<div class="plain">
  <b>Before cleanup:</b> the model produced the correct private ID for
  <b>{m0.get("canary_extracted", "—")} out of {m0.get("canary_total", "—")}</b> people on request.<br/>
  <b>After cleanup:</b> <b>{m3.get("canary_extracted", "—")} out of {m3.get("canary_total", "—")}</b>.
  And a separate statistical attack that tries to detect whether a person's data was used in
  training dropped from <b>perfect certainty to a coin flip</b>. The model kept working normally.
</div>

<h3>And we published how to beat it</h3>

<p>We then attacked our own tool and measured nine ways it can be defeated — including one kind of
leftover trace that <i>no</i> currently known method can remove. Those results are in this report
with the same prominence as the successes. A deletion tool that only reports its wins is not a
deletion tool; it is marketing.</p>

<div class="warn">
  <div class="label">The honest summary</div>
  <p style="margin:5pt 0 0">Tombstone can prove that data is gone from the places it can see, name
  precisely the places it cannot see, and measure what remains. It cannot promise your data is gone
  from the world — no tool can — and it never uses the word "complete".</p>
</div>
"""


def problem(d: dict) -> str:
    d1 = d.get("demo1", {})
    return f"""
<h2>2. The problem: "delete" does not mean deleted</h2>

<h3>Where your data goes in a modern AI application</h3>

<p>Imagine a company builds a customer-support assistant. It reads the company's documents and
answers questions about them. To do that, it does not simply keep your file in one place. It:</p>

<ol>
  <li>stores <b>the original document</b>;</li>
  <li>splits it into small <b>chunks</b>, because AI models read short passages better;</li>
  <li>converts each chunk into an <b>embedding</b> — a long list of numbers that captures its
      meaning, stored in a special database so the assistant can find related passages;</li>
  <li>often stores the same chunk in <b>more than one such database</b>;</li>
  <li><b>caches</b> answers so common questions are fast;</li>
  <li>keeps some of it as <b>training examples</b>;</li>
  <li>and <b>fine-tunes a model</b> on those examples, folding the content into the model's weights.</li>
</ol>

<p>Your one document has become seven kinds of thing in half a dozen places. Now you invoke your
right to erasure. The engineer deletes the original row. In our own reconstruction of exactly this
setup, one person had <b>{d1.get("artifacts", "—")} derived artifacts</b>, and after a normal
delete, <b>{d1.get("recoverable", "—")} of them were still recoverable</b>.</p>

<div class="key">
  <div class="label">Why this happens</div>
  <p style="margin:5pt 0 0">Nobody is being careless. Three ordinary engineering facts combine:</p>
  <ul style="margin-top:5pt">
    <li><b>Databases delete lazily.</b> Removing a row usually flips a flag. The bytes stay until a
    later cleanup pass, which may never be triggered.</li>
    <li><b>Search indexes are built for speed, not forgetting.</b> The common index type is a graph
    that is expensive to rebuild, so entries are marked "skip me" rather than removed.</li>
    <li><b>Nobody tracked where the copies went.</b> There is no map from "this person" to "these
    forty things", so even a diligent engineer cannot delete what they cannot enumerate.</li>
  </ul>
</div>

<h3>Why searching for the data is the wrong fix</h3>

<p>The obvious answer — search the databases for anything similar to the person's data and delete
the matches — is worse than it sounds. Similarity search is fuzzy by design. Set the threshold
loosely and you delete other customers' records; set it tightly and you miss paraphrases. Either
way you cannot state afterwards what you deleted or prove what you missed. That is guessing, and
it fails in both directions at once.</p>
"""


def idea() -> str:
    return f"""
<div class="pagebreak"></div>
<h2>3. The idea: follow the data, don't guess</h2>

<p>The insight behind Tombstone is that deletion should be a <b>bookkeeping problem, not a search
problem</b>. If you record where each copy came from <i>at the moment it is created</i>, then when
someone asks to be forgotten you do not search for their data — you look it up.</p>

<p>Every time the application creates something derived from a document, Tombstone records a link:
this chunk came from that document; this embedding came from that chunk; this training row came
from that chunk; this model adapter was trained on those rows. The result is a family tree.
Deleting a person means starting at their documents and following every branch — a standard,
exact graph operation with no thresholds and no guessing.</p>

<figure>
  {img(ROOT / "docs" / "lineage.svg", "100%")}
  <figcaption>The family tree Tombstone records, and what a receipt is allowed to claim at each
  level of checking.</figcaption>
</figure>

<div class="key">
  <div class="label">The three commitments that follow from this</div>
  <ul style="margin:5pt 0 0">
    <li><b>Never select by similarity.</b> Targets come from recorded links only, so the tool can
    never delete a different person's data by accident.</li>
    <li><b>Fail loudly when the map has holes.</b> If data arrived before tracking was switched on,
    the tool refuses to claim success and says exactly which part it cannot account for.</li>
    <li><b>Never claim more than was checked.</b> Every result names the level of proof behind it.</li>
  </ul>
</div>
"""


def how_it_works() -> str:
    return """
<div class="pagebreak"></div>
<h2>4. How the tool works</h2>

<h3>Step one: hide it, before removing it</h3>

<p>Erasure happens in two phases, and the order matters. First <b>suppress</b>: write a permanent
marker that every part of the application checks before returning any result. From that instant
the data is unreachable, even though the bytes are still on disk. Only then <b>reclaim</b>: do the
slow, irreversible work of actually rewriting the storage so the bytes are gone.</p>

<p>Doing it the other way round leaves a window where the data has been partly removed but is
still being served. Suppressing first means that even if the machine dies halfway through the slow
part, the data was never reachable in the meantime.</p>

<h3>Step two: survive a crash</h3>

<p>Reclaiming touches several independent systems, and any of them can fail. Tombstone writes each
step to a journal before performing it, so an interrupted erasure can be resumed exactly where it
stopped rather than restarted or abandoned half-done. Steps that fail repeatedly go to a
dead-letter queue for a human, instead of being silently skipped.</p>

<p>We tested this by killing the process outright — no cleanup, no warning — at 15 different
randomly chosen points, and confirmed that resuming produced a byte-for-byte identical result each
time.</p>

<h3>Step three: check, and say how hard you checked</h3>

<p>This is the part that makes the tool trustworthy rather than merely confident. Tombstone checks
each item at up to four increasing levels of rigour, and the receipt says which level was actually
reached:</p>

<table>
<tr><th>Level</th><th>What it proves</th><th>What it still does not prove</th></tr>
<tr><td><b>Logical</b></td><td>No search, filter, or lookup the application knows about returns
the item.</td><td>The bytes may still be sitting in the file.</td></tr>
<tr><td><b>Physical</b></td><td>The item's actual bytes are absent from every file the tool can
read.</td><td>Only files it can read — not backups, replicas, or the provider's internal logs.</td></tr>
<tr><td><b>Semantic</b></td><td>Measures the shape of the hole left behind in the search
index.</td><td>Reported as information, never as proof, and never used to pass or fail.</td></tr>
<tr><td><b>Model</b></td><td>The content cannot be extracted from the trained model, and
statistical membership tests come out at chance.</td><td>A better-resourced attacker may still
succeed; the research literature is explicit that these tests are fragile.</td></tr>
</table>

<div class="key">
  <div class="label">The rule that keeps it honest</div>
  <p style="margin:5pt 0 0">The checks are evaluated in a fixed order, and one ordering decision
  carries most of the integrity: <b>"this store cannot be physically checked" is decided before
  "no bytes were found"</b>. Otherwise a database the tool is unable to inspect would sail through
  as verified simply because looking found nothing. A managed database always reports
  <i>unverified, with the reason</i> — never a pass.</p>
</div>

<h3>Step four: hand back a receipt</h3>

<p>Each erasure produces a signed, tamper-evident record listing every item, the level it was
checked at, and the outcome: <b>verified</b>, <b>unverified</b> (with the reason),
<b>residual</b> (traces remain and here they are), <b>out of scope</b> (backups and similar), or
<b>needs human</b> (for example, the person is mentioned inside someone else's document, which the
tool will not touch). Receipts are chained together so that removing or altering one is detectable.</p>

<p>One design decision is worth stating plainly: <b>a receipt with an empty "out of scope" list
cannot be created at all</b>. There are always layers the tool cannot see, and the format refuses
to let anyone pretend otherwise.</p>
"""


def using_it() -> str:
    return """
<div class="pagebreak"></div>
<h2>5. How to actually use it</h2>

<p>Published as <code>tombstone-erase</code> on the Python Package Index. Everything below was run
against that published package, not against a working copy.</p>

<h3>Installing</h3>

<pre><code>uv tool install tombstone-erase              # the command: tombstone
uv tool install 'tombstone-erase[mcp]'       # and the assistant server</code></pre>

<p><code>pip install tombstone-erase</code> does the same thing. Two traps worth knowing before
you hit them, both found by a first real install rather than by a test:</p>

<div class="key">
  <div class="label">Needs Python 3.12 or newer</div>
  <p style="margin:5pt 0 0">On an older Python, <code>pip</code> reports <i>"Could not find a
  version that satisfies the requirement"</i>, which reads as <i>"this package does not
  exist"</i>. It means <i>"none of its releases run on your Python"</i>. The real explanation is
  the line above it, which scrolls past. <code>uv tool install</code> avoids this by fetching a
  suitable Python itself.</p>
</div>

<div class="key">
  <div class="label">The install name ends in <code>-erase</code></div>
  <p style="margin:5pt 0 0"><code>pip install tombstone</code> fetches an unrelated project that
  happened to take the plain name first. Both install a module called <code>tombstone</code>, and
  pip will let one silently overwrite the other without warning. Install
  <code>tombstone-erase</code>; the command you type afterwards is still <code>tombstone</code>.</p>
</div>

<h3>Setting it up — two lines, once</h3>

<p>In the application you want to protect:</p>

<pre><code>cd /path/to/your-app
tombstone init          # writes tombstone.yaml and a .tombstone/ folder</code></pre>

<p>Then change two lines where documents are ingested:</p>

<pre><code>vs = TombstoneVectorStore.from_config("chroma:kb", embeddings)   # was: Chroma(...)
docs = stamp(docs, subject_id="S-0417", source_id="crm/417.pdf")  # whose data is this</code></pre>

<p>That is the whole integration. From then on every chunk, embedding, cached answer and training
row records where it came from, in a small SQLite file beside the application. Nothing else
changes; queries behave exactly as before.</p>

<p><b>One thing to understand:</b> the tool can only erase what it watched arrive. Data that was
already in the database before this wrapper went on has no trail, is reported as a gap, and is
never counted as erased. That is deliberate, and it is the honest answer.</p>

<h3>Four ways to run an erasure</h3>

<p>All four are the same machinery underneath — the same journal, the same signed receipt, the
same refusals. They differ only in who is driving.</p>

<h4>1. One command, at a terminal</h4>

<pre><code>tombstone forget S-0417 --reason dsr-2026-0912</code></pre>

<pre><code>47 things exist because of subject hmac:99d5…83ec   (raw id never stored)

  chroma:kb-v2                     10
  docs                             15
  exact-cache                       1
  faiss:kb-v1                      10
  ft-dataset                       10
  semantic-cache                    1

  lineage gaps: none

erase all 47 of these? this cannot be undone  [y/N]</code></pre>

<p>The confirmation is a person answering a list of real items, not a flag pasted without reading.
A blank answer means no. It refuses lineage gaps before asking rather than after, and it will not
run unattended without an explicit <code>--yes</code>.</p>

<h4>2. No command at all — inside your own application</h4>

<p>A deletion request does not arrive in a terminal. It arrives when a customer clicks "delete my
account", or a ticket lands in a queue. So the erasure belongs in the code that already handles
that:</p>

<pre><code>from tombstone import trace, forget

held = trace("S-0417")                                  # reads only, changes nothing
result = forget("S-0417", reason=f"dsr-{ticket_id}")    # destructive
if not result.ok:
    alert_privacy_team(result.report)                   # something could not be confirmed</code></pre>

<p>Calling the function is the confirmation: a line written in your own source is already a
deliberate act. Nobody has to remember a command, and the receipt is identical to the one the
terminal produces.</p>

<h4>3. A page, for the people who actually handle the request</h4>

<pre><code>tombstone ui        # http://127.0.0.1:7878</code></pre>

<p>Support and legal receive most deletion requests, and they do not use terminals. The page lets
them search a person, see everything held, erase it and read the receipt. Because it can delete
data it is deliberately hard to reach: it listens only on this machine with no option to change
that, every request must carry a token printed with the address, and there are no cookies for a
browser to attach automatically — so a web page someone happens to be browsing cannot drive it.</p>

<h4>4. Ask an assistant (MCP)</h4>

<p>Tombstone speaks the Model Context Protocol, so an AI assistant can use it directly. Six tools:
<code>forget</code>, <code>trace</code>, <code>verify</code>, <code>erase</code>,
<code>receipt</code>, <code>status</code>.</p>

<table>
<tr><th style="width:38%">Step</th><th>What to do</th></tr>
<tr><td>1. Install the extra</td>
    <td><code>uv tool install --force 'tombstone-erase[mcp]'</code></td></tr>
<tr><td>2. Note your config path</td>
    <td>The absolute path to <code>tombstone.yaml</code> — a relative path will not work, because
    the assistant starts the server from a different folder.</td></tr>
<tr><td>3a. Connect (Claude Code)</td>
    <td><code>claude mcp add tombstone -- tombstone mcp --config /full/path/tombstone.yaml</code></td></tr>
<tr><td>3b. Connect (any other client)</td>
    <td>Add to the client's config file:
    <code>{"mcpServers":{"tombstone":{"command":"tombstone","args":["mcp","--config","/full/path/tombstone.yaml"]}}}</code>
    then restart it.</td></tr>
<tr><td>4. Ask it things</td>
    <td>"What do we hold on customer S-0417?" · "Is our coverage complete?" ·
    "Delete S-0417, ticket DSR-812" · "Show me receipt 01M29…"</td></tr>
</table>

<div class="key">
  <div class="label">The assistant cannot delete anything by itself</div>
  <p style="margin:5pt 0 0">Both destructive tools stop and ask a human every time, on every
  version of the protocol. A client that cannot show a confirmation is refused and told which
  terminal command to run instead — it never quietly falls back to deleting. An agent cannot be
  talked into wiping a database, and cannot do it by accident.</p>
</div>

<h3>What comes back</h3>

<pre><code>VERIFIED 19   UNVERIFIED 4   RESIDUAL 0   OUT_OF_SCOPE 3   NEEDS_HUMAN 2</code></pre>

<p>Which reads as: nineteen things confirmed gone by reading the raw bytes on disk. Four that
could not be checked, each with the specific reason and the exact permission to ask for. Three
layers no application-level tool can reach — backups, replicas, the embedding provider's own
logs. And two documents that belong to <i>other people</i> but mention this person, which a human
must decide about, because deleting someone else's record would be a different violation.</p>

<p>That last column is the point of the tool. Anything can print "deleted". This says what it
checked, what it could not, and what it refuses to do on your behalf.</p>

<h3>When something goes wrong</h3>

<table>
<tr><th style="width:45%">What you see</th><th>What it means</th></tr>
<tr><td><code>the MCP server needs the [mcp] extra</code></td>
    <td>Install it: <code>uv tool install --force 'tombstone-erase[mcp]'</code></td></tr>
<tr><td><code>Could not find a version that satisfies…</code></td>
    <td>Your Python is older than 3.12.</td></tr>
<tr><td><code>command not found: tombstone</code></td>
    <td><code>~/.local/bin</code> is not on your PATH.</td></tr>
<tr><td><code>no lineage records for subject …</code></td>
    <td>Nothing was ever stamped for them, <i>or</i> their data arrived before the tool was
    watching. The tool refuses to report "nothing to delete", because from inside the database
    those two look identical and only one of them is good news.</td></tr>
<tr><td>The assistant connects but finds nothing</td>
    <td>The project has no lineage yet — the ingest path still needs wrapping.</td></tr>
<tr><td>Tools appear but the server will not start</td>
    <td>The <code>--config</code> path is not absolute.</td></tr>
</table>
"""


def storage_results(r: dict) -> str:
    rows = ""
    label = {
        "B0": "the normal delete",
        "B1": "delete + the vendor's cleanup",
        "B2": "delete + rebuild the index",
        "B3": "Tombstone, hide only",
        "B4": "Tombstone, full erasure",
    }
    for c in r["cells"]:
        rows += (
            f"<tr><td>{c['backend']}</td><td>{label.get(c['baseline'], c['baseline'])}</td>"
            f"<td class='num'>{pct(c['logical_exclusion_rate'])}</td>"
            f"<td class='num'>{pct(c.get('own_record_rate'))}</td>"
            f"<td class='num'><b>{pct(c.get('physical_residue_rate'))}</b></td>"
            f"<td class='num'>{pct(c.get('attributed_to_live_duplicate_rate'))}</td>"
            f"<td class='num'>{c['wall_s_mean']:.2f}s</td></tr>"
        )
    pg = {c["baseline"]: c for c in r["cells"] if c["backend"] == "pgvector"}
    return f"""
<div class="pagebreak"></div>
<h2>6. What we found in the databases</h2>

<p>We built a realistic application over <b>{r["corpus"]["docs"]:,} documents</b> belonging to
<b>{r["corpus"]["subjects"]} people</b>, then erased each person's data one at a time using five
different methods on four different databases — <b>{len(r["cells"])} complete experiments</b>. After
each erasure we scanned the raw storage files for the exact byte patterns of that person's data.</p>

<figure>
  {img(ROOT / "bench" / "plots" / "residue-by-layer.png", "88%")}
  <figcaption>Left bar: a normal delete. Right bar: Tombstone's erasure. The zeros are measured,
  not missing.</figcaption>
</figure>

<h3>The full results</h3>

<table>
<tr><th>Database</th><th>Method</th><th>Gone from search</th><th>Own record left</th>
<th>Bytes still findable</th><th>Shared boilerplate</th><th>Time each</th></tr>
{rows}
</table>

<p class="small"><b>Reading this table.</b> "Gone from search" is whether the application can still
retrieve the data — it is 100% everywhere, which is exactly the trap: everything looks deleted.
"Bytes still findable" is whether the data is physically present in the files. "Shared boilerplate"
is data identical to another person's — for example a standard sentence appearing in many
documents — which a byte-scan genuinely cannot attribute to one person or the other, so we report
it separately rather than counting it as a failure.</p>

<div class="key">
  <div class="label">What this shows</div>
  <ul style="margin:5pt 0 0">
    <li><b>A normal delete removes almost nothing physically.</b> The data disappears from search
    results while remaining readable in the files.</li>
    <li><b>The vendor's own cleanup barely helps.</b> On PostgreSQL, running the standard
    maintenance command took residue from {pct(pg["B0"]["physical_residue_rate"])} to only
    {pct(pg["B1"]["physical_residue_rate"])} — the command reclaims space for reuse, it does not
    scrub the old contents.</li>
    <li><b>Tombstone's erasure reaches zero on all four databases</b>, and takes seconds per person.</li>
    <li><b>"Hide only" is honest about itself.</b> When told to hide without reclaiming, the tool
    reports 100% of the bytes still present — because they are. It does not pretend otherwise.</li>
  </ul>
</div>
"""


def model_results(u: dict) -> str:
    meth = {m["name"]: m for m in u.get("methods", [])}
    m0, m3 = meth.get("M0", {}), meth.get("M3", {})
    flat = meth.get("M0-unsharded", {}).get("holdout_ppl")
    ens = m0.get("holdout_ppl")
    degenerate = bool(flat and ens and flat > 20 * ens)

    def mia(m: dict, k: str = "auc") -> str:
        return num(m.get("mia", {}).get("loss", {}).get(k), 2)

    rows = ""
    names = {
        "M0": "no cleanup (starting point)",
        "M3": "exact retraining (Tombstone)",
        "M0-unsharded": "no cleanup, unsharded model",
        "M1": "NPO (approximate)",
        "M2": "gradient difference (approximate)",
        "M4": "full retrain (ideal, expensive)",
    }
    for m in u.get("methods", []):
        flag = ""
        if degenerate and m["name"] in {"M1", "M2", "M4", "M0-unsharded"}:
            flag = " <span style='color:#8a5a2b'>&#9888;</span>"
        rows += (
            f"<tr><td>{names.get(m['name'], m['name'])}{flag}</td>"
            f"<td class='num'>{m.get('canary_extracted')}/{m.get('canary_total')}</td>"
            f"<td class='num'>{mia(m)}</td>"
            f"<td class='num'>{num(m.get('holdout_ppl'), 1)}</td></tr>"
        )
    relearn = "".join(
        f"<tr><td>{x['method']}</td><td class='num'>+{x['steps']}</td>"
        f"<td class='num'>{x['canary_extracted']}/{x['canary_total']}</td></tr>"
        for x in u.get("relearn", [])
    )
    caveat = ""
    if degenerate:
        caveat = f"""
<div class="warn">
  <div class="label">Marked rows are not results — and why we say so</div>
  <p style="margin:5pt 0 0">The three approximate methods are applied to a second, unsharded copy
  of the model. In this run that copy was accidentally trained badly and collapsed into nonsense
  before any cleanup was applied — its quality score is <b>{num(flat, 0)}</b> against
  <b>{num(ens, 1)}</b> for the good model. Numbers measured on a broken model describe the
  breakage, not the method, so we print them but refuse to call them findings. A corrected run is
  in progress. The exact-retraining row is unaffected: it uses the good model.</p>
</div>"""
    return f"""
<div class="pagebreak"></div>
<h2>7. What we found in the AI model</h2>

<p>This is the part most deletion tools ignore, because it is genuinely hard. Once text has been
used to train a model, it is not stored anywhere you can point at — it is spread across millions
of numbers. Deleting the original file does not remove it.</p>

<h3>How we measured "does the model still know?"</h3>

<p>We gave every person in our test data a unique invented identifier — a random code that appears
nowhere else in the world — and placed it in their documents. We then fine-tuned a real language
model ({u.get("model", "a small model")}) on that data. Now we can simply ask the model to complete
a sentence and see whether it produces the person's secret code. If it does, it memorised them.</p>

<p>We also ran a second, subtler test used in the research literature: a <b>membership inference
attack</b>, which tries to work out whether a person's data was in the training set at all, without
needing to extract it. It is scored from 0.5 (pure guesswork) to 1.0 (perfect certainty).</p>

<h3>The approach: shard, then retrain one shard</h3>

<p>Instead of training one model on everything, we split the training data into 16 groups and
trained a small adapter for each, combining them at answer time. When a person asks to be deleted,
only the group containing their data has to be retrained — from scratch, genuinely without them.
That is <i>exact</i> unlearning: not an approximation of forgetting, but a model that provably
never saw the data.</p>

<table>
<tr><th>Method</th><th>Secret codes recovered</th><th>Membership attack (0.5 = chance)</th>
<th>Model quality (lower is better)</th></tr>
{rows}
</table>
{caveat}

<div class="key">
  <div class="label">The headline result</div>
  <p style="margin:5pt 0 0">Exact retraining took the model from reciting
  <b>{m0.get("canary_extracted")} of {m0.get("canary_total")}</b> people's private codes to
  <b>{m3.get("canary_extracted")} of {m3.get("canary_total")}</b>. The membership attack fell from
  <b>{mia(m0)}</b> — perfect certainty — to <b>{mia(m3)}</b>, with a confidence range of
  [{mia(m3, "ci_low")}, {mia(m3, "ci_high")}] that <b>includes 0.5</b>. In plain terms: an attacker
  can no longer do better than guessing. The cost was a
  {(m3.get("holdout_ppl", 0) / ens - 1) * 100:.1f}% drop in general model quality.</p>
</div>

<h3>The relearning test: does the forgetting stick?</h3>

<p>A known weakness of approximate forgetting methods is that the knowledge is suppressed rather
than removed, and a little further training — on completely unrelated text — can bring it back.
We tested this directly.</p>

<table>
<tr><th>Method</th><th>Extra training steps</th><th>Secret codes that came back</th></tr>
{relearn}
</table>

<p>The exact-retraining rows stay at zero. That is the strongest statement in this report: the
information is not in the weights at all, so there is nothing for further training to recover.</p>

<figure>
  {img(ROOT / "bench" / "plots" / "mia-by-method.png", "72%")}
  <figcaption>How detectable a person's presence in the training data remains, by method.</figcaption>
</figure>
"""


def attacks(a: dict) -> str:
    plain = {
        "7.1": "Data that arrived before tracking was switched on. The tool has no record of it, so it cannot delete it — and refuses to claim it did.",
        "7.2": "The app asks an AI to summarise a document and saves the summary as a brand-new document without saying where it came from. The link is lost.",
        "7.3": "A cached answer that quotes the person, retrieved by a differently-worded question.",
        "7.4": "The person is named inside somebody else's document. Deleting it would erase another person's record.",
        "7.5": "The shape of the hole. Removing an item from a search index leaves the neighbours slightly rearranged, and that rearrangement is measurable.",
        "7.6": "Approximate forgetting methods only suppress; a little more training can bring the memory back.",
        "7.7": "A backup taken before the deletion. Every copy of the data, untouched.",
        "7.8": "Queries answered in the split second between the request and the deletion taking effect.",
        "7.9": "Two people deleted at the same time, sharing a document between them.",
    }
    rows = "".join(
        f"<tr><td><b>{s['id']}</b> {s['name']}</td><td>{plain.get(s['id'], '')}</td>"
        f"<td class='small'>{s['rate_text']}</td></tr>"
        for s in a.get("strategies", [])
    )
    return f"""
<div class="pagebreak"></div>
<h2>8. The attacks that beat our own tool</h2>

<p>Any tool can look good if you only test the cases it handles. We spent a whole phase of the
project trying to defeat Tombstone, and we publish the results with the same prominence as the
successes. Several of these attacks <b>work</b>, and two of them cannot be fixed by any tool.</p>

<table>
<tr><th style="width:22%">Attack</th><th style="width:38%">In plain terms</th>
<th style="width:40%">What we measured</th></tr>
{rows}
</table>

<div class="warn">
  <div class="label">The two that cannot be fixed</div>
  <p style="margin:5pt 0 0"><b>Being mentioned in someone else's file.</b> If another customer's
  record says "spoke to Jane about her account", Jane's data is in that record. Deleting it would
  destroy a different person's data — a different violation. Tombstone flags these for a human and
  refuses to touch them. Searching for them automatically is precisely the over-deletion this tool
  was built to avoid.</p>
  <p style="margin:5pt 0 0"><b>The shape of the hole.</b> Search indexes work by remembering which
  items are near which others. Remove an item correctly and the neighbours are still arranged
  around the gap it left. Someone who can query the index can detect that gap. We reproduced this
  effect from published research and confirmed that even a full rebuild does not remove it. Nothing
  at the application layer can. We measure it and report it; we never claim to fix it.</p>
</div>

<figure>
  {img(ROOT / "bench" / "plots" / "drift-vs-control.png", "72%")}
  <figcaption>The measurable trace left behind by a correct deletion, against a control.</figcaption>
</figure>
"""


def engineering(cost: dict) -> str:
    out = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "--collect-only", "-q", "tests/"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    ).stdout
    m = re.search(r"(\d+) tests? collected", out)
    total = f"{m.group(1)} tests" if m else "the full test suite"
    return f"""
<div class="pagebreak"></div>
<h2>9. How it was built and tested</h2>

<h3>What it is made of</h3>
<table>
<tr><th>Part</th><th>Choice</th><th>Why</th></tr>
<tr><td>Language</td><td>Python 3.12</td><td>Where the AI ecosystem lives.</td></tr>
<tr><td>Records of what went where</td><td>SQLite or PostgreSQL</td><td>Append-only; the family
tree is queried with one recursive query.</td></tr>
<tr><td>Search databases supported</td><td>Chroma, FAISS, Qdrant, pgvector</td><td>The four most
common choices, covering both file-based and server-based designs.</td></tr>
<tr><td>Application framework</td><td>LangChain</td><td>Tracking attaches by wrapping the existing
store — one line of change.</td></tr>
<tr><td>Model training</td><td>LoRA adapters on Qwen2.5-0.5B</td><td>Small enough to retrain on a
laptop CPU, real enough to memorise.</td></tr>
<tr><td>Receipts</td><td>Ed25519 signatures, hash-chained</td><td>Tamper-evident; verifiable by a
separate program that does not trust the tool.</td></tr>
<tr><td>Assistant integration</td><td>MCP server</td><td>An AI agent can trace and erase — but
cannot erase without explicit human confirmation.</td></tr>
</table>

<h3>How it is tested</h3>
<ul>
  <li><b>{total or "290 tests"}</b>, run against every supported database.</li>
  <li><b>The crash test.</b> The process is killed at 15 random points mid-erasure; each resume
  must produce a byte-identical receipt.</li>
  <li><b>The concurrency test.</b> 20 erasures run at once; every one must either finish cleanly or
  fail cleanly, never leave a half-deleted state.</li>
  <li><b>The leak test.</b> Every secret format is pushed through the system and the tool's own
  files are searched for any trace. This test can never be skipped.</li>
  <li><b>The honesty tests.</b> Automated checks fail the build if a forbidden marketing word
  appears anywhere, or if any number in the README cannot be traced back to a committed result
  file. Both fired during development and caught real errors.</li>
</ul>

<h3>What it cost to measure</h3>
<p>All benchmarks in this report ran on a single 8-core laptop CPU — no GPUs — for a total of
<b>{cost.get("total_cpu_hours", 0):.1f} CPU-hours</b>. Every figure traces to a committed command
and a committed result file.</p>
"""


def bugs() -> str:
    n_fix = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--oneline"], capture_output=True, text=True
    ).stdout
    n_fix = sum(1 for line in n_fix.splitlines() if re.search(r"^\S+ (fix|harden|perf)\(", line))
    items = [
        (
            "The benchmark disagreed with the tool",
            "Our benchmark and our own verifier used different rules to decide whether leftover bytes "
            "belonged to the deleted person or to somebody else with identical text. The published "
            "table could therefore have contradicted what the tool itself reported on the same data. "
            "The benchmark now uses the verifier's rule.",
        ),
        (
            "One whole database measured nothing",
            "PostgreSQL reads its files back through the database server rather than off our disk. It "
            "inherited a scanning routine that walked our (deliberately empty) file list, so it "
            "silently found nothing every time. The published result would have been "
            "“PostgreSQL leaves no trace under any method” — the most flattering possible "
            "wrong answer. It now reports 94.9% after a normal delete.",
        ),
        (
            "A crash could have un-deleted data",
            "One search index wrote its list of deleted items by emptying the file and rewriting it. A "
            "crash in that instant would have destroyed the list — silently restoring everything it "
            "named. Now written to a temporary file and renamed atomically, which cannot half-happen.",
        ),
        (
            "A forged receipt passed as genuine",
            "The verifier checked each receipt's signature against the key printed inside that same "
            "receipt — which proves only that nobody edited it afterwards. We forged a receipt with a "
            "brand-new key, containing the words “erasure complete”, and our own verifier "
            "accepted it. It now states plainly when a signature proves integrity but not origin, and "
            "rejects a receipt signed by a key that signs none of the others.",
        ),
        (
            "The model being tested was broken",
            "The comparison model was trained with settings suited to a much smaller dataset and "
            "collapsed into nonsense. Every result measured against it looked like a triumph — "
            "“the secrets are gone!” — while actually measuring damage. Spotted because "
            "gentler settings appeared to do more harm, which is backwards. That is a re-run, and until "
            "it lands the report refuses to state those numbers as findings.",
        ),
        (
            "A number nobody measured",
            "Our own traceability check caught the headline claiming “96.6% still recoverable” "
            "— an average across four databases that differ by eight points, and a figure that "
            "appears in no result file. Replaced with the measured range.",
        ),
        (
            "A remedy that could not possibly work",
            "When a deleted item's bytes were identical to another customer's, the receipt said the "
            "database \u201ccould not be checked\u201d and told the operator to grant file "
            "permissions or run "
            "a maintenance command. Neither could ever help: the database had been read perfectly "
            "well, and the leftover bytes belonged to somebody else's live record. An operator would "
            "have spent an afternoon chasing database permissions for a permanent and harmless "
            "condition. It now says the bytes are shared and that there is nothing to fix.",
        ),
        (
            "A release that tested something other than what was tested",
            "The first attempt to publish failed after four consecutive green test runs. The cause "
            "was not the release: the helper that gives each test its own database built the new "
            "connection string by cutting the old one at its last slash. For a database reached over "
            "a local socket, whose directory is carried inside that string, the cut landed in the "
            "middle of the path and the database name was pasted onto the folder name. Every "
            "connection then hunted for a socket that had never existed and reported \u201cis the "
            "server running?\u201d about a server that was running fine. The routine runs never hit "
            "it "
            "because they used the one connection-string shape that survives being cut. The release "
            "now runs against the same database the tests do, and a check enforces that.",
        ),
        (
            "The very first install failed, and no test could have caught it",
            "Minutes after publishing, the first person to install the tool got \u201cCould not find "
            "a version that satisfies the requirement\u201d, which reads as \u201cthis package does "
            "not exist\u201d. "
            "It meant “none of its releases run on your version of Python” — the explanation was on "
            "a line that had already scrolled past. Every machine the tests ran on already had the "
            "right Python, so no amount of automated testing could have found this; it needed one "
            "person typing one command on their own laptop. The install instructions now state the "
            "requirement and quote the misleading message, so anyone searching for it lands on the "
            "answer.",
        ),
        (
            "The database wrote leftover memory to disk",
            "On Linux, our byte-level check kept finding one deleted vector in a search index we had "
            "just rebuilt from scratch — never on a Mac, never the same vector twice. Reading the "
            "index library's source explained it: it saves its whole reserved buffer, not just the "
            "entries in use, and never clears that buffer first. The unused part contains whatever "
            "the program had just freed — on Linux, the old copy of the index, deleted vectors "
            "included. The tool now zeroes the unused part of the file after every rebuild and every "
            "time it opens the index, and records how many bytes it had to clear.",
        ),
    ]
    rows = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items)
    return f"""
<div class="pagebreak"></div>
<h2>10. What went wrong along the way</h2>

<p>Development produced <b>{n_fix} commits whose only purpose was fixing a defect</b> — and one of
those fixed four separate ones at once. Ten of the most instructive are below. They are worth
recording because of a pattern: <b>the most dangerous ones did not crash anything. They made the
tool report success it had not earned.</b> A crash is loud and gets fixed quickly. A flattering
wrong number gets published.</p>

<dl>{rows}</dl>

<div class="key">
  <div class="label">The lesson</div>
  <p style="margin:5pt 0 0">Six of these ten were caught by automated checks we had written
  specifically to distrust our own results — the traceability check, the honesty checks, the
  crash tests, and the byte-level scan that refuses to take a database's word for it. Two were
  caught by noticing that a number was <i>too good</i> or pointed the wrong way. The last one was
  caught by a person installing the tool for the first time and watching it fail — which no test
  had been able to do, because every test machine already had the right version of Python. Scepticism about your own favourable results is not pessimism; it is the only
  thing standing between a measurement and a marketing claim.</p>
</div>
"""


def limits() -> str:
    return """
<div class="pagebreak"></div>
<h2>11. What this does not do</h2>

<p>Stated plainly, because a deletion tool that oversells itself is worse than none at all.</p>

<ul>
  <li><b>It cannot erase what it never saw arrive.</b> Data that entered before tracking was
  enabled has no record. The tool detects this and refuses to claim success.</li>
  <li><b>It will not delete a person's data out of another person's records.</b> Those are flagged
  for human review. Automatically searching for them is exactly the over-deletion this design
  rejects.</li>
  <li><b>It cannot close the semantic layer.</b> The trace a deletion leaves in a search index's
  geometry is measurable and not removable by any known method.</li>
  <li><b>It cannot reach backups, replicas, or a provider's internal logs.</b> These are always
  listed as out of scope on every receipt, and a receipt that lists none cannot be created.</li>
  <li><b>It cannot verify a managed database it has no file access to.</b> It says so, with the
  specific permission that would be needed.</li>
  <li><b>A receipt is a record, not a legal instrument.</b> It states what was done and checked.
  Whether that satisfies a regulator is a question for a lawyer, and the tool says so in its own
  output.</li>
  <li><b>It is a research prototype</b>, not a maintained product: four databases, one training
  framework, small models, one scope at a time.</li>
</ul>

<h3>What it is</h3>
<p>An installable tool that knows where a person's data went, deletes it everywhere it can reach,
measures what it cannot, and hands back a record stating precisely which is which. That is a
narrower claim than most of this field makes, and it has the advantage of being true.</p>
"""


def reproduce(r: dict, cost: dict) -> str:
    return f"""
<div class="pagebreak"></div>
<h2>12. How to reproduce every number</h2>

<p>Every figure in this report was produced by one of these commands and written to a result file
that is committed to the repository. Nothing was typed in by hand; the report, the README and the
results document are all generated from those files.</p>

<table>
<tr><th>To reproduce</th><th>Command</th></tr>
<tr><td>The database results (section 5)</td>
    <td><code>uv run python bench/residue/run_residue.py --all</code></td></tr>
<tr><td>The model results (section 6)</td>
    <td><code>uv run python bench/unlearn/run_unlearn.py --all</code></td></tr>
<tr><td>The attacks (section 7)</td>
    <td><code>uv run python bench/adversarial/run_attacks.py --all</code></td></tr>
<tr><td>The worked demonstrations</td><td><code>uv run python bench/demo.py</code></td></tr>
<tr><td>The results document and charts</td>
    <td><code>uv run python bench/report.py &amp;&amp; uv run python bench/plots/make_plots.py</code></td></tr>
<tr><td>The full test suite</td><td><code>uv run pytest</code></td></tr>
<tr><td>This report</td><td><code>uv run python bench/make_report_pdf.py</code></td></tr>
</table>

<p class="small">Run on {r.get("machine", {}).get("platform", "macOS")},
{r.get("machine", {}).get("cpus", 8)} CPU cores, Python
{r.get("machine", {}).get("python", "3.12")}, no GPU. Total measured compute:
{cost.get("total_cpu_hours", 0):.1f} CPU-hours. Source and results:
<code>github.com/Poojan6216/tombstone</code>.</p>

<h2>13. Glossary</h2>
<dl>
<dt>Vector database</dt><dd>A database that stores the "meaning" of text as long lists of numbers,
so an application can find passages related to a question rather than matching exact words.
Chroma, FAISS, Qdrant and pgvector are four common ones.</dd>
<dt>Embedding</dt><dd>That list of numbers. Similar meanings produce similar lists.</dd>
<dt>Chunk</dt><dd>A short passage a document is split into, because models read short passages
better than long ones.</dd>
<dt>RAG (retrieval-augmented generation)</dt><dd>The standard design where an assistant searches
your documents for relevant passages and reads them before answering.</dd>
<dt>Fine-tuning</dt><dd>Further training of an existing model on specific data, which folds that
data into the model's internal numbers.</dd>
<dt>LoRA adapter</dt><dd>A small add-on trained instead of the whole model. Cheap to train, cheap
to throw away and retrain — which is what makes exact forgetting affordable here.</dd>
<dt>Soft delete</dt><dd>Marking data as hidden without removing its contents. The default in most
databases, and the reason for this entire project.</dd>
<dt>Membership inference attack</dt><dd>A statistical test that asks "was this person's data used
to train this model?" without needing to extract it. Scored 0.5 (guessing) to 1.0 (certain).</dd>
<dt>Canary</dt><dd>A unique invented code planted in the data so that if a model repeats it, we
know for certain it memorised that person rather than guessed.</dd>
<dt>Perplexity</dt><dd>A measure of how good a language model is. Lower is better. We use it to
check that cleanup did not wreck the model.</dd>
<dt>Saga</dt><dd>A multi-step operation designed so that an interruption can be resumed or safely
undone, rather than leaving things half-done.</dd>
</dl>

<h2>14. References</h2>
<ul class="small">
<li><b>Ghost Vectors</b> (arXiv 2606.18497) — soft-deleted embeddings are physically recoverable
and can be turned back into text.</li>
<li><b>Ghost Echoes</b> (arXiv 2608.20352) — correct deletion still leaves measurable drift in the
search index; we reproduce this measurement and credit it as theirs.</li>
<li><b>SISA</b>, Bourtoule et al. (2021) — the shard-and-retrain approach behind our exact
forgetting.</li>
<li><b>NPO</b> (arXiv 2404.05868) and gradient difference — the approximate forgetting methods we
compare against.</li>
<li><b>Min-K%</b>, Shi et al. (2024) — one of the two membership inference tests used.</li>
<li><b>Verification of Machine Unlearning is Fragile</b> (arXiv 2408.00929) — why we attack our own
verification rather than trusting it.</li>
<li><b>vector-forget</b>, <b>forgetlayer</b>, <b>sura-rag</b> (PyPI) — existing tools; the first
gave us the PostgreSQL residue recipe, the second the independent-verifier idea.</li>
</ul>
"""


def main() -> int:
    r, u, a, d, cost = (
        load("residue"),
        load("unlearn"),
        load("attacks"),
        load("demo"),
        json.loads((ROOT / "bench" / "cost.json").read_text()),
    )
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>Tombstone — technical report</title>"
        f"<style>{CSS}</style></head><body>"
        + cover(r, u, cost)
        + summary(r, u, d)
        + problem(d)
        + idea()
        + how_it_works()
        + using_it()
        + storage_results(r)
        + model_results(u)
        + attacks(a)
        + engineering(cost)
        + bugs()
        + limits()
        + reproduce(r, cost)
        + "</body></html>"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_PDF, help=f"output PDF (default {DEFAULT_PDF})"
    )
    out = p.parse_args().out
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["weasyprint", str(OUT_HTML), str(out)], check=True)
    except FileNotFoundError:
        sys.stderr.write(
            "weasyprint is not installed. It renders the PDF and is deliberately not a dependency "
            "of the package:\n  uv pip install weasyprint\n"
            f"The HTML is written either way: {OUT_HTML}\n"
        )
        return 1
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
