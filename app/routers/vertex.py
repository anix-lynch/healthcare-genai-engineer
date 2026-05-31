"""GET /vertex — Vertex AI ER Insight Console.

Doctor-facing 4-pane command console:
  Pane 1: INPUT  · query + presets + method + ERState
  Pane 2: ANSWER · ESI badge (traffic light) + grounding + answer + prediction signals
  Pane 3: EVIDENCE · citations + red flags + vote breakdown
  Bottom: NEXT ACTIONS chips + trace waterfall

Calls POST /v1/ask (same origin) and populates all panes live.
No build step, no React, no new deps. HTML + vanilla JS + Tailwind CDN inline.
Pattern: mirrors app/routers/web.py exactly.
"""
from __future__ import annotations
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ER Insight Console · Vertex AI</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --canvas:#fafbfc; --panel:#ffffff; --border:#e6eaee;
      --ink-1:#0e1726; --ink-2:#324054; --ink-3:#6b7a90;
      --accent:#2a6fa8; --accent-bg:#eaf3fb;
      --ok:#16a34a; --warn:#d97706;
      --danger:#dc2626;
      --shadow:0 4px 14px rgba(20,30,50,0.06);
    }
    html,body { font-family:'Inter',system-ui,sans-serif; background:var(--canvas); color:var(--ink-1); }
    .mono { font-family:'JetBrains Mono',ui-monospace,monospace; }
    .panel { background:var(--panel); border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow); }
    .toggle-btn { transition:all 160ms ease; }
    .toggle-btn.active { background:var(--accent); color:#fff; }
    .cite-card { animation:fadeIn 240ms ease both; }
    @keyframes fadeIn { from{opacity:0;transform:translateY(4px);} to{opacity:1;transform:none;} }
    .bar { transition:width 400ms ease-out; }
    .shimmer { background:linear-gradient(90deg,#eef2f6 0%,#dfe5ec 50%,#eef2f6 100%); background-size:200% 100%; animation:shimmer 1.4s infinite; }
    @keyframes shimmer { 0%{background-position:200% 0;} 100%{background-position:-200% 0;} }
    /* ESI tier colors */
    .esi-1 { background:#fef2f2; border-color:#dc2626; color:#dc2626; }
    .esi-2 { background:#fff7ed; border-color:#ea580c; color:#ea580c; }
    .esi-3 { background:#fefce8; border-color:#ca8a04; color:#ca8a04; }
    .esi-4 { background:#f0fdf4; border-color:#16a34a; color:#16a34a; }
    .esi-5 { background:#eff6ff; border-color:#2563eb; color:#2563eb; }
    .chip-action { transition:all 120ms ease; }
    .chip-action:hover { transform:translateY(-1px); }
    textarea:focus { outline:none; border-color:var(--accent); box-shadow:0 0 0 3px rgba(42,111,168,0.12); }
  </style>
</head>
<body class="min-h-screen">

  <!-- HEADER -->
  <header class="sticky top-0 bg-white/95 backdrop-blur border-b border-[var(--border)] z-10">
    <div class="max-w-[1400px] mx-auto px-6 py-3 flex items-center gap-4 flex-wrap">
      <span class="text-xl">🏥</span>
      <div class="flex-1 min-w-[200px]">
        <h1 class="font-semibold text-[15px] text-[var(--ink-1)]">ER Insight Console</h1>
        <p class="text-[11px] text-[var(--ink-3)]">Vertex AI Search · Grounded answers · ESI triage · Citation-valid</p>
      </div>
      <div id="headerEsiBadge" class="hidden items-center gap-1.5 text-[12px] font-semibold px-2.5 py-1 rounded-lg border-2 mono"></div>
      <div id="headerGrounding" class="hidden items-center gap-1.5 text-[12px] font-medium"></div>
      <div id="headerCiteCount" class="hidden text-[11px] text-[var(--ink-3)] mono"></div>
      <a href="/" class="text-[12px] text-[var(--ink-3)] hover:text-[var(--accent)]">← RAG Eval</a>
      <a href="/docs" class="text-[12px] text-[var(--ink-3)] hover:text-[var(--accent)]">⚙️ /docs</a>
    </div>
  </header>

  <!-- CASE BAR -->
  <div class="max-w-[1400px] mx-auto px-6 pt-4">
    <div class="flex items-center gap-4 text-[12px] text-[var(--ink-3)] mono bg-white border border-[var(--border)] rounded-lg px-4 py-2.5">
      <span>Case: <span id="caseId" class="text-[var(--ink-2)] font-medium">ER-2026-—</span></span>
      <span>·</span>
      <span>Dept: <span class="text-[var(--ink-2)]">ER / General</span></span>
      <span>·</span>
      <span>Mode: <span id="caseMode" class="text-[var(--ink-2)]">Vertex Search</span></span>
      <span>·</span>
      <span id="caseTimestamp" class="ml-auto"></span>
    </div>
  </div>

  <!-- MAIN GRID: 3 columns -->
  <main class="max-w-[1400px] mx-auto px-6 py-5 grid grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_300px] gap-5 items-start">

    <!-- ═══ PANE 1: INPUT ═══ -->
    <section class="panel p-5 space-y-4">
      <div class="flex items-center gap-2">
        <span>🔍</span>
        <h2 class="font-semibold text-[14px]">INPUT</h2>
      </div>

      <!-- Query -->
      <div>
        <label class="text-[11px] text-[var(--ink-3)] uppercase tracking-wide font-medium block mb-1.5">Chief complaint / clinical query</label>
        <textarea id="query" rows="4" placeholder="e.g. Stroke vs migraine — unilateral headache, visual aura, no fever…"
          class="w-full border border-[var(--border)] rounded-lg px-3 py-2.5 text-[13px] resize-none"></textarea>
      </div>

      <!-- Preset chips -->
      <div>
        <p class="text-[11px] text-[var(--ink-3)] mb-2 font-medium">Quick prompts</p>
        <div class="space-y-1.5">
          <button class="preset block w-full text-left text-[12px] text-[var(--ink-2)] hover:text-[var(--accent)] py-1 px-2 rounded hover:bg-[var(--accent-bg)] transition"
            data-q="Stroke vs migraine unilateral headache visual aura">▸ Stroke vs migraine?</button>
          <button class="preset block w-full text-left text-[12px] text-[var(--ink-2)] hover:text-[var(--accent)] py-1 px-2 rounded hover:bg-[var(--accent-bg)] transition"
            data-q="62yo male chest pain hypertension rule out MI">▸ Chest pain — rule out MI</button>
          <button class="preset block w-full text-left text-[12px] text-[var(--ink-2)] hover:text-[var(--accent)] py-1 px-2 rounded hover:bg-[var(--accent-bg)] transition"
            data-q="sepsis workup fever tachycardia suspected infection">▸ Sepsis workup</button>
          <button class="preset block w-full text-left text-[12px] text-[var(--ink-2)] hover:text-[var(--accent)] py-1 px-2 rounded hover:bg-[var(--accent-bg)] transition"
            data-q="DKA presentation vomiting high glucose ketones">▸ DKA presentation</button>
          <button class="preset block w-full text-left text-[12px] text-[var(--ink-2)] hover:text-[var(--accent)] py-1 px-2 rounded hover:bg-[var(--accent-bg)] transition"
            data-q="triage escalation patient deteriorating vitals worsening">▸ Triage escalation</button>
        </div>
      </div>

      <!-- Retrieval method -->
      <div>
        <p class="text-[11px] text-[var(--ink-3)] mb-1.5 font-medium uppercase tracking-wide">Retrieval mode</p>
        <div class="inline-flex rounded-lg border border-[var(--border)] overflow-hidden text-[11px] w-full">
          <button class="toggle-btn active flex-1 py-1.5 text-center" data-method="bm25" title="Keyword BM25">BM25</button>
          <button class="toggle-btn flex-1 py-1.5 text-center border-l border-[var(--border)]" data-method="dense" title="Semantic dense">Dense</button>
          <button class="toggle-btn flex-1 py-1.5 text-center border-l border-[var(--border)]" data-method="hybrid" title="BM25 + Dense (RRF)">Hybrid</button>
        </div>
      </div>

      <!-- ERState (collapsible) -->
      <div>
        <button id="erStateToggle" class="flex items-center gap-1.5 text-[11px] text-[var(--ink-3)] hover:text-[var(--accent)] font-medium w-full">
          <span id="erStateArrow" class="transition-transform">▶</span>
          <span>ER Operational State</span>
          <span class="text-[10px] italic ml-auto">(optional)</span>
        </button>
        <div id="erStateForm" class="hidden mt-2 space-y-2">
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-[10px] text-[var(--ink-3)] block mb-0.5">Queue length</label>
              <input id="erQueue" type="number" min="0" placeholder="—" class="w-full border border-[var(--border)] rounded px-2 py-1 text-[12px] mono">
            </div>
            <div>
              <label class="text-[10px] text-[var(--ink-3)] block mb-0.5">Beds available</label>
              <input id="erBeds" type="number" min="0" placeholder="—" class="w-full border border-[var(--border)] rounded px-2 py-1 text-[12px] mono">
            </div>
            <div>
              <label class="text-[10px] text-[var(--ink-3)] block mb-0.5">Occupancy %</label>
              <input id="erOccupancy" type="number" min="0" max="100" placeholder="—" class="w-full border border-[var(--border)] rounded px-2 py-1 text-[12px] mono">
            </div>
            <div>
              <label class="text-[10px] text-[var(--ink-3)] block mb-0.5">Avg wait (min)</label>
              <input id="erWait" type="number" min="0" placeholder="—" class="w-full border border-[var(--border)] rounded px-2 py-1 text-[12px] mono">
            </div>
          </div>
        </div>
      </div>

      <!-- Ask button -->
      <button id="askBtn"
        class="w-full bg-[var(--accent)] text-white text-[13px] font-semibold py-2.5 rounded-lg hover:opacity-90 active:scale-[0.99] transition flex items-center justify-center gap-2">
        <span id="askBtnLabel">Ask  ▶</span>
      </button>

      <p id="errorMsg" class="hidden text-[12px] text-[var(--danger)] rounded-lg bg-red-50 border border-red-200 px-3 py-2"></p>
    </section>

    <!-- ═══ PANE 2: ANSWER ═══ -->
    <section class="panel p-5 space-y-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span>🩺</span>
          <h2 class="font-semibold text-[14px]">ANSWER</h2>
        </div>
        <div id="warningBadge" class="hidden text-[10px] text-[var(--warn)] bg-amber-50 border border-amber-200 rounded px-2 py-0.5 mono"></div>
      </div>

      <!-- ESI tier block -->
      <div id="esiBlock" class="rounded-xl border-2 p-4 flex items-center gap-4 esi-5">
        <div class="flex flex-col items-center min-w-[72px]">
          <span class="text-[10px] uppercase tracking-wider opacity-70 font-medium">ESI Tier</span>
          <span id="esiTier" class="text-[42px] font-bold leading-none">—</span>
          <span id="esiLabel" class="text-[11px] font-bold uppercase tracking-widest mt-0.5">—</span>
        </div>
        <div class="flex-1 space-y-1.5">
          <div class="flex items-center gap-2">
            <span class="text-[10px] uppercase tracking-wide opacity-70 w-[80px]">Confidence</span>
            <div class="flex-1 h-2 bg-white/50 rounded">
              <div id="esiConfBar" class="bar h-2 rounded bg-current" style="width:0%;"></div>
            </div>
            <span id="esiConfText" class="text-[12px] mono font-semibold w-[40px]">—</span>
          </div>
          <p id="esiVotesText" class="text-[11px] opacity-80"><b>Votes:</b> —</p>
          <div id="esiDisagreement" class="hidden text-[10px] bg-amber-100 border border-amber-300 rounded px-2 py-1 text-amber-800">
            ⚠️ Rule/RAG disagreement — safety floor applied
          </div>
        </div>
      </div>

      <!-- Grounding chip -->
      <div class="flex items-center gap-3 flex-wrap">
        <div id="groundingChip" class="text-[12px] font-semibold px-3 py-1 rounded-full border">—</div>
        <div id="triageLevelChip" class="hidden text-[11px] font-bold px-3 py-1 rounded-full border mono"></div>
      </div>

      <!-- Grounded answer -->
      <div>
        <p class="text-[11px] uppercase tracking-wide text-[var(--ink-3)] font-medium mb-2">Grounded Answer</p>
        <div id="answerText" class="text-[13px] leading-relaxed text-[var(--ink-1)] rounded-lg bg-[var(--canvas)] border border-[var(--border)] px-4 py-3 min-h-[80px]">
          <span class="text-[var(--ink-3)] italic">Waiting for query…</span>
        </div>
      </div>

      <!-- Explanation for human -->
      <div id="explanationWrap" class="hidden">
        <p class="text-[11px] uppercase tracking-wide text-[var(--ink-3)] font-medium mb-1">Decision rationale</p>
        <p id="explanationText" class="text-[12px] text-[var(--ink-2)] italic leading-relaxed"></p>
      </div>

      <!-- Prediction signals -->
      <div id="predictionBlock" class="hidden">
        <p class="text-[11px] uppercase tracking-wide text-[var(--ink-3)] font-medium mb-2">Prediction signals</p>
        <div class="grid grid-cols-2 gap-2">
          <div class="rounded-lg border border-[var(--border)] bg-[var(--canvas)] px-3 py-2">
            <p class="text-[10px] text-[var(--ink-3)] uppercase tracking-wide">Risk level</p>
            <p id="predRisk" class="text-[14px] font-bold mt-0.5">—</p>
          </div>
          <div class="rounded-lg border border-[var(--border)] bg-[var(--canvas)] px-3 py-2">
            <p class="text-[10px] text-[var(--ink-3)] uppercase tracking-wide">Deterioration</p>
            <p id="predDet" class="text-[14px] font-bold mt-0.5">—</p>
          </div>
          <div class="rounded-lg border border-[var(--border)] bg-[var(--canvas)] px-3 py-2">
            <p class="text-[10px] text-[var(--ink-3)] uppercase tracking-wide">Bed pressure</p>
            <p id="predBed" class="text-[14px] font-bold mt-0.5">—</p>
          </div>
          <div class="rounded-lg border border-[var(--border)] bg-[var(--canvas)] px-3 py-2">
            <p class="text-[10px] text-[var(--ink-3)] uppercase tracking-wide">Predicted LOS</p>
            <p id="predLos" class="text-[14px] font-bold mt-0.5">—</p>
          </div>
        </div>
      </div>

      <!-- Decision basis -->
      <div id="decisionBasisBlock" class="hidden">
        <p class="text-[11px] uppercase tracking-wide text-[var(--ink-3)] font-medium mb-1.5">Decision basis</p>
        <ul id="decisionBasisList" class="space-y-1"></ul>
      </div>

      <!-- Agent collaboration -->
      <div id="agentCollabBlock" class="hidden">
        <div class="flex items-center justify-between gap-3 mb-2">
          <p class="text-[11px] uppercase tracking-wide text-[var(--ink-3)] font-medium">Agent handoff</p>
          <span id="agentRuntimeTag" class="text-[10px] mono text-[var(--ink-3)]"></span>
        </div>
        <div id="agentCollabSummary" class="text-[12px] text-[var(--ink-2)] mb-2"></div>
        <div id="agentCollabList" class="space-y-2"></div>
      </div>
    </section>

    <!-- ═══ PANE 3: EVIDENCE ═══ -->
    <aside class="panel p-5 space-y-4">
      <div class="flex items-center gap-2">
        <span>📋</span>
        <h2 class="font-semibold text-[14px]">EVIDENCE</h2>
      </div>

      <!-- Red flags -->
      <div id="redFlagsBlock" class="hidden">
        <p class="text-[11px] uppercase tracking-wide text-[var(--ink-3)] font-medium mb-1.5">Red flags triggered</p>
        <div id="redFlagsList" class="flex flex-wrap gap-1.5"></div>
      </div>

      <!-- Override notice -->
      <div id="overrideBlock" class="hidden rounded-lg bg-amber-50 border border-amber-200 px-3 py-2">
        <p class="text-[10px] font-semibold text-amber-800 uppercase tracking-wide mb-0.5">Safety override applied</p>
        <p id="overrideReason" class="text-[11px] text-amber-700"></p>
      </div>

      <!-- Citation cards -->
      <div>
        <p class="text-[11px] uppercase tracking-wide text-[var(--ink-3)] font-medium mb-2">Retrieved evidence</p>
        <div id="citations" class="space-y-2">
          <div class="rounded-lg border border-dashed border-[var(--border)] p-5 text-center text-[12px] text-[var(--ink-3)]">
            📂 Evidence will appear after your first query
          </div>
        </div>
      </div>

      <!-- ESI vote breakdown -->
      <div id="voteBlock" class="hidden">
        <p class="text-[11px] uppercase tracking-wide text-[var(--ink-3)] font-medium mb-1.5">RAG vote breakdown</p>
        <p id="voteText" class="text-[12px] mono text-[var(--ink-2)]"></p>
      </div>
    </aside>

  </main>

  <!-- BOTTOM: NEXT ACTIONS + TRACE -->
  <footer class="max-w-[1400px] mx-auto px-6 pb-8">
    <div class="panel p-5 space-y-4">

      <!-- Next actions -->
      <div>
        <div class="flex items-center gap-2 mb-3">
          <span>⚡</span>
          <h2 class="font-semibold text-[14px]">NEXT ACTIONS</h2>
        </div>
        <div id="nextActions" class="flex flex-wrap gap-2">
          <span class="text-[12px] text-[var(--ink-3)] italic">Recommendations will appear after your first query</span>
        </div>
      </div>

      <!-- Trace waterfall -->
      <div class="border-t border-[var(--border)] pt-4">
        <div class="flex items-center gap-2 mb-3">
          <span class="text-[11px] font-semibold text-[var(--ink-3)] uppercase tracking-wide">⚡ Live Trace</span>
          <span class="text-[10px] text-[var(--ink-3)]">· per-node latency · this request</span>
          <span id="traceTotalMs" class="ml-auto text-[12px] mono font-semibold text-[var(--ink-2)]"></span>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div class="space-y-1">
            <div class="flex items-center justify-between text-[10px] text-[var(--ink-3)] mono">
              <span>guard</span><span id="traceGuardMs">— ms</span>
            </div>
            <div class="h-2 bg-[var(--border)] rounded overflow-hidden">
              <div id="traceBarGuard" class="bar h-2 rounded" style="width:0%; background:#6b7a90;"></div>
            </div>
          </div>
          <div class="space-y-1">
            <div class="flex items-center justify-between text-[10px] text-[var(--ink-3)] mono">
              <span>retrieve</span><span id="traceRetrieveMs">— ms</span>
            </div>
            <div class="h-2 bg-[var(--border)] rounded overflow-hidden">
              <div id="traceBarRetrieve" class="bar h-2 rounded" style="width:0%; background:var(--accent);"></div>
            </div>
          </div>
          <div class="space-y-1">
            <div class="flex items-center justify-between text-[10px] text-[var(--ink-3)] mono">
              <span>generate</span><span id="traceGenerateMs">— ms</span>
            </div>
            <div class="h-2 bg-[var(--border)] rounded overflow-hidden">
              <div id="traceBarGenerate" class="bar h-2 rounded" style="width:0%; background:var(--ok);"></div>
            </div>
          </div>
        </div>
        <div class="mt-2 pt-2 border-t border-[var(--border)] flex items-center gap-3 text-[10px] text-[var(--ink-3)]">
          <span>Weave trace:</span>
          <a id="weaveLink" href="https://wandb.ai/alynch-zeroshot/healthcare-genai" target="_blank"
            class="text-[var(--accent)] hover:underline">project lobby ↗</a>
          <span id="weaveHint" class="italic">(no WANDB_API_KEY — set to enable per-request deep-link)</span>
          <span class="ml-auto mono" id="methodUsedTag"></span>
        </div>
      </div>

    </div>
  </footer>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let currentMethod = 'bm25';
let loading = false;

// ── ESI helpers ───────────────────────────────────────────────────────────
const ESI_CONFIG = {
  1: { cls:'esi-1', label:'NOW',  urgency:'CRITICAL' },
  2: { cls:'esi-2', label:'NOW',  urgency:'EMERGENT' },
  3: { cls:'esi-3', label:'SOON', urgency:'URGENT' },
  4: { cls:'esi-4', label:'WAIT', urgency:'LESS URGENT' },
  5: { cls:'esi-5', label:'WAIT', urgency:'NON-URGENT' },
};

function esiClass(tier) {
  return ESI_CONFIG[tier]?.cls ?? 'esi-5';
}
function esiLabel(tier) {
  return ESI_CONFIG[tier]?.urgency ?? '—';
}

// ── Grounding chip ─────────────────────────────────────────────────────────
function groundingChip(conf) {
  if (conf == null) return { text:'— No signal', cls:'bg-gray-100 border-gray-300 text-gray-500' };
  if (conf >= 0.7)  return { text:'🟢 Grounded',        cls:'bg-green-50 border-green-400 text-green-700' };
  if (conf >= 0.4)  return { text:'🟡 Partial Evidence', cls:'bg-yellow-50 border-yellow-400 text-yellow-700' };
  return               { text:'🔴 Weak Evidence',    cls:'bg-red-50 border-red-400 text-red-700' };
}

// ── Risk level color ───────────────────────────────────────────────────────
function riskColor(level) {
  if (level === 'high')   return 'text-red-600';
  if (level === 'medium') return 'text-yellow-600';
  return 'text-green-600';
}

// ── Source type tag — driven by backend source_type field, prefix as fallback ──
function sourceTag(sourceId, sourceType) {
  const t = (sourceType || '').toLowerCase();
  if (t === 'vid')    return { label:'media',      cls:'bg-purple-50 text-purple-700' };
  if (t === 'web')    return { label:'website',    cls:'bg-blue-50 text-blue-700' };
  if (t === 'struct') return { label:'structured', cls:'bg-teal-50 text-teal-700' };
  if (t === 'doc')    return { label:'docs',       cls:'bg-gray-50 text-gray-600' };
  // fallback: infer from source_id prefix (legacy / unknown)
  const s = (sourceId || '').toLowerCase();
  if (s.startsWith('vid') || s.startsWith('img') || s.startsWith('media')) return { label:'media',      cls:'bg-purple-50 text-purple-700' };
  if (s.startsWith('web'))                                                   return { label:'website',    cls:'bg-blue-50 text-blue-700' };
  if (s.startsWith('struct') || s.startsWith('row'))                        return { label:'structured', cls:'bg-teal-50 text-teal-700' };
  return                                                                            { label:'docs',       cls:'bg-gray-50 text-gray-600' };
}

// ── Case timestamp ─────────────────────────────────────────────────────────
function nowTs() {
  const d = new Date();
  return d.toISOString().slice(0,16).replace('T',' ') + ' UTC';
}

// ── Method toggle ──────────────────────────────────────────────────────────
document.querySelectorAll('[data-method]').forEach(btn => {
  btn.addEventListener('click', () => {
    currentMethod = btn.dataset.method;
    document.querySelectorAll('[data-method]').forEach(b => b.classList.remove('active'));
    document.querySelectorAll(`[data-method="${currentMethod}"]`).forEach(b => b.classList.add('active'));
    document.getElementById('caseMode').textContent = `Vertex Search · ${currentMethod.toUpperCase()}`;
  });
});

// ── ERState toggle ─────────────────────────────────────────────────────────
document.getElementById('erStateToggle').addEventListener('click', () => {
  const form = document.getElementById('erStateForm');
  const arrow = document.getElementById('erStateArrow');
  const hidden = form.classList.toggle('hidden');
  arrow.style.transform = hidden ? '' : 'rotate(90deg)';
});

// ── Preset chips ───────────────────────────────────────────────────────────
document.querySelectorAll('.preset').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('query').value = btn.dataset.q;
    document.getElementById('query').focus();
  });
});

// ── Build ERState payload ──────────────────────────────────────────────────
function buildErState() {
  const q = document.getElementById('erQueue').value;
  const b = document.getElementById('erBeds').value;
  const o = document.getElementById('erOccupancy').value;
  const w = document.getElementById('erWait').value;
  if (!q && !b && !o && !w) return null;
  const state = {};
  if (q) state.queue_length = parseInt(q);
  if (b) state.available_beds = parseInt(b);
  if (o) state.occupancy_pct = parseFloat(o);
  if (w) state.avg_wait_minutes = parseInt(w);
  return state;
}

// ── Render trace waterfall ─────────────────────────────────────────────────
function renderTrace(guard_ms, retrieve_ms, generate_ms, total_ms) {
  const total = guard_ms + retrieve_ms + generate_ms || 1;
  document.getElementById('traceBarGuard').style.width    = (guard_ms    / total * 100) + '%';
  document.getElementById('traceBarRetrieve').style.width = (retrieve_ms / total * 100) + '%';
  document.getElementById('traceBarGenerate').style.width = (generate_ms / total * 100) + '%';
  document.getElementById('traceGuardMs').textContent    = guard_ms    + ' ms';
  document.getElementById('traceRetrieveMs').textContent = retrieve_ms + ' ms';
  document.getElementById('traceGenerateMs').textContent = generate_ms + ' ms';
  document.getElementById('traceTotalMs').textContent    = `total ${total_ms} ms`;
}

// ── Render citation cards ──────────────────────────────────────────────────
function renderCitations(citations) {
  const el = document.getElementById('citations');
  if (!citations || !citations.length) {
    el.innerHTML = '<div class="text-[12px] text-[var(--ink-3)] italic">No citations returned.</div>';
    return;
  }
  el.innerHTML = citations.map((c, i) => {
    const pct = Math.round((c.similarity || 0) * 100);
    const tag = sourceTag(c.source_id, c.source_type);
    return `
      <div class="cite-card rounded-lg border border-[var(--border)] p-3 space-y-1.5" style="animation-delay:${i*60}ms">
        <div class="flex items-center justify-between">
          <span class="text-[11px] mono font-semibold text-[var(--accent)] bg-[var(--accent-bg)] px-2 py-0.5 rounded">${c.source_id}</span>
          <span class="text-[10px] font-medium px-1.5 py-0.5 rounded ${tag.cls}">${tag.label}</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="flex-1 h-1.5 bg-[var(--border)] rounded overflow-hidden">
            <div class="h-1.5 rounded" style="width:${pct}%; background:var(--accent);"></div>
          </div>
          <span class="text-[10px] mono text-[var(--ink-3)] w-[36px] text-right">${pct}%</span>
        </div>
        <p class="text-[11px] text-[var(--ink-2)] leading-relaxed line-clamp-3">${escHtml(c.snippet)}</p>
      </div>`;
  }).join('');
}

// ── Render next actions ────────────────────────────────────────────────────
function renderNextActions(recs) {
  const el = document.getElementById('nextActions');
  if (!recs || !recs.length) {
    el.innerHTML = '<span class="text-[12px] text-[var(--ink-3)] italic">No recommendations.</span>';
    return;
  }
  el.innerHTML = recs.map(r =>
    `<span class="chip-action inline-flex items-center text-[12px] font-medium px-3 py-1.5 rounded-lg border border-[var(--border)] bg-white hover:border-[var(--accent)] hover:text-[var(--accent)] cursor-default transition">${escHtml(r)}</span>`
  ).join('');
}

// ── Render decision basis ──────────────────────────────────────────────────
function renderDecisionBasis(basis) {
  const el = document.getElementById('decisionBasisList');
  const block = document.getElementById('decisionBasisBlock');
  if (!basis || !basis.length) { block.classList.add('hidden'); return; }
  block.classList.remove('hidden');
  el.innerHTML = basis.map(b =>
    `<li class="flex items-start gap-1.5 text-[12px] text-[var(--ink-2)]"><span class="text-[var(--ink-3)] mt-0.5">›</span>${escHtml(b)}</li>`
  ).join('');
}

// ── Render bounded multi-agent handoff ────────────────────────────────────
function renderAgentCollaboration(plan) {
  const block = document.getElementById('agentCollabBlock');
  const list = document.getElementById('agentCollabList');
  if (!plan || !plan.handoffs || !plan.handoffs.length) {
    block.classList.add('hidden');
    return;
  }
  block.classList.remove('hidden');
  document.getElementById('agentRuntimeTag').textContent =
    `${plan.runtime_mode || 'stateless'} · max ${plan.max_graph_steps || plan.handoffs.length} steps`;
  document.getElementById('agentCollabSummary').textContent =
    `${plan.summary}. Loop guard: ${plan.loop_guard || 'bounded retries'}.`;
  list.innerHTML = plan.handoffs.map((h, i) => {
    const rp = h.retry_policy || {};
    const retryText = rp.max_attempts ? `${rp.max_attempts} attempt${rp.max_attempts > 1 ? 's' : ''}` : 'bounded';
    const stopText = (rp.stop_conditions || []).slice(0, 2).join(' · ') || 'stop condition required';
    const actionText = (h.actions || []).slice(0, 2).map(escHtml).join('<br>');
    return `
      <div class="rounded-lg border border-[var(--border)] bg-[var(--canvas)] px-3 py-2">
        <div class="flex items-start gap-2">
          <span class="mono text-[10px] text-[var(--ink-3)] mt-0.5">${i + 1}</span>
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 flex-wrap">
              <span class="text-[12px] font-semibold text-[var(--ink-1)]">${escHtml(h.label)}</span>
              <span class="text-[10px] mono px-1.5 py-0.5 rounded bg-white border border-[var(--border)]">${h.executed ? 'executed' : 'planned'}</span>
              <span class="text-[10px] mono text-[var(--ink-3)]">${escHtml(retryText)}</span>
            </div>
            <p class="text-[11px] text-[var(--ink-3)] mt-0.5">${escHtml(h.trigger)}</p>
            <p class="text-[11px] text-[var(--ink-2)] mt-1 leading-relaxed">${actionText}</p>
            <p class="text-[10px] text-[var(--ink-3)] mt-1">Stops: ${escHtml(stopText)} · Escalates: ${escHtml(rp.escalation || 'human owner')}</p>
          </div>
        </div>
      </div>`;
  }).join('');
}

// ── Escape HTML ────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Set loading state ──────────────────────────────────────────────────────
function setLoading(on) {
  loading = on;
  const btn = document.getElementById('askBtn');
  const label = document.getElementById('askBtnLabel');
  if (on) {
    btn.disabled = true;
    btn.classList.add('opacity-60');
    label.innerHTML = '<span class="shimmer rounded w-20 h-4 inline-block"></span>';
    document.getElementById('answerText').innerHTML =
      '<div class="shimmer rounded h-4 w-3/4 mb-2"></div><div class="shimmer rounded h-4 w-full mb-2"></div><div class="shimmer rounded h-4 w-1/2"></div>';
  } else {
    btn.disabled = false;
    btn.classList.remove('opacity-60');
    label.textContent = 'Ask  ▶';
  }
}

// ── Main ask handler ───────────────────────────────────────────────────────
async function doAsk() {
  if (loading) return;
  const query = document.getElementById('query').value.trim();
  if (!query) { document.getElementById('query').focus(); return; }

  document.getElementById('errorMsg').classList.add('hidden');
  setLoading(true);

  const payload = { query, k: 5, method: currentMethod };
  const erState = buildErState();
  if (erState) payload.er_state = erState;

  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 30000);
    const resp = await fetch('/v1/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
    clearTimeout(timer);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail?.message || err.detail || `HTTP ${resp.status}`);
    }

    const d = await resp.json();
    renderResponse(d);
  } catch(e) {
    const msg = document.getElementById('errorMsg');
    msg.textContent = e.name === 'AbortError' ? 'Request timed out (30 s).' : String(e.message);
    msg.classList.remove('hidden');
  } finally {
    setLoading(false);
  }
}

// ── Render full response ───────────────────────────────────────────────────
function renderResponse(d) {
  // Case bar
  const ts = nowTs();
  document.getElementById('caseTimestamp').textContent = ts;
  document.getElementById('caseId').textContent =
    'ER-' + new Date().toISOString().slice(0,10).replace(/-/g,'') + '-' + String(Math.floor(Math.random()*999)+1).padStart(3,'0');

  // ─ Pane 2: ESI block ─
  const esiTier = d.esi_final || d.esi_rule_based;
  const cfg = ESI_CONFIG[esiTier] || ESI_CONFIG[5];
  const esiBlock = document.getElementById('esiBlock');
  esiBlock.className = `rounded-xl border-2 p-4 flex items-center gap-4 ${cfg.cls}`;
  document.getElementById('esiTier').textContent  = esiTier ?? '—';
  document.getElementById('esiLabel').textContent = cfg.urgency;

  const conf = d.esi_confidence;
  document.getElementById('esiConfBar').style.width = conf != null ? Math.round(conf * 100) + '%' : '0%';
  document.getElementById('esiConfText').textContent = conf != null ? (conf * 100).toFixed(0) + '%' : '—';

  // Votes
  const votes = d.esi_votes || {};
  const voteStr = Object.entries(votes)
    .sort((a,b) => b[1]-a[1])
    .map(([t, n]) => `${n}× ESI ${t}`)
    .join(' · ') || '—';
  document.getElementById('esiVotesText').innerHTML = `<b>Votes:</b> ${escHtml(voteStr)}`;

  // Disagreement
  document.getElementById('esiDisagreement').classList.toggle('hidden', !d.esi_disagreement);

  // Grounding chip
  const gp = groundingChip(conf);
  const gc = document.getElementById('groundingChip');
  gc.textContent = gp.text;
  gc.className = `text-[12px] font-semibold px-3 py-1 rounded-full border ${gp.cls}`;

  // Triage level chip
  if (d.triage_level) {
    const tc = document.getElementById('triageLevelChip');
    tc.textContent = d.triage_level;
    tc.className = `text-[11px] font-bold px-3 py-1 rounded-full border mono ${
      d.triage_level === 'NOW'  ? 'bg-red-50 border-red-400 text-red-700' :
      d.triage_level === 'SOON' ? 'bg-yellow-50 border-yellow-400 text-yellow-700' :
                                   'bg-green-50 border-green-400 text-green-700'
    }`;
    tc.classList.remove('hidden');
  }

  // Header badges
  const hesi = document.getElementById('headerEsiBadge');
  hesi.textContent = `ESI ${esiTier ?? '—'} · ${cfg.urgency}`;
  hesi.className = `flex items-center gap-1.5 text-[12px] font-semibold px-2.5 py-1 rounded-lg border-2 mono ${cfg.cls}`;
  hesi.classList.remove('hidden');

  const hg = document.getElementById('headerGrounding');
  hg.textContent = gp.text;
  hg.className = `flex items-center gap-1.5 text-[12px] font-medium ${gp.cls.includes('green') ? 'text-green-700' : gp.cls.includes('yellow') ? 'text-yellow-700' : 'text-red-700'}`;
  hg.classList.remove('hidden');

  const hc = document.getElementById('headerCiteCount');
  hc.textContent = `${(d.citations||[]).length} citations`;
  hc.classList.remove('hidden');

  // Answer text
  document.getElementById('answerText').innerHTML =
    `<span class="text-[var(--ink-1)]">${escHtml(d.answer || '—')}</span>`;

  // Warnings
  const wb = document.getElementById('warningBadge');
  if (d.warnings && d.warnings.length) {
    wb.textContent = `⚠ ${d.warnings.length} warning${d.warnings.length>1?'s':''}`;
    wb.classList.remove('hidden');
  } else {
    wb.classList.add('hidden');
  }

  // Explanation
  if (d.explanation_for_human) {
    document.getElementById('explanationText').textContent = d.explanation_for_human;
    document.getElementById('explanationWrap').classList.remove('hidden');
  }

  // Prediction signals
  if (d.prediction_signal) {
    const ps = d.prediction_signal;
    document.getElementById('predRisk').className = `text-[14px] font-bold mt-0.5 ${riskColor(ps.risk_level)}`;
    document.getElementById('predRisk').textContent = ps.risk_level?.toUpperCase() ?? '—';
    document.getElementById('predDet').className  = `text-[14px] font-bold mt-0.5 ${riskColor(ps.deterioration_risk)}`;
    document.getElementById('predDet').textContent = ps.deterioration_risk?.toUpperCase() ?? '—';
    document.getElementById('predBed').className  = `text-[14px] font-bold mt-0.5 ${riskColor(ps.bed_pressure_risk)}`;
    document.getElementById('predBed').textContent = ps.bed_pressure_risk?.toUpperCase() ?? '—';
    document.getElementById('predLos').textContent = ps.predicted_los_hours != null
      ? `${ps.predicted_los_hours.toFixed(0)} hrs` : '—';
    document.getElementById('predictionBlock').classList.remove('hidden');
  }

  renderDecisionBasis(d.decision_basis);
  renderAgentCollaboration(d.agent_collaboration);

  // ─ Pane 3: Evidence ─
  const redFlags = d.esi_red_flags || [];
  const rfBlock  = document.getElementById('redFlagsBlock');
  if (redFlags.length) {
    rfBlock.classList.remove('hidden');
    document.getElementById('redFlagsList').innerHTML = redFlags.map(f =>
      `<span class="text-[11px] font-medium bg-red-50 border border-red-300 text-red-700 px-2 py-0.5 rounded-full">${escHtml(f)}</span>`
    ).join('');
  } else {
    rfBlock.classList.add('hidden');
  }

  // Override block
  const ob = document.getElementById('overrideBlock');
  if (d.override_applied && d.override_reason) {
    document.getElementById('overrideReason').textContent = d.override_reason;
    ob.classList.remove('hidden');
  } else {
    ob.classList.add('hidden');
  }

  renderCitations(d.citations || []);

  // Vote breakdown
  const vb = document.getElementById('voteBlock');
  if (Object.keys(votes).length) {
    document.getElementById('voteText').textContent = voteStr;
    vb.classList.remove('hidden');
  } else {
    vb.classList.add('hidden');
  }

  // ─ Next Actions + Trace ─
  renderNextActions(d.operational_recommendations);
  renderTrace(d.guard_ms || 0, d.retrieve_ms || 0, d.generate_ms || 0, d.latency_ms || 0);

  // Method tag
  document.getElementById('methodUsedTag').textContent = `method: ${d.method_used}`;

  // Weave link
  if (d.trace_call_id) {
    document.getElementById('weaveLink').href =
      `https://wandb.ai/alynch-zeroshot/healthcare-genai/r/call/${d.trace_call_id}`;
    document.getElementById('weaveHint').textContent = `call: ${d.trace_call_id.slice(0,8)}…`;
  }
}

// ── Keyboard shortcut: ⌘+Enter / Ctrl+Enter ───────────────────────────────
document.getElementById('query').addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); doAsk(); }
});

document.getElementById('askBtn').addEventListener('click', doAsk);

// ── Init timestamp ─────────────────────────────────────────────────────────
document.getElementById('caseTimestamp').textContent = nowTs();
</script>
</body>
</html>"""


@router.get("/vertex", response_class=HTMLResponse)
def vertex_console() -> HTMLResponse:
    return HTMLResponse(_HTML)
