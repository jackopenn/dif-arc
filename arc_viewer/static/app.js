const COLORS = [
  "#000000", // 0 black
  "#0074D9", // 1 blue
  "#FF4136", // 2 red
  "#2ECC40", // 3 green
  "#FFDC00", // 4 yellow
  "#AAAAAA", // 5 gray
  "#F012BE", // 6 magenta
  "#FF851B", // 7 orange
  "#7FDBFF", // 8 cyan
  "#870C25", // 9 maroon
  "#FFFFFF", // 10 white (border)
  "#374151", // 11 slate (pad)
];

function $(id) { return document.getElementById(id); }

function shape2d(grid) {
  if (!grid || !Array.isArray(grid) || grid.length === 0) return [0, 0];
  const h = grid.length;
  const w = Array.isArray(grid[0]) ? grid[0].length : 0;
  return [h, w];
}

function renderGrid(canvas, grid) {
  const ctx = canvas.getContext("2d");
  const [h, w] = shape2d(grid);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (h === 0 || w === 0) {
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const maxCanvas = 320;
  const cell = Math.max(6, Math.floor(Math.min(maxCanvas / w, maxCanvas / h)));
  const drawW = w * cell;
  const drawH = h * cell;
  canvas.width = drawW;
  canvas.height = drawH;

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const v = (grid[r][c] ?? 0) | 0;
      ctx.fillStyle = COLORS[v] || "#111827";
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
  }

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let r = 0; r <= h; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * cell + 0.5);
    ctx.lineTo(drawW, r * cell + 0.5);
    ctx.stroke();
  }
  for (let c = 0; c <= w; c++) {
    ctx.beginPath();
    ctx.moveTo(c * cell + 0.5, 0);
    ctx.lineTo(c * cell + 0.5, drawH);
    ctx.stroke();
  }
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

async function apiGet(url) {
  const res = await fetch(url, { cache: "no-store" });
  const json = await res.json();
  if (!res.ok) throw new Error(json?.error || `HTTP ${res.status}`);
  return json;
}

let selectedPuzzleId = null;
let selectedAugId = null;
let currentAugList = [];

function setStatus(text) {
  $("status").textContent = text || "";
}

function fmtPct(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(2) + "%";
}

function setMetrics(global, filtered) {
  if (!global || !filtered) {
    $("metrics").textContent = "";
    return;
  }
  const g = `global: cell ${fmtPct(global.cell_accuracy)} | puzzle ${fmtPct(global.puzzle_accuracy)} | cropped ${global.num_cropped ?? 0}`;
  const f = `filtered: cell ${fmtPct(filtered.cell_accuracy)} | puzzle ${fmtPct(filtered.puzzle_accuracy)} | cropped ${filtered.num_cropped ?? 0}`;
  $("metrics").textContent = `${g}   •   ${f}`;
}

function getFilters() {
  return {
    q: $("search").value.trim(),
    onlyCorrect: $("onlyCorrect").checked,
    onlyWithPred: $("onlyWithPred").checked,
  };
}

function clearAugsAndExamples() {
  $("augList").innerHTML = "";
  $("augInfo").textContent = "";
  $("examples").innerHTML = "";
  $("metaPre").textContent = "";
  currentAugList = [];
}

async function refreshSummary() {
  const { q, onlyCorrect, onlyWithPred } = getFilters();
  const sum = await apiGet(`/api/summary?q=${encodeURIComponent(q)}&only_correct=${onlyCorrect ? 1 : 0}&only_with_pred=${onlyWithPred ? 1 : 0}`);
  setMetrics(sum.global, sum.filtered);
}

async function loadPuzzleSidebar() {
  const { q, onlyCorrect, onlyWithPred } = getFilters();
  setStatus("Loading puzzles…");
  const puzzles = await apiGet(`/api/puzzles?q=${encodeURIComponent(q)}&limit=2000&only_correct=${onlyCorrect ? 1 : 0}&only_with_pred=${onlyWithPred ? 1 : 0}`);
  const list = $("puzzleList");
  list.innerHTML = "";
  $("puzzleInfo").textContent = `${puzzles.count} shown / ${puzzles.total}`;

  for (const p of puzzles.puzzles || []) {
    const pid = p.puzzle_id;
    const subtitle = `augs=${p.count} • aug_acc ${fmtPct(p.aug_accuracy)} (${p.num_correct}/${p.count})`;

    const el = document.createElement("div");
    el.className = "id depth-0";
    el.dataset.key = `p:${pid}`;
    el.textContent = pid;
    const sub = document.createElement("div");
    sub.className = "meta2";
    sub.textContent = subtitle;
    el.appendChild(sub);

    el.addEventListener("click", () => {
      selectedPuzzleId = pid;
      selectedAugId = null;
      for (const x of document.querySelectorAll("#puzzleList .id")) x.classList.toggle("active", x.dataset.key === `p:${pid}`);
      clearAugsAndExamples();
      loadAugSidebar();
    });

    list.appendChild(el);
  }
  setStatus("");
}

async function loadAugSidebar() {
  const { onlyCorrect, onlyWithPred } = getFilters();
  if (!selectedPuzzleId) {
    $("augInfo").textContent = "select a puzzle_id";
    $("augList").innerHTML = "";
    return;
  }
  setStatus("Loading augs…");
  const augsResp = await apiGet(`/api/augs?puzzle_id=${encodeURIComponent(selectedPuzzleId)}&limit=5000`);
  const list = $("augList");
  list.innerHTML = "";

  const filtered = [];
  for (const a of augsResp.augs || []) {
    if (onlyWithPred && !a.has_pred) continue;
    if (onlyCorrect && a.equal !== true) continue;
    filtered.push(a);
  }
  currentAugList = filtered.map((a) => a.aug_puzzle_idx);
  $("augInfo").textContent = `${filtered.length} shown / ${augsResp.count} augs`;

  for (const a of filtered) {
    const augId = a.aug_puzzle_idx;
    const text = !a.has_pred ? "no pred" : a.equal === true ? "match" : "diff";
    const cls = !a.has_pred ? "miss" : a.equal === true ? "ok" : "miss";

    const el = document.createElement("div");
    el.className = "id depth-0";
    el.dataset.key = `a:${augId}`;
    el.textContent = String(augId);
    const pill = document.createElement("span");
    pill.className = `pill ${cls}`;
    pill.textContent = text;
    el.appendChild(pill);
    const sub = document.createElement("div");
    sub.className = "meta2";
    sub.textContent = `examples=${(a.example_idxs || []).length} • acc ${fmtPct(a.accuracy)} • cropped ${a.cropped ? 1 : 0}`;
    el.appendChild(sub);
    el.addEventListener("click", () => {
      selectedAugId = augId;
      for (const x of document.querySelectorAll("#augList .id")) x.classList.toggle("active", x.dataset.key === `a:${augId}`);
      loadAugPage(augId);
    });
    list.appendChild(el);
  }

  setStatus("");
}

function findNeighbor(dir) {
  if (selectedAugId == null) return null;
  const idx = currentAugList.indexOf(selectedAugId);
  if (idx === -1) return null;
  const nextIdx = idx + (dir === "next" ? 1 : -1);
  if (nextIdx < 0 || nextIdx >= currentAugList.length) return null;
  return currentAugList[nextIdx];
}

function mkGridCard(title, grid, foot) {
  const card = document.createElement("div");
  card.className = "grid-card";
  const t = document.createElement("div");
  t.className = "grid-title";
  t.textContent = title;
  const wrap = document.createElement("div");
  wrap.className = "grid-wrap";
  const canvas = document.createElement("canvas");
  canvas.width = 300;
  canvas.height = 300;
  wrap.appendChild(canvas);
  const f = document.createElement("div");
  f.className = "grid-foot";
  f.textContent = foot || "";
  card.appendChild(t);
  card.appendChild(wrap);
  card.appendChild(f);
  renderGrid(canvas, grid);
  return card;
}

async function loadAugPage(augId) {
  setStatus(`Loading aug ${augId}…`);
  const data = await apiGet(`/api/aug/${augId}`);
  selectedAugId = augId;
  $("augInput").value = String(augId);

  $("metaPre").textContent = pretty({
    puzzle_id: data.puzzle_id,
    aug_puzzle_idx: data.aug_puzzle_idx,
    aug_stats: data.aug_stats,
  });

  const container = $("examples");
  container.innerHTML = "";
  for (const ex of data.examples || []) {
    const exId = ex.example_idx;
    const block = document.createElement("div");
    block.className = "ex-block";
    block.id = `ex-${augId}-${exId}`;

    const head = document.createElement("div");
    head.className = "ex-head";
    const left = document.createElement("div");
    left.className = "ex-title";
    left.textContent = `example_idx ${exId}`;
    const right = document.createElement("div");
    right.className = "ex-sub";
    const m = ex.match || {};
    const acc = m.same_shape && m.accuracy != null ? fmtPct(m.accuracy) : "—";
    right.textContent = `acc ${acc} • equal ${m.equal === true ? "yes" : "no"}`;
    head.appendChild(left);
    head.appendChild(right);
    block.appendChild(head);

    const grids = document.createElement("div");
    grids.className = "ex-grids";

    const [xh, xw] = shape2d(ex.x_raw);
    const [yh, yw] = shape2d(ex.y_raw);
    const [xlh, xlw] = shape2d(ex.x_dl);
    const [ylh, ylw] = shape2d(ex.y_dl);
    const [prh, prw] = shape2d(ex.pred_raw ?? data.pred_raw);
    const [puh, puw] = shape2d(ex.pred_used);

    grids.appendChild(mkGridCard("x (raw)", ex.x_raw, `${xh}×${xw}`));
    grids.appendChild(mkGridCard("y (raw)", ex.y_raw, `${yh}×${yw}`));
    grids.appendChild(mkGridCard("x (dataloader)", ex.x_dl, xlh && xlw ? `${xlh}×${xlw} (pad=11,border=10)` : "—"));
    grids.appendChild(mkGridCard("y (dataloader)", ex.y_dl, ylh && ylw ? `${ylh}×${ylw} (pad=11,border=10)` : "—"));
    grids.appendChild(mkGridCard("pred (raw)", (ex.pred_raw ?? data.pred_raw), data.has_pred ? `${prh}×${prw}` : "no pred"));
    let usedFoot = data.has_pred ? `${puh}×${puw}` : "no pred";
    if (ex.pred_alignment && ex.pred_alignment.cropped) usedFoot += " • cropped top-left to y";
    grids.appendChild(mkGridCard("pred (cropped/used)", ex.pred_used, usedFoot));

    block.appendChild(grids);
    container.appendChild(block);
  }

  setStatus("");
}

function goInput() {
  const s = $("augInput").value.trim();
  if (!s || !/^\d+$/.test(s)) return;
  const id = parseInt(s, 10);
  loadAugPage(id).catch((e) => setStatus(`Error: ${e.message}`));
}

async function init() {
  $("reload").addEventListener("click", async () => {
    await refreshSummary();
    await loadPuzzleSidebar();
    await loadAugSidebar();
  });

  $("goBtn").addEventListener("click", goInput);
  $("augInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") goInput();
  });

  $("prevBtn").addEventListener("click", () => {
    const id = findNeighbor("prev");
    if (id != null) loadAugPage(id);
  });
  $("nextBtn").addEventListener("click", () => {
    const id = findNeighbor("next");
    if (id != null) loadAugPage(id);
  });

  let timer = null;
  $("search").addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      selectedPuzzleId = null;
      selectedAugId = null;
      clearAugsAndExamples();
      await refreshSummary();
      await loadPuzzleSidebar();
    }, 200);
  });

  const refilter = async () => {
    await refreshSummary();
    // puzzle list filtering happens server-side via /api/puzzles params
    const prevPuzzle = selectedPuzzleId;
    const prevAug = selectedAugId;
    await loadPuzzleSidebar();

    // If the selected puzzle got filtered out, clear selection + dependent panes.
    if (prevPuzzle) {
      const stillThere = document.querySelector(`#puzzleList .id[data-key="p:${CSS.escape(prevPuzzle)}"]`);
      if (!stillThere) {
        selectedPuzzleId = null;
        selectedAugId = null;
        clearAugsAndExamples();
        return;
      }
    }
    await loadAugSidebar();
    if (selectedAugId != null) await loadAugPage(selectedAugId);
  };
  $("onlyCorrect").addEventListener("change", refilter);
  $("onlyWithPred").addEventListener("change", refilter);

  await refreshSummary();
  await loadPuzzleSidebar();
}

init().catch((e) => {
  console.error(e);
  setStatus(`Error: ${e.message}`);
});

