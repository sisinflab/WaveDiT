/* ==========================================================================
   WaveDiT Studio - UI logic (vanilla ES2020, no framework)
   Talks to the local studio server: JSON API + SSE. See macos/ARCHITECTURE.md.
   ========================================================================== */
(() => {
  "use strict";

  /* ======================================================================
     Section: tiny DOM + format helpers
     ====================================================================== */
  const byId = (id) => document.getElementById(id);
  const on = (el, ev, fn) => el && el.addEventListener(ev, fn);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const SEED_MAX = 2147483647;
  const randSeed = () => Math.floor(Math.random() * (SEED_MAX + 1));
  const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const fmtSec = (s) => {
    if (!Number.isFinite(s)) return "?";
    if (s >= 90) return `${(s / 60).toFixed(1)} min`;
    if (s >= 10) return `${Math.round(s)} s`;
    return `${s.toFixed(1)} s`;
  };
  const fmtGB = (gb) => `${Number(gb).toFixed(1)} GB`;
  const fmtMB = (mb) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`);
  const fmtAgo = (epochS) => {
    const d = Math.max(0, Date.now() / 1000 - epochS);
    if (d < 60) return "just now";
    if (d < 3600) return `${Math.round(d / 60)} min ago`;
    if (d < 86400) return `${Math.round(d / 3600)} h ago`;
    return `${Math.round(d / 86400)} d ago`;
  };

  const setFill = (input) => {
    const lo = parseFloat(input.min || "0");
    const hi = parseFloat(input.max || "100");
    const v = parseFloat(input.value);
    const pct = hi > lo ? ((v - lo) / (hi - lo)) * 100 : 0;
    input.style.setProperty("--fill", `${pct}%`);
  };

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_) {
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        ta.remove();
        return ok;
      } catch (_e) {
        return false;
      }
    }
  }

  /* ======================================================================
     Section: in-memory store
     ====================================================================== */
  const store = {
    version: "",
    deviceType: "cpu",
    ageRange: [6, 95],
    settings: {},
    calibration: {},
    weights: [],
    library: [],
    params: { model: null, age: 72, seed: 42, steps: 10, cfg_scale: 1.0, sampler: "euler", morpheus: 1.0 },
    lastSeed: null,
    colormap: "gray",
    gamma: 1.0,
    busy: false,
    jobId: null,
    jobKind: null, // "gen" | "sweep"
    activeId: null,
    pendingItem: null,
    sweep: null, // {frames:[item], idx, playing, loop, timer, total}
    shelfCollapsed: false,
    immersive: false,
    layout: "row",      // "row" | "grid" | "3d"
    guides: true,       // crosshair + clip plane visible
    ramGb: 0,           // total unified/system memory, for the load meter
    heroShown: false,   // the bundled sample brain is on screen
  };
  const SURPRISE_COLORMAPS = ["viridis", "plasma", "bone", "magma"];
  const STORAGE_PER_ITEM = 27.5e6; // approx bytes of one gzipped float32 volume

  /* ======================================================================
     Section: toasts
     ====================================================================== */
  function toast(message, kind = "info", ms = 3500) {
    const stack = byId("toast-stack");
    if (!stack) return;
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = message;
    stack.appendChild(el);
    while (stack.children.length > 5) stack.firstChild.remove();
    setTimeout(() => {
      el.classList.add("leaving");
      setTimeout(() => el.remove(), 220);
    }, ms);
  }

  /* ======================================================================
     Section: API client
     ====================================================================== */
  async function api(path, body) {
    const opts = body === undefined
      ? { method: "GET" }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
      const err = new Error((data && data.error) || `${res.status} ${res.statusText}`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  /* ======================================================================
     Section: modal system (single root, focus trap, esc to close)
     ====================================================================== */
  const modal = { name: null, onClose: null };

  function openModal(name, setup) {
    const root = byId("modal-root");
    const card = byId("modal-card");
    const tpl = byId(`tpl-${name}`);
    if (!root || !card || !tpl) return null;
    closeModal(true);
    card.replaceChildren(tpl.content.cloneNode(true));
    if (name !== "confirm") {
      const x = document.createElement("button");
      x.className = "icon-btn modal-close";
      x.title = "Close";
      x.setAttribute("aria-label", "Close");
      x.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
      on(x, "click", () => closeModal());
      card.appendChild(x);
    }
    modal.name = name;
    modal.onClose = null;
    root.hidden = false;
    requestAnimationFrame(() => root.classList.add("open"));
    if (typeof setup === "function") setup(card);
    const first = card.querySelector("button:not(.modal-close), input, select");
    if (first) first.focus();
    return card;
  }

  function closeModal(immediate = false) {
    const root = byId("modal-root");
    const card = byId("modal-card");
    if (!root || root.hidden) return;
    const done = modal.onClose;
    modal.name = null;
    modal.onClose = null;
    root.classList.remove("open");
    const finish = () => {
      // A modal reopened during the close animation must survive the stale timer.
      if (modal.name) return;
      root.hidden = true;
      card.replaceChildren();
    };
    if (immediate || REDUCED) finish(); else setTimeout(finish, 200);
    if (typeof done === "function") done();
  }

  function confirmDialog({ title, message, confirmLabel = "Delete" }) {
    return new Promise((resolve) => {
      const card = openModal("confirm", (c) => {
        c.querySelector('[data-role="confirm-title"]').textContent = title;
        c.querySelector('[data-role="confirm-message"]').textContent = message;
        const ok = c.querySelector('[data-action="confirm-ok"]');
        ok.textContent = confirmLabel;
        on(ok, "click", () => { modal.onClose = null; closeModal(); resolve(true); });
        on(c.querySelector('[data-action="confirm-cancel"]'), "click", () => { closeModal(); });
      });
      if (!card) { resolve(false); return; }
      modal.onClose = () => resolve(false);
    });
  }

  // focus trap + esc
  on(window, "keydown", (e) => {
    if (e.key === "Escape") {
      if (modal.name) { e.preventDefault(); closeModal(); return; }
      if (store.immersive) { e.preventDefault(); toggleImmersive(false); }
      return;
    }
    if (e.key === "Tab" && modal.name) {
      const card = byId("modal-card");
      const focusables = [...card.querySelectorAll("button, [href], input, select, textarea")]
        .filter((el) => el.offsetParent !== null);
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  });
  on(byId("modal-backdrop"), "click", () => closeModal());

  /* ======================================================================
     Section: age control + microcopy
     ====================================================================== */
  const AGE_BANDS = [
    [12, "Childhood: gray matter near its lifetime peak."],
    [19, "Adolescence: synaptic pruning in full swing."],
    [39, "Adulthood: white matter at peak myelination."],
    [64, "Midlife: slow, graceful cortical thinning begins."],
    [79, "Senior years: ventricles widen, sulci deepen."],
    [Infinity, "Late life: a brain with stories etched in every sulcus."],
  ];
  const microcopyFor = (age) => AGE_BANDS.find(([hi]) => age <= hi)[1];

  function setAge(age, fromSlider = false) {
    const [lo, hi] = store.ageRange;
    store.params.age = clamp(Math.round(age), lo, hi);
    byId("age-readout").textContent = String(store.params.age);
    byId("age-microcopy").textContent = microcopyFor(store.params.age);
    const slider = byId("age-slider");
    if (!fromSlider) slider.value = String(store.params.age);
    setFill(slider);
  }

  /* ======================================================================
     Section: sidebar controls (seed, steps, cfg, sampler, morpheus)
     ====================================================================== */
  function setSeed(seed) {
    store.params.seed = clamp(Math.round(seed) || 0, 0, SEED_MAX);
    byId("seed-input").value = String(store.params.seed);
  }

  function setSteps(steps, fromSlider = false) {
    store.params.steps = clamp(Math.round(steps) || 1, 1, 200);
    byId("steps-value").textContent = String(store.params.steps);
    const slider = byId("steps-slider");
    if (!fromSlider) slider.value = String(store.params.steps);
    setFill(slider);
    updateEstimate();
  }

  function setCfg(v, fromSlider = false) {
    store.params.cfg_scale = clamp(Math.round(v * 10) / 10, 1, 8);
    byId("cfg-value").textContent = store.params.cfg_scale.toFixed(1);
    const slider = byId("cfg-slider");
    if (!fromSlider) slider.value = String(store.params.cfg_scale);
    setFill(slider);
    updateEstimate();
  }

  function setSampler(name) {
    store.params.sampler = name === "euler" ? "euler" : "heun";
    byId("sampler-seg").querySelectorAll(".seg-cell").forEach((c) => {
      const sel = c.dataset.sampler === store.params.sampler;
      c.classList.toggle("selected", sel);
      c.setAttribute("aria-checked", sel ? "true" : "false");
    });
    updateEstimate();
  }

  function setMorpheus(v, fromSlider = false) {
    store.params.morpheus = clamp(Math.round(v * 20) / 20, 0, 2);
    byId("morpheus-value").textContent = store.params.morpheus.toFixed(2);
    byId("morpheus-hint").hidden = store.params.morpheus !== 0;
    const slider = byId("morpheus-slider");
    if (!fromSlider) slider.value = String(store.params.morpheus);
    setFill(slider);
  }

  /* ======================================================================
     Section: time estimates from calibration
     ====================================================================== */
  const nfeTotal = (steps, sampler, cfg) =>
    (sampler === "heun" ? 2 * steps - 1 : steps) * (Math.abs(cfg - 1.0) > 1e-9 ? 2 : 1);

  // Mirror engine._resolve_precision so the estimate uses the right calibration key.
  function predictedPrecision() {
    if (store.deviceType === "cpu") return "float32";
    const mode = store.settings.precision || "auto";
    if (mode === "bf16" || mode === "float32") return mode;
    if (store.settings.bf16_ok === false) return "float32";
    return "bf16";
  }

  function secPerNfe(model) {
    const cal = store.calibration || {};
    const dev = store.deviceType;
    for (const prec of [predictedPrecision(), "float32", "bf16"]) {
      const v = cal[`${model}|${prec}|${dev}`];
      if (typeof v === "number" && v > 0) return v;
    }
    return null;
  }

  function estimateSeconds(steps, sampler, cfg, model) {
    const spn = secPerNfe(model || store.params.model);
    if (spn === null) return null;
    return nfeTotal(steps, sampler, cfg) * spn;
  }

  function updateEstimate() {
    const el = byId("steps-estimate");
    const p = store.params;
    const est = estimateSeconds(p.steps, p.sampler, p.cfg_scale, p.model);
    const nfe = nfeTotal(p.steps, p.sampler, p.cfg_scale);
    el.textContent = est === null
      ? `${nfe} network evaluations, time estimate appears after the first run`
      : `${nfe} network evaluations, about ${fmtSec(est)}`;
  }

  /* ======================================================================
     Section: model segmented control
     ====================================================================== */
  function modelCaption(w) {
    if (w.source === "custom") return "custom";
    const known = { Base: "fast", FinePatch: "detailed", Deep: "deep", Wide: "wide" };
    if (known[w.label]) return known[w.label];
    return w.size_mb ? fmtMB(w.size_mb) : "checkpoint";
  }

  function renderModelSeg() {
    const seg = byId("model-seg");
    seg.replaceChildren();
    if (!store.weights.length) {
      const p = document.createElement("p");
      p.className = "microcopy";
      p.textContent = "No checkpoints found. Open the weights panel to download one.";
      seg.appendChild(p);
      return;
    }
    for (const w of store.weights) {
      const cell = document.createElement("button");
      cell.className = "seg-cell";
      cell.dataset.file = w.file;
      cell.setAttribute("role", "radio");
      const selected = store.params.model === w.file;
      cell.classList.toggle("selected", selected);
      cell.setAttribute("aria-checked", selected ? "true" : "false");
      const name = document.createElement("span");
      name.textContent = w.label;
      const cap = document.createElement("span");
      cap.className = "seg-caption";
      if (w.downloading) cap.textContent = "downloading…";
      else if (w.coming_soon) {
        cell.classList.add("coming-soon");
        cap.textContent = "soon";
      } else if (!w.downloaded) {
        cell.classList.add("undownloaded");
        cap.innerHTML = '<svg class="dl-glyph" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2v7M5.4 6.6 8 9.2l2.6-2.6M3 12.5h10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> get';
      } else cap.textContent = modelCaption(w);
      cell.append(name, cap);
      on(cell, "click", () => {
        if (!w.downloaded) { openWeightsModal(); return; }
        store.params.model = w.file;
        renderModelSeg();
        updateEstimate();
      });
      seg.appendChild(cell);
    }
    const dot = byId("weights-badge-dot");
    if (dot) dot.hidden = !store.weights.some((w) => !w.downloaded);
  }

  function ensureModelSelected() {
    const downloaded = store.weights.filter((w) => w.downloaded);
    const current = store.weights.find((w) => w.file === store.params.model);
    if (!current || !current.downloaded) {
      const preferred = store.settings.default_model;
      const pick = downloaded.find((w) => w.file === preferred) || downloaded[0];
      store.params.model = pick ? pick.file : (store.weights[0] ? store.weights[0].file : null);
    }
  }

  /* ======================================================================
     Section: Niivue viewer
     ====================================================================== */
  let nv = null;
  let NVC = null; // {SLICE_TYPE, SHOW_RENDER, DRAG_MODE}
  const CLIP_PLANE = [0.3, 180, 20];
  const CLIP_OFF = [2, 0, 0];   // depth 2 sits outside the volume: no cut, no plane drawn
  const MP_LAYOUT = { grid: 2, row: 3 };  // Niivue multiplanarLayout: GRID=2, ROW=3
  // Bundled sample brain shown at startup so the app never opens to an empty stage.
  const HERO = {
    id: "__hero__", vol_url: "assets/hero.nii.gz", hero: true,
    model_label: "FinePatch", age: 72, seed: 42, steps: 100, sampler: "heun",
    cfg_scale: 1.0, morpheus: 1.0, precision: "bf16",
  };
  const LAYOUT_ORDER = ["row", "grid", "3d"];
  const LAYOUT_TITLE = { row: "Viewer layout: rows", grid: "Viewer layout: 2x2 grid", "3d": "Viewer layout: 3D only" };

  function showWebglFallback(msg) {
    const fb = byId("webgl-fallback");
    if (fb) {
      fb.hidden = false;
      byId("fallback-msg").textContent = msg || "";
    }
    const empty = byId("empty-state");
    if (empty) empty.hidden = true;
  }

  async function initViewer() {
    try {
      if (!window.niivue || !window.niivue.Niivue) {
        showWebglFallback("vendor/niivue.umd.js did not load.");
        return;
      }
      let gl2 = null;
      try { gl2 = document.createElement("canvas").getContext("webgl2"); } catch (_) { /* below */ }
      if (!gl2) { showWebglFallback("getContext(webgl2) returned null."); return; }

      const { Niivue, SLICE_TYPE, SHOW_RENDER, DRAG_MODE } = window.niivue;
      NVC = { SLICE_TYPE, SHOW_RENDER, DRAG_MODE };
      nv = new Niivue({
        backColor: [0.039, 0.047, 0.063, 1],
        show3Dcrosshair: true,
        crosshairColor: [0.5, 0.85, 1.0, 0.9],
        isColorbar: false,
        dragMode: DRAG_MODE.slicer3D,
        isOrientCube: false,
        multiplanarShowRender: SHOW_RENDER.ALWAYS,
        isResizeCanvas: true,
      });
      await nv.attachTo("gl");
      applyLayout();
      nv.setInterpolation(false);
      if (store.pendingItem) {
        const item = store.pendingItem;
        store.pendingItem = null;
        await loadItemIntoViewer(item);
      }
    } catch (e) {
      nv = null;
      showWebglFallback(String((e && e.message) || e));
    }
  }

  function applyViewerLook() {
    if (!nv) return;
    try {
      if (nv.volumes && nv.volumes.length) nv.setColormap(nv.volumes[0].id, store.colormap);
      applyGuides();
      nv.setInterpolation(false);
      nv.setGamma(store.gamma);
      nv.drawScene();
    } catch (_) { /* transient draw errors are non-fatal */ }
  }

  // The blue X/Y/Z crosshair lines and the blue clip plane are the "guides".
  function applyGuides() {
    if (!nv) return;
    try {
      nv.opts.show3Dcrosshair = store.guides;
      nv.setCrosshairWidth(store.guides ? 1 : 0);
      nv.setClipPlane((store.guides ? CLIP_PLANE : CLIP_OFF).slice());
      nv.setClipPlaneColor([0.4, 0.7, 1.0, store.guides ? 0.4 : 0.0]);
    } catch (_) { /* non-fatal */ }
  }

  function setGuides(on) {
    store.guides = on === undefined ? !store.guides : on;
    const btn = byId("guides-btn");
    btn.setAttribute("aria-pressed", store.guides ? "true" : "false");
    btn.title = store.guides ? "Hide crosshair and clip plane" : "Show crosshair and clip plane";
    applyGuides();
    if (nv) try { nv.drawScene(); } catch (_) { /* non-fatal */ }
  }

  // Three viewer layouts: row (ortho strip + 3D), 2x2 grid, or 3D render only.
  function applyLayout() {
    if (!nv || !NVC) return;
    try {
      if (store.layout === "3d") {
        nv.setSliceType(NVC.SLICE_TYPE.RENDER);
      } else {
        nv.setSliceType(NVC.SLICE_TYPE.MULTIPLANAR);
        if (typeof nv.setMultiplanarLayout === "function") {
          nv.setMultiplanarLayout(MP_LAYOUT[store.layout]);
        }
      }
      nv.drawScene();
    } catch (_) { /* non-fatal */ }
  }

  function setLayout(mode) {
    store.layout = LAYOUT_ORDER.includes(mode) ? mode : "row";
    const btn = byId("layout-btn");
    btn.title = LAYOUT_TITLE[store.layout];
    byId("layout-btn").querySelectorAll("svg").forEach((s) => { s.hidden = true; });
    const icon = { row: ".ic-layout-row", grid: ".ic-layout-grid", "3d": ".ic-layout-3d" }[store.layout];
    const el = btn.querySelector(icon);
    if (el) el.hidden = false;
    applyLayout();
  }

  function cycleLayout() {
    const i = LAYOUT_ORDER.indexOf(store.layout);
    setLayout(LAYOUT_ORDER[(i + 1) % LAYOUT_ORDER.length]);
  }

  // Load the bundled sample brain when there is nothing else to show.
  async function loadHero() {
    if (!nv) { byId("empty-state").hidden = false; return; }
    try {
      byId("empty-state").hidden = true;
      await nv.loadVolumes([{ url: HERO.vol_url, name: "hero.nii.gz", colormap: store.colormap }]);
      applyLayout();
      applyViewerLook();
      store.heroShown = true;
      store.activeId = null;
      byId("provenance-bar").textContent =
        "Sample preview · FinePatch · age 72 · seed 42 · 100 steps";
    } catch (_) {
      byId("empty-state").hidden = false;  // no hero asset: fall back to the empty hero
    }
  }

  function crossfadeStart() {
    const cover = byId("stage-cover");
    if (!cover || REDUCED) return;
    cover.style.transition = "none";
    cover.style.opacity = "0.85";
  }

  function crossfadeEnd() {
    const cover = byId("stage-cover");
    if (!cover) return;
    requestAnimationFrame(() => {
      cover.style.transition = "opacity 300ms ease-out";
      cover.style.opacity = "0";
    });
  }

  // Serialize Niivue loads: concurrent loadVolumes calls (sweep scrubbing or
  // autoplay) interleave Niivue's clear-then-append and stack volumes. Latest
  // request wins; intermediate scrub positions are coalesced away.
  let loadBusy = false;
  let loadNext = null;

  async function loadItemIntoViewer(item, { crossfade = true } = {}) {
    store.activeId = item.id;
    store.heroShown = false;
    byId("empty-state").hidden = true;
    updateProvenance(item);
    highlightActiveCard();
    if (!nv) { store.pendingItem = item; return; }
    if (loadBusy) { loadNext = { item, crossfade }; return; }
    loadBusy = true;
    try {
      let cur = { item, crossfade };
      while (cur) {
        loadNext = null;
        try {
          if (cur.crossfade) crossfadeStart();
          await nv.loadVolumes([{ url: cur.item.vol_url, name: `${cur.item.id}.nii.gz`, colormap: store.colormap }]);
          applyLayout();
          applyViewerLook();
        } catch (e) {
          toast(`Could not load volume: ${(e && e.message) || e}`, "error");
        } finally {
          if (cur.crossfade) crossfadeEnd();
        }
        cur = loadNext;
      }
    } finally {
      loadBusy = false;
    }
  }

  function resetView() {
    if (!nv) return;
    try {
      applyGuides();
      if (nv.scene) { nv.scene.renderAzimuth = 180; nv.scene.renderElevation = 20; }
      nv.setGamma(store.gamma);
      nv.drawScene();
    } catch (_) { /* non-fatal */ }
  }

  function setColormap(name) {
    store.colormap = name;
    if (!nv) return;
    try {
      if (nv.volumes && nv.volumes.length) { nv.setColormap(nv.volumes[0].id, name); nv.drawScene(); }
    } catch (_) { /* non-fatal */ }
  }

  function setGamma(v) {
    store.gamma = v;
    if (!nv) return;
    try { nv.setGamma(v); nv.drawScene(); } catch (_) { /* non-fatal */ }
  }

  /* ======================================================================
     Section: accelerator load meter (memory in use of the GPU/CPU)
     ====================================================================== */
  function updateLoadMeter(memGb) {
    if (typeof memGb !== "number" || memGb < 0) return;
    const total = store.ramGb > 0 ? store.ramGb : 0;
    const frac = total > 0 ? Math.min(memGb / total, 1) : 0;
    byId("load-fill").style.width = `${(frac * 100).toFixed(1)}%`;
    byId("load-val").textContent = `${memGb.toFixed(1)} GB`;
    byId("load-meter").title = total > 0
      ? `${memGb.toFixed(1)} GB of ${total} GB in use on the ${store.deviceType === "cpu" ? "CPU" : "accelerator"}`
      : `${memGb.toFixed(1)} GB in use`;
  }

  function setLoadActive(active) {
    byId("load-meter").classList.toggle("active", active);
  }

  function toggleImmersive(force) {
    store.immersive = force !== undefined ? force : !store.immersive;
    document.body.classList.toggle("immersive", store.immersive);
    const btn = byId("fullscreen-btn");
    btn.setAttribute("aria-pressed", store.immersive ? "true" : "false");
    btn.title = store.immersive ? "Exit expanded viewer (Esc)" : "Expand viewer";
  }

  /* ======================================================================
     Section: provenance caption
     ====================================================================== */
  const itemById = (id) => store.library.find((it) => it.id === id) || null;

  function provenanceString(item) {
    const bits = [
      `age ${item.age}`, `seed ${item.seed}`, `${item.steps} steps`, item.sampler,
      item.model_label, item.precision, fmtSec(item.wall_s),
    ];
    if (item.morpheus !== 1.0) bits.splice(5, 0, `morpheus ${item.morpheus}`);
    if (item.cfg_scale !== 1.0) bits.splice(3, 0, `cfg ${item.cfg_scale}`);
    return bits.join(" · ");
  }

  function updateProvenance(item) {
    byId("provenance-bar").textContent = item ? provenanceString(item) : "no volume loaded yet";
  }

  async function copyProvenance() {
    const item = itemById(store.activeId);
    if (!item) { toast("Nothing to copy yet", "info"); return; }
    const ok = await copyText(provenanceString(item));
    toast(ok ? "Copied" : "Copy failed", ok ? "success" : "error");
  }

  /* ======================================================================
     Section: generation (POST + SSE progress + smooth ring)
     ====================================================================== */
  const prog = { pct: 0, shown: 0, rate: 0, eta: 0, lastT: 0, raf: 0, label: "" };

  function setBusy(busy) {
    store.busy = busy;
    const dot = byId("device-dot");
    dot.classList.toggle("ready", !busy);
    dot.classList.toggle("busy", busy);
    byId("generate-btn").classList.toggle("running", busy);
    byId("cancel-btn").hidden = !busy;
    setLoadActive(busy);
    if (!busy) {
      byId("generate-label").textContent = "Generate";
      byId("generate-btn").style.setProperty("--pct", "0");
    }
  }

  function overlayShow(modelLabel, stepText) {
    const ov = byId("gen-overlay");
    ov.classList.remove("hiding");
    ov.hidden = false;
    byId("gen-model").textContent = modelLabel || "";
    byId("gen-step").textContent = stepText || "";
    byId("empty-state").hidden = true;
    prog.pct = 0; prog.shown = 0; prog.rate = 0; prog.eta = 0; prog.lastT = performance.now();
    startProgressLoop();
  }

  function overlayHide() {
    stopProgressLoop();
    const ov = byId("gen-overlay");
    if (ov.hidden) return;
    ov.classList.add("hiding");
    setTimeout(() => { ov.hidden = true; ov.classList.remove("hiding"); }, REDUCED ? 0 : 220);
  }

  function progressTick(pct, etaS) {
    const now = performance.now();
    const dt = now - prog.lastT;
    if (dt > 30 && pct > prog.pct) prog.rate = clamp((pct - prog.pct) / dt, 0, 0.05); // pct per ms
    prog.pct = pct;
    prog.eta = etaS;
    prog.lastT = now;
  }

  function startProgressLoop() {
    stopProgressLoop();
    const step = () => {
      const now = performance.now();
      const since = now - prog.lastT;
      // extrapolate gently between SSE ticks, never past 99.5
      const target = Math.min(prog.pct + prog.rate * since, prog.pct + 8, 99.5);
      prog.shown += Math.max(0, target - prog.shown) * (REDUCED ? 1 : 0.14);
      const pctInt = Math.floor(prog.shown);
      const etaLeft = Math.max(0, prog.eta - since / 1000);
      byId("gen-ring").style.setProperty("--pct", prog.shown.toFixed(2));
      byId("gen-pct").textContent = `${pctInt}%`;
      byId("gen-eta").textContent = prog.eta > 0 ? `~${Math.ceil(etaLeft)} s left` : "warming up…";
      const gbtn = byId("generate-btn");
      gbtn.style.setProperty("--pct", prog.shown.toFixed(2));
      byId("generate-label").textContent =
        `Generating  ${pctInt}%` + (prog.eta > 0 ? `  ~${Math.ceil(etaLeft)}s left` : "");
      prog.raf = requestAnimationFrame(step);
    };
    prog.raf = requestAnimationFrame(step);
  }

  function stopProgressLoop() {
    if (prog.raf) cancelAnimationFrame(prog.raf);
    prog.raf = 0;
  }

  function currentModelDownloaded() {
    const w = store.weights.find((x) => x.file === store.params.model);
    return Boolean(w && w.downloaded);
  }

  async function startGenerate() {
    if (modal.name === "onboarding") return;
    if (!currentModelDownloaded()) {
      toast("Download a model checkpoint first", "info");
      openWeightsModal();
      return;
    }
    if (store.busy) { toast("A generation is already running", "info"); return; }
    try {
      const p = store.params;
      const res = await api("/api/generate", {
        model: p.model, age: p.age, seed: p.seed, steps: p.steps,
        cfg_scale: p.cfg_scale, sampler: p.sampler, morpheus: p.morpheus,
      });
      store.jobId = res.job_id;
      store.jobKind = "gen";
    } catch (e) {
      if (e.status === 409) toast("Already generating, hang tight", "info");
      else toast(`Generate failed: ${e.message}`, "error");
    }
  }

  async function cancelJob() {
    if (!store.jobId) return;
    try { await api("/api/cancel", { job_id: store.jobId }); }
    catch (e) { toast(`Cancel failed: ${e.message}`, "error"); }
  }

  /* ======================================================================
     Section: sweep (time-lapse) + scrubber
     ====================================================================== */
  async function startSweep() {
    if (!currentModelDownloaded()) {
      toast("Download a model checkpoint first", "info");
      openWeightsModal();
      return;
    }
    if (store.busy) { toast("A generation is already running", "info"); return; }
    const [lo, hi] = store.ageRange;
    const a0 = clamp(parseInt(byId("sweep-start").value, 10) || lo, lo, hi);
    const a1 = clamp(parseInt(byId("sweep-end").value, 10) || hi, lo, hi);
    const frames = clamp(parseInt(byId("sweep-frames").value, 10) || 5, 3, 10);
    const fixSeed = byId("sweep-fixed-seed").checked;
    try {
      const p = store.params;
      const res = await api("/api/sweep", {
        model: p.model, age_start: a0, age_end: a1, frames,
        seed: fixSeed ? p.seed : randSeed(), fix_seed: fixSeed,
        steps: p.steps, cfg_scale: p.cfg_scale, sampler: p.sampler, morpheus: p.morpheus,
      });
      store.jobId = res.job_id;
      store.jobKind = "sweep";
    } catch (e) {
      if (e.status === 409) toast("Already generating, hang tight", "info");
      else toast(`Sweep failed: ${e.message}`, "error");
    }
  }

  function sweepReset(total) {
    sweepStop();
    store.sweep = { frames: [], idx: 0, playing: false, loop: false, timer: 0, total: total || 0 };
    byId("sweep-bar").hidden = true;
    byId("sweep-loop").setAttribute("aria-pressed", "false");
  }

  function sweepBarShow() {
    const sw = store.sweep;
    if (!sw || !sw.frames.length) return;
    byId("sweep-bar").hidden = false;
    const scrub = byId("sweep-scrub");
    scrub.max = String(Math.max(0, sw.frames.length - 1));
    scrub.value = String(sw.idx);
    setFill(scrub);
    const ticks = byId("sweep-ticks");
    ticks.replaceChildren();
    sw.frames.forEach((f, i) => {
      const s = document.createElement("span");
      s.textContent = String(Math.round(f.age));
      s.classList.toggle("active", i === sw.idx);
      ticks.appendChild(s);
    });
  }

  function warmFetch(url) {
    try { fetch(url).catch(() => {}); } catch (_) { /* best effort preload */ }
  }

  function sweepShowFrame(idx, { fromScrub = false } = {}) {
    const sw = store.sweep;
    if (!sw || !sw.frames.length) return;
    sw.idx = clamp(idx, 0, sw.frames.length - 1);
    if (!fromScrub) {
      const scrub = byId("sweep-scrub");
      scrub.value = String(sw.idx);
      setFill(scrub);
    }
    byId("sweep-ticks").querySelectorAll("span").forEach((s, i) => s.classList.toggle("active", i === sw.idx));
    loadItemIntoViewer(sw.frames[sw.idx], { crossfade: false });
    const next = sw.frames[sw.idx + 1];
    if (next) warmFetch(next.vol_url);
  }

  function sweepPlay() {
    const sw = store.sweep;
    if (!sw || sw.frames.length < 2) return;
    sw.playing = true;
    byId("sweep-play").querySelector(".ic-play").hidden = true;
    byId("sweep-play").querySelector(".ic-pause").hidden = false;
    clearInterval(sw.timer);
    sw.timer = setInterval(() => {
      if (sw.idx >= sw.frames.length - 1) {
        if (sw.loop) sweepShowFrame(0);
        else sweepStop();
      } else sweepShowFrame(sw.idx + 1);
    }, 500); // 2 fps
  }

  function sweepStop() {
    const sw = store.sweep;
    if (!sw) return;
    sw.playing = false;
    clearInterval(sw.timer);
    const play = byId("sweep-play");
    if (play) {
      play.querySelector(".ic-play").hidden = false;
      play.querySelector(".ic-pause").hidden = true;
    }
  }

  /* ======================================================================
     Section: library shelf
     ====================================================================== */
  function modelLetter(item) {
    return (item.model_label || "?").slice(0, 1).toUpperCase();
  }

  const ICONS = {
    reuse: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 8a5.5 5.5 0 0 1 9.4-3.9l1.6 1.6M13.5 2v3.7H9.8M13.5 8a5.5 5.5 0 0 1-9.4 3.9l-1.6-1.6M2.5 14v-3.7h3.7" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    exp: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5v8M5 6.5l3 3 3-3M2.5 11v2.2a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1V11" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    del: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 4.5h10M6.5 4V2.8a1 1 0 0 1 1-1h1a1 1 0 0 1 1 1V4M4.5 4.5l.6 8.2a1 1 0 0 0 1 .9h3.8a1 1 0 0 0 1-.9l.6-8.2" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };

  function makeCard(item, isNew) {
    const card = document.createElement("div");
    card.className = "shelf-card";
    if (isNew) card.classList.add("new");
    card.dataset.id = item.id;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.title = provenanceString(item);

    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = `age ${item.age} brain`;
    img.src = item.thumb_url;
    card.appendChild(img);

    const ageB = document.createElement("span");
    ageB.className = "card-badge age";
    ageB.textContent = String(Math.round(item.age));
    card.appendChild(ageB);

    const modB = document.createElement("span");
    modB.className = "card-badge model";
    modB.textContent = modelLetter(item);
    card.appendChild(modB);

    const time = document.createElement("span");
    time.className = "card-time";
    time.dataset.created = String(item.created);
    time.textContent = fmtAgo(item.created);
    card.appendChild(time);

    const acts = document.createElement("div");
    acts.className = "card-actions";
    const mkAct = (icon, title, fn, danger) => {
      const b = document.createElement("button");
      b.className = "card-act" + (danger ? " danger" : "");
      b.title = title;
      b.setAttribute("aria-label", title);
      b.innerHTML = icon;
      on(b, "click", (e) => { e.stopPropagation(); fn(); });
      return b;
    };
    acts.append(
      mkAct(ICONS.reuse, "Reuse settings", () => reuseSettings(item)),
      mkAct(ICONS.exp, "Export .nii.gz", () => exportItem(item.id)),
      mkAct(ICONS.del, "Delete", () => deleteItem(item), true),
    );
    card.appendChild(acts);

    const open = () => { exitSweepView(); loadItemIntoViewer(item); };
    on(card, "click", open);
    on(card, "keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    return card;
  }

  function renderShelf(newIds = []) {
    const wrap = byId("shelf-cards");
    wrap.replaceChildren();
    byId("shelf-empty").hidden = store.library.length > 0;
    for (const item of store.library) wrap.appendChild(makeCard(item, newIds.includes(item.id)));
    highlightActiveCard();
    renderShelfMeta();
  }

  function renderShelfMeta() {
    const n = store.library.length;
    const meta = byId("shelf-meta");
    const bytes = n * STORAGE_PER_ITEM;
    const size = bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
    meta.replaceChildren();
    const c1 = document.createElement("span");
    c1.textContent = `${n} ${n === 1 ? "volume" : "volumes"}`;
    const c2 = document.createElement("span");
    c2.textContent = n ? `≈ ${size} on disk` : "";
    meta.append(c1, c2);
  }

  function highlightActiveCard() {
    byId("shelf-cards").querySelectorAll(".shelf-card").forEach((c) =>
      c.classList.toggle("active", c.dataset.id === store.activeId));
  }

  function refreshShelfTimes() {
    byId("shelf-cards").querySelectorAll(".card-time").forEach((el) => {
      el.textContent = fmtAgo(parseFloat(el.dataset.created));
    });
  }

  function reuseSettings(item) {
    const w = store.weights.find((x) => x.file === item.model);
    if (w && w.downloaded) store.params.model = item.model;
    setAge(item.age);
    setSeed(item.seed);
    setSteps(item.steps);
    setCfg(item.cfg_scale);
    setSampler(item.sampler);
    setMorpheus(item.morpheus);
    renderModelSeg();
    updateEstimate();
    toast("Settings restored from this volume", "success");
  }

  async function exportItem(id) {
    const itemId = id || store.activeId;
    if (!itemId) { toast("Nothing to export yet", "info"); return; }
    try {
      const res = await api("/api/export", { id: itemId });
      if (res && res.cancelled) return;
      toast(res && res.path ? `Exported to ${res.path}` : "Exported", "success");
    } catch (e) {
      toast(`Export failed: ${e.message}`, "error");
    }
  }

  async function deleteItem(item) {
    const ok = await confirmDialog({
      title: "Delete this volume?",
      message: `Age ${item.age}, seed ${item.seed}, generated ${fmtAgo(item.created)}. This cannot be undone.`,
    });
    if (!ok) return;
    try {
      await api("/api/library/delete", { id: item.id });
      store.library = store.library.filter((it) => it.id !== item.id);
      if (store.sweep) {
        store.sweep.frames = store.sweep.frames.filter((f) => f.id !== item.id);
        if (store.sweep.frames.length) {
          store.sweep.idx = Math.min(store.sweep.idx, store.sweep.frames.length - 1);
          sweepBarShow();
        } else {
          exitSweepView();
        }
      }
      renderShelf();
      if (store.activeId === item.id) {
        store.activeId = null;
        if (store.library.length) loadItemIntoViewer(store.library[0]);
        else loadHero();  // back to the bundled sample brain
      }
      toast("Volume deleted", "success");
    } catch (e) {
      toast(`Delete failed: ${e.message}`, "error");
    }
  }

  function toggleShelf(force) {
    store.shelfCollapsed = force !== undefined ? force : !store.shelfCollapsed;
    byId("shelf").classList.toggle("collapsed", store.shelfCollapsed);
  }

  function exitSweepView() {
    sweepStop();
    byId("sweep-bar").hidden = true;
  }

  /* ======================================================================
     Section: weights manager (onboarding step 2 + toolbar panel)
     ====================================================================== */
  async function refreshWeights() {
    try {
      store.weights = await api("/api/weights") || [];
    } catch (e) {
      toast(`Could not list weights: ${e.message}`, "error");
    }
    ensureModelSelected();
    renderModelSeg();
    renderWeightsLists();
    updateObFinish();
  }

  function renderWeightsLists() {
    const official = store.weights.filter((w) => w.source !== "custom" && !w.coming_soon);
    const soon = store.weights.filter((w) => w.coming_soon);
    const custom = store.weights.filter((w) => w.source === "custom");
    document.querySelectorAll('[data-role="weights-list"]').forEach((list) => {
      list.replaceChildren();
      const section = (title, items) => {
        if (!items.length) return;
        const h = document.createElement("p");
        h.className = "weights-group-title";
        h.textContent = title;
        list.appendChild(h);
        for (const w of items) list.appendChild(makeWeightCard(w));
      };
      section("Official models", official);
      section("Coming soon", soon);
      section("Your models", custom);
      list.appendChild(makeImportRow());
    });
  }

  function makeImportRow() {
    const row = document.createElement("div");
    row.className = "weights-import";
    const hint = document.createElement("span");
    hint.className = "microcopy";
    hint.textContent = "Have your own WaveDiT checkpoint (new config or dataset)?";
    const btn = document.createElement("button");
    btn.className = "chip-btn import-btn";
    btn.textContent = "Import a checkpoint…";
    on(btn, "click", importModel);
    row.append(hint, btn);
    return row;
  }

  async function importModel() {
    try {
      const res = await api("/api/weights/import", {});
      if (res && res.cancelled) return;
      toast(`Imported ${res.label || res.file}`, "success");
      await refreshWeights();
      if (res.file) {
        store.params.model = res.file;  // select the freshly imported model
        renderModelSeg();
        updateEstimate();
      }
    } catch (e) {
      toast(`Import failed: ${e.message}`, "error");
    }
  }

  function makeWeightCard(w) {
    const card = document.createElement("div");
    card.className = "weight-card";
    if (w.coming_soon) card.classList.add("coming-soon");
    if (w.source === "custom") card.classList.add("custom");
    card.dataset.file = w.file;

    const info = document.createElement("div");
    info.className = "weight-info";
    const name = document.createElement("span");
    name.className = "weight-name";
    name.textContent = `${w.label} (${modelCaption(w)})`;
    const status = document.createElement("span");
    status.className = "weight-status";
    status.dataset.role = "status";
    status.textContent = w.downloading ? "downloading…"
      : w.coming_soon ? "coming soon · not yet on Hugging Face"
      : w.downloaded ? `${w.source === "custom" ? "imported" : "downloaded"}${w.size_mb ? ` · ${fmtMB(w.size_mb)}` : ""}`
      : `available${w.size_mb ? ` · ${fmtMB(w.size_mb)}` : ""}`;
    info.append(name, status);
    const bar = document.createElement("div");
    bar.className = "weight-bar";
    bar.hidden = !w.downloading;
    const fill = document.createElement("div");
    fill.className = "weight-bar-fill";
    fill.dataset.role = "bar";
    bar.appendChild(fill);
    info.appendChild(bar);
    card.appendChild(info);

    const act = document.createElement("div");
    act.className = "weight-act";
    if (w.coming_soon) {
      // Announced variant: no action yet. It flips to a Download button on its own
      // once it lands on the Hub, with no app change.
      const soon = document.createElement("span");
      soon.className = "weight-soon-pill";
      soon.textContent = "soon";
      act.appendChild(soon);
    } else if (w.downloaded) {
      const check = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      check.setAttribute("viewBox", "0 0 16 16");
      check.setAttribute("class", "weight-check");
      check.innerHTML = '<path d="M2.8 8.6 6.2 12l7-8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>';
      act.appendChild(check);
      const del = document.createElement("button");
      del.className = "chip-btn";
      del.textContent = w.source === "custom" ? "Remove" : "Delete";
      on(del, "click", async () => {
        const origin = modal.name; // the confirm replaces the open panel; restore it after
        const sure = await confirmDialog({
          title: `${w.source === "custom" ? "Remove" : "Delete"} ${w.label}?`,
          message: w.source === "custom"
            ? "This removes the imported checkpoint from this Mac."
            : "The checkpoint can always be downloaded again from Hugging Face.",
        });
        if (origin === "weights") openWeightsModal();
        else if (origin === "onboarding") openOnboarding(true);
        if (!sure) return;
        try {
          await api("/api/weights/delete", { file: w.file });
          toast(`${w.label} ${w.source === "custom" ? "removed" : "deleted"}`, "success");
        } catch (e) {
          toast(`Delete failed: ${e.message}`, "error");
        }
        refreshWeights();
      });
      act.appendChild(del);
    } else {
      const dl = document.createElement("button");
      dl.className = "chip-btn";
      dl.textContent = w.downloading ? "Downloading…" : "Download";
      dl.disabled = Boolean(w.downloading);
      on(dl, "click", async () => {
        dl.disabled = true;
        dl.textContent = "Starting…";
        bar.hidden = false;
        try {
          await api("/api/weights/download", { file: w.file });
        } catch (e) {
          toast(`Download failed: ${e.message}`, "error");
          refreshWeights();
        }
      });
      act.appendChild(dl);
    }
    card.appendChild(act);
    return card;
  }

  function onWeightsProgress(d) {
    document.querySelectorAll(`.weight-card[data-file="${CSS.escape(d.file)}"]`).forEach((card) => {
      const bar = card.querySelector('[data-role="bar"]');
      const status = card.querySelector('[data-role="status"]');
      if (bar) { bar.parentElement.hidden = false; bar.style.width = `${clamp(d.pct, 0, 100)}%`; }
      if (status) {
        status.textContent =
          `downloading ${Math.round(d.pct)}% · ${fmtMB(d.mb_done)} of ${fmtMB(d.mb_total)} · ${d.speed_mbps ? d.speed_mbps.toFixed(0) + " MB/s" : ""}`;
      }
      const btn = card.querySelector(".weight-act .chip-btn");
      if (btn) { btn.disabled = true; btn.textContent = "Downloading…"; }
    });
    // mirror the percentage in the sidebar model picker
    const cell = byId("model-seg").querySelector(`.seg-cell[data-file="${CSS.escape(d.file)}"] .seg-caption`);
    if (cell) cell.textContent = `downloading ${Math.round(d.pct)}%`;
  }

  function openWeightsModal() {
    openModal("weights", () => {
      renderWeightsLists();
      refreshWeights(); // an active download's flag may be stale in store.weights
    });
  }

  function updateObFinish() {
    const btn = document.querySelector('[data-action="ob-finish"]');
    if (btn) btn.disabled = !store.weights.some((w) => w.downloaded);
  }

  /* ======================================================================
     Section: onboarding + about modals
     ====================================================================== */
  function openOnboarding(startAtWeights = false) {
    openModal("onboarding", (card) => {
      const s1 = card.querySelector(".ob-step-1");
      const s2 = card.querySelector(".ob-step-2");
      const showWeightsStep = () => {
        s1.hidden = true;
        s2.hidden = false;
        renderWeightsLists();
        refreshWeights();
        updateObFinish();
      };
      on(card.querySelector('[data-action="ob-next"]'), "click", showWeightsStep);
      if (startAtWeights) showWeightsStep();
      on(card.querySelector('[data-action="ob-finish"]'), "click", async () => {
        try {
          store.settings = await api("/api/settings", { onboarding_done: true }) || store.settings;
        } catch (e) {
          toast(`Could not save settings: ${e.message}`, "error");
        }
        closeModal();
        toast("All set. Pick an age and press Generate.", "success");
      });
    });
  }

  function openAbout() {
    openModal("about", (card) => {
      card.querySelector('[data-role="about-version"]').textContent =
        `version ${store.version || "?"} · research preview`;
      on(card.querySelector('[data-action="copy-bibtex"]'), "click", async () => {
        const ok = await copyText(card.querySelector('[data-role="bibtex"]').textContent);
        toast(ok ? "BibTeX copied" : "Copy failed", ok ? "success" : "error");
      });
    });
  }

  /* ======================================================================
     Section: presets
     ====================================================================== */
  function applyPreset(name) {
    switch (name) {
      case "child": setAge(8); break;
      case "adult": setAge(35); break;
      case "elder": setAge(82); break;
      case "showcase": {
        const fp = store.weights.find((w) => w.label === "FinePatch");
        if (!fp || !fp.downloaded) {
          toast("Showcase wants the FinePatch checkpoint", "info");
          openWeightsModal();
          return;
        }
        store.params.model = fp.file;
        renderModelSeg();
        setAge(72);
        setSeed(42);
        setSteps(100);
        toast("Showcase loaded: FinePatch, 100 steps. Worth the wait.", "info");
        break;
      }
      case "surprise": {
        const [lo, hi] = store.ageRange;
        setAge(lo + Math.floor(Math.random() * (hi - lo + 1)));
        setSeed(randSeed());
        const cm = SURPRISE_COLORMAPS[Math.floor(Math.random() * SURPRISE_COLORMAPS.length)];
        byId("colormap-select").value = cm;
        setColormap(cm);
        toast(`Surprise: age ${store.params.age}, ${cm} colors`, "info");
        break;
      }
      default: break;
    }
    updateEstimate();
  }

  /* ======================================================================
     Section: SSE client with backoff + resync
     ====================================================================== */
  let es = null;
  let esBackoff = 1000;

  function connectSSE() {
    try { es = new EventSource("/api/events"); }
    catch (_) { setTimeout(connectSSE, esBackoff); return; }

    es.onopen = () => { esBackoff = 1000; resync(); };
    es.onerror = () => {
      try { es.close(); } catch (_) { /* already closed */ }
      setTimeout(connectSSE, esBackoff);
      esBackoff = Math.min(esBackoff * 2, 15000);
    };

    const onEv = (name, fn) => es.addEventListener(name, (e) => {
      let d = null;
      try { d = JSON.parse(e.data); } catch (_) { return; }
      try { fn(d); } catch (err) { console.error(`SSE ${name} handler`, err); }
    });

    onEv("gen_start", (d) => {
      store.jobId = d.job_id;
      if (store.jobKind !== "sweep") store.jobKind = "gen";
      setBusy(true);
      if (store.jobKind === "gen") {
        const w = store.weights.find((x) => x.file === (d.params && d.params.model));
        overlayShow(w ? `WaveDiT ${w.label}` : "WaveDiT", `step 0 of ${d.nfe_total}`);
      }
    });

    onEv("gen_progress", (d) => {
      byId("gen-step").textContent = `step ${d.nfe_done} of ${d.nfe_total}`;
      if (typeof d.mem_gb === "number") updateLoadMeter(d.mem_gb);
      if (store.jobKind !== "sweep") progressTick(d.pct, d.eta_s);
    });

    onEv("gen_done", (d) => {
      setBusy(false);
      store.jobId = null;
      store.jobKind = null;
      overlayHide();
      const item = d.item;
      if (item) {
        store.library.unshift(item);
        store.lastSeed = item.seed;
        renderShelf([item.id]);
        exitSweepView();
        loadItemIntoViewer(item);
        if (typeof item.peak_mem_gb === "number" && item.peak_mem_gb > 0) {
          const badge = byId("memory-badge");
          badge.textContent = `peak ${fmtGB(item.peak_mem_gb)}`;
          badge.hidden = false;
        }
        if (predictedPrecision() === "bf16" && item.precision === "float32" && store.deviceType !== "cpu") {
          toast("bf16 was unstable here, this run fell back to float32", "info");
        }
      }
      softStateRefresh();
    });

    onEv("gen_error", (d) => {
      setBusy(false);
      store.jobId = null;
      store.jobKind = null;
      overlayHide();
      if (d.cancelled) toast("Cancelled", "info");
      else toast(`Generation failed: ${d.message}`, "error");
      if (!store.library.length && !store.heroShown) loadHero();
      softStateRefresh();
    });

    onEv("sweep_start", (d) => {
      store.jobId = d.job_id;
      store.jobKind = "sweep";
      setBusy(true);
      sweepReset(d.frames);
      const w = store.weights.find((x) => x.file === (d.params && d.params.model));
      overlayShow(w ? `WaveDiT ${w.label} · time-lapse` : "time-lapse", `frame 1 of ${d.frames}`);
    });

    onEv("sweep_progress", (d) => {
      progressTick(d.pct, d.eta_s);
      if (typeof d.mem_gb === "number") updateLoadMeter(d.mem_gb);
      byId("gen-step").textContent = `frame ${d.frame + 1} of ${d.frames}`;
    });

    onEv("sweep_frame_done", (d) => {
      if (!store.sweep) sweepReset(0);
      store.sweep.frames.push(d.item);
      store.library.unshift(d.item);
      renderShelf([d.item.id]);
      sweepBarShow();
      if (store.sweep.frames.length === 1) loadItemIntoViewer(d.item, { crossfade: false });
      warmFetch(d.item.vol_url); // keep scrubbing snappy once the sweep lands
    });

    onEv("sweep_done", () => {
      setBusy(false);
      store.jobId = null;
      store.jobKind = null;
      overlayHide();
      sweepBarShow();
      sweepShowFrame(0);
      if (!REDUCED) sweepPlay();
      softStateRefresh();
    });

    onEv("weights_progress", onWeightsProgress);
    onEv("weights_done", (d) => {
      toast(`${d.file} downloaded`, "success");
      refreshWeights();
    });
    onEv("weights_error", (d) => {
      toast(`Weights download failed: ${d.message}`, "error");
      refreshWeights();
    });
  }

  // After each run: pick up fresh calibration (per-NFE timing) and settings (bf16_ok).
  async function softStateRefresh() {
    try {
      const st = await api("/api/state");
      store.calibration = st.calibration || {};
      store.settings = st.settings || {};
      updateEstimate();
    } catch (_) { /* transient */ }
  }

  async function resync() {
    try {
      const st = await api("/api/state");
      hydrate(st, { quiet: true });
      const lib = await api("/api/library");
      const hadActive = store.activeId;
      store.library = lib || [];
      renderShelf();
      // Reconcile job state in both directions: a gen_done/gen_error missed
      // during the SSE outage must not leave a stale jobId/jobKind around.
      if (!st.busy && store.busy) {
        setBusy(false);
        overlayHide();
        store.jobId = null;
        store.jobKind = null;
      } else if (st.busy && !store.busy) {
        setBusy(true);
      }
      if (!hadActive && store.library.length && !store.busy) loadItemIntoViewer(store.library[0]);
    } catch (_) { /* server warming up; SSE will retry */ }
  }

  /* ======================================================================
     Section: hydration from /api/state
     ====================================================================== */
  function hydrate(st, { quiet = false } = {}) {
    store.version = st.version || store.version;
    byId("version-badge").textContent = `v${store.version || "?"}`;

    const dev = st.device || {};
    store.deviceType = dev.device || dev.type || "cpu";
    const devName = {
      mps: "Apple Silicon · MPS",
      cuda: "GPU · CUDA",
      cpu: "CPU",
    }[store.deviceType] || store.deviceType;
    byId("device-name").textContent = devName;
    byId("device-chip").title = [dev.chip, dev.ram_gb ? `${dev.ram_gb} GB RAM` : null,
      dev.torch ? `torch ${dev.torch}` : null].filter(Boolean).join(" · ") || "Compute device";

    store.ramGb = Number(dev.ram_gb) || 0;
    byId("load-label").textContent = store.deviceType === "cpu" ? "CPU" : "GPU";
    if (typeof st.mem_now_gb === "number") updateLoadMeter(st.mem_now_gb);

    if (Array.isArray(st.age_range) && st.age_range.length === 2) {
      store.ageRange = [Math.round(st.age_range[0]), Math.round(st.age_range[1])];
      const [lo, hi] = store.ageRange;
      const ageSlider = byId("age-slider");
      ageSlider.min = String(lo);
      ageSlider.max = String(hi);
      for (const id of ["sweep-start", "sweep-end"]) {
        const inp = byId(id);
        inp.min = String(lo);
        inp.max = String(hi);
        inp.value = String(clamp(parseInt(inp.value, 10) || lo, lo, hi));
      }
      setAge(store.params.age);
    }

    store.settings = st.settings || {};
    store.calibration = st.calibration || {};
    store.weights = st.weights || [];
    ensureModelSelected();
    renderModelSeg();
    renderWeightsLists();
    updateObFinish();
    updateEstimate();

    if (!quiet && !store.settings.onboarding_done) openOnboarding();
  }

  /* ======================================================================
     Section: event wiring
     ====================================================================== */
  function bindUI() {
    // age
    const ageSlider = byId("age-slider");
    on(ageSlider, "input", () => setAge(parseInt(ageSlider.value, 10), true));
    on(byId("age-dice"), "click", () => {
      const [lo, hi] = store.ageRange;
      setAge(lo + Math.floor(Math.random() * (hi - lo + 1)));
    });

    // seed
    const seedInput = byId("seed-input");
    on(seedInput, "change", () => setSeed(parseInt(seedInput.value, 10) || 0));
    on(byId("seed-dice"), "click", () => setSeed(randSeed()));
    on(byId("seed-last"), "click", () => {
      if (store.lastSeed === null) { toast("No previous generation yet", "info"); return; }
      setSeed(store.lastSeed);
    });

    // steps / cfg / sampler / morpheus
    const stepsSlider = byId("steps-slider");
    on(stepsSlider, "input", () => setSteps(parseInt(stepsSlider.value, 10), true));
    const cfgSlider = byId("cfg-slider");
    on(cfgSlider, "input", () => setCfg(parseFloat(cfgSlider.value), true));
    byId("sampler-seg").querySelectorAll(".seg-cell").forEach((c) =>
      on(c, "click", () => setSampler(c.dataset.sampler)));
    const morSlider = byId("morpheus-slider");
    on(morSlider, "input", () => setMorpheus(parseFloat(morSlider.value), true));

    // generate / cancel
    on(byId("generate-btn"), "click", startGenerate);
    on(byId("empty-generate-btn"), "click", startGenerate);
    on(byId("cancel-btn"), "click", cancelJob);

    // presets
    byId("presets-row").querySelectorAll("[data-preset]").forEach((b) =>
      on(b, "click", () => applyPreset(b.dataset.preset)));

    // sweep panel
    const framesSlider = byId("sweep-frames");
    on(framesSlider, "input", () => {
      byId("sweep-frames-value").textContent = framesSlider.value;
      setFill(framesSlider);
    });
    on(byId("sweep-run"), "click", startSweep);

    // sweep scrubber
    on(byId("sweep-play"), "click", () => {
      if (store.sweep && store.sweep.playing) sweepStop(); else sweepPlay();
    });
    const scrub = byId("sweep-scrub");
    on(scrub, "input", () => {
      sweepStop();
      setFill(scrub);
      sweepShowFrame(parseInt(scrub.value, 10), { fromScrub: true });
    });
    on(byId("sweep-loop"), "click", () => {
      if (!store.sweep) return;
      store.sweep.loop = !store.sweep.loop;
      byId("sweep-loop").setAttribute("aria-pressed", store.sweep.loop ? "true" : "false");
    });
    on(byId("sweep-close"), "click", exitSweepView);

    // viewer toolbar
    on(byId("colormap-select"), "change", (e) => setColormap(e.target.value));
    const gammaSlider = byId("gamma-slider");
    on(gammaSlider, "input", () => { setFill(gammaSlider); setGamma(parseFloat(gammaSlider.value)); });
    on(byId("layout-btn"), "click", cycleLayout);
    on(byId("guides-btn"), "click", () => setGuides());
    on(byId("reset-view-btn"), "click", resetView);
    on(byId("fullscreen-btn"), "click", () => toggleImmersive());
    on(byId("export-btn"), "click", () => exportItem());
    on(byId("fallback-export-btn"), "click", () => exportItem());
    on(byId("copy-prov-btn"), "click", copyProvenance);
    on(byId("provenance-bar"), "click", copyProvenance);
    on(byId("weights-btn"), "click", openWeightsModal);

    // shelf
    on(byId("shelf-toggle"), "click", () => toggleShelf());

    // about
    on(byId("about-link"), "click", (e) => { e.preventDefault(); openAbout(); });

    // keyboard shortcuts (esc handled by the modal section)
    on(window, "keydown", (e) => {
      if (!(e.metaKey || e.ctrlKey)) return;
      if (e.key === "Enter") { e.preventDefault(); startGenerate(); }
      else if (e.key.toLowerCase() === "e") { e.preventDefault(); exportItem(); }
      else if (e.key.toLowerCase() === "l") { e.preventDefault(); toggleShelf(); }
    });

    // initial slider fills
    ["age-slider", "steps-slider", "cfg-slider", "morpheus-slider", "sweep-frames", "gamma-slider"]
      .forEach((id) => setFill(byId(id)));

    setInterval(refreshShelfTimes, 45000);
  }

  /* ======================================================================
     Section: boot
     ====================================================================== */
  async function init() {
    bindUI();
    setAge(store.params.age);
    setSeed(store.params.seed);
    setSteps(store.params.steps);
    setCfg(store.params.cfg_scale);
    setSampler(store.params.sampler);
    setMorpheus(store.params.morpheus);
    updateProvenance(null);
    renderShelfMeta();
    byId("shelf-empty").hidden = false;

    initViewer(); // async, independent of the API

    try {
      const st = await api("/api/state");
      hydrate(st);
      if (st.busy) setBusy(true);
    } catch (e) {
      toast(`Could not reach the engine: ${e.message}`, "error");
    }

    try {
      store.library = await api("/api/library") || [];
      renderShelf();
      if (store.library.length) loadItemIntoViewer(store.library[0], { crossfade: false });
      else loadHero();  // open on the bundled sample brain, not an empty stage
    } catch (e) {
      toast(`Could not load the library: ${e.message}`, "error");
    }

    connectSSE();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
