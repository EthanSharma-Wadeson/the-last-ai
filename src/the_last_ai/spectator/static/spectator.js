/* Live spectator renderer — displays Python simulation state only. */
(() => {
  const canvas = document.getElementById("world");
  const ctx = canvas.getContext("2d");

  const state = {
    world: { width: 20, height: 12, walls: [] },
    snapshot: null,
    follow: true,
    camX: 0,
    camY: 0,
    zoom: 28,
    dragging: false,
    lastMouse: null,
    messages: [],
    events: [],
    life: [],
    lastMetrics: null,
    flashLinks: [],
    commFilter: "all",
    dim: true,
  };

  const METRIC_KEYS = [
    ["energy_pct", "Energy", (v) => `${Number(v).toFixed(0)}%`],
    ["social_drive", "Social drive", (v) => Number(v).toFixed(3)],
    ["exploration", "Exploration", (v) => Number(v).toFixed(3)],
    ["uncertainty", "Uncertainty", (v) => Number(v).toFixed(3)],
    ["security", "Security", (v) => Number(v).toFixed(3)],
    ["novelty", "Novelty", (v) => Number(v).toFixed(3)],
    ["prediction_error", "Prediction error", (v) => Number(v).toFixed(3)],
    ["social_loss", "Social loss", (v) => Number(v).toFixed(3)],
    ["goal_success", "Goal success", (v) => Number(v).toFixed(3)],
    ["search_pressure", "Search pressure", (v) => Number(v).toFixed(3)],
    ["memories", "Memories", (v) => String(v)],
    ["relationships", "Relationships", (v) => String(v)],
    ["historical_entities", "Historical entities", (v) => String(v)],
    ["communication_memories", "Communication mem", (v) => String(v)],
    ["tracked_entities", "Tracked entities", (v) => String(v)],
    ["seek_attempts", "Seek attempts", (v) => String(v)],
  ];

  function resizeCanvas() {
    const parent = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = parent.clientWidth;
    const h = Math.max(320, parent.clientHeight - 42);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function worldToScreen(x, y) {
    const cx = canvas.clientWidth / 2;
    const cy = canvas.clientHeight / 2;
    return {
      x: cx + (x - state.camX) * state.zoom,
      y: cy + (y - state.camY) * state.zoom,
    };
  }

  function screenToWorld(sx, sy) {
    const cx = canvas.clientWidth / 2;
    const cy = canvas.clientHeight / 2;
    return {
      x: state.camX + (sx - cx) / state.zoom,
      y: state.camY + (sy - cy) / state.zoom,
    };
  }

  function followSpectated() {
    if (!state.follow || !state.snapshot) return;
    const a = (state.snapshot.agents || []).find((x) => x.spectated);
    if (!a) return;
    state.camX = a.x + 0.5;
    state.camY = a.y + 0.5;
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0d0f14";
    ctx.fillRect(0, 0, w, h);

    const ww = state.world.width || 20;
    const wh = state.world.height || 12;

    if (state.zoom >= 14) {
      ctx.strokeStyle = "#1c2130";
      ctx.lineWidth = 1;
      for (let x = 0; x <= ww; x++) {
        const a = worldToScreen(x, 0);
        const b = worldToScreen(x, wh);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      for (let y = 0; y <= wh; y++) {
        const a = worldToScreen(0, y);
        const b = worldToScreen(ww, y);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }

    ctx.fillStyle = "#3a4154";
    for (const [x, y] of state.world.walls || []) {
      const p = worldToScreen(x, y);
      ctx.fillRect(p.x, p.y, state.zoom + 0.5, state.zoom + 0.5);
    }

    for (const [x, y] of (state.snapshot && state.snapshot.resources) || []) {
      const p = worldToScreen(x + 0.5, y + 0.5);
      const r = Math.max(2, state.zoom * 0.22);
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = "#6f9f5a";
      ctx.fill();
      ctx.strokeStyle = "#cfe8c4";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Spotlight: last-known locations
    for (const spot of (state.snapshot && state.snapshot.spotlight) || []) {
      const p = worldToScreen(spot.x + 0.5, spot.y + 0.5);
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 280);
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(8, state.zoom * 0.55), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(232,194,122,${0.35 + 0.35 * pulse})`;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(232,194,122,0.12)";
      ctx.fill();
      if (state.zoom >= 14) {
        ctx.fillStyle = "#e8c27a";
        ctx.font = "11px IBM Plex Sans, system-ui";
        ctx.fillText(`expected: ${spot.entity_id}`, p.x + 10, p.y - 8);
      }
    }

    const links = [
      ...((state.snapshot && state.snapshot.social_links) || []),
      ...state.flashLinks,
    ];
    const byId = Object.fromEntries(((state.snapshot && state.snapshot.agents) || []).map((a) => [a.id, a]));
    for (const link of links) {
      const a = byId[link.a];
      const b = byId[link.b];
      if (!a || !b) continue;
      const pa = worldToScreen(a.x + 0.5, a.y + 0.5);
      const pb = worldToScreen(b.x + 0.5, b.y + 0.5);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.strokeStyle = link.kind === "communicate" ? "#4a9fd8" : "#5a6275";
      ctx.lineWidth = link.kind === "communicate" ? 2 : 1;
      ctx.stroke();
    }

    const agents = (state.snapshot && state.snapshot.agents) || [];
    const dimOthers = state.dim && (state.snapshot && state.snapshot.cinema_dim !== false);
    for (const agent of agents) {
      const p = worldToScreen(agent.x + 0.5, agent.y + 0.5);
      const r = Math.max(2.5, state.zoom * (agent.spectated ? 0.34 : 0.22));
      const alpha = dimOthers && agent.dim ? 0.28 : 1;

      if (agent.spectated) {
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 5, 0, Math.PI * 2);
        ctx.strokeStyle = "#f0f3f8";
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = "#e8eaef";
        ctx.fill();
        ctx.strokeStyle = "#12141a";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        if (state.zoom >= 14) {
          ctx.fillStyle = "#f0f3f8";
          ctx.font = "12px IBM Plex Sans, system-ui";
          ctx.fillText(agent.id, p.x + r + 6, p.y + 4);
        }
      } else {
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = agent.significant_partner ? "#d7b56d" : "#8b93a7";
        ctx.fill();
        if (agent.significant_partner) {
          ctx.strokeStyle = "#e8c27a";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        if (state.zoom >= 26 && !agent.dim) {
          ctx.fillStyle = "#9aa3b2";
          ctx.font = "10px IBM Plex Sans, system-ui";
          ctx.fillText(agent.id.replace("agent_", ""), p.x + r + 3, p.y + 3);
        }
      }
      ctx.globalAlpha = 1;
    }
  }

  function fmtTime(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    const m = String(Math.floor(sec / 60)).padStart(2, "0");
    const s = String(sec % 60).padStart(2, "0");
    return `${m}:${s}`;
  }

  function renderChain(chain) {
    const ol = document.getElementById("causalChain");
    const summary = document.getElementById("chainSummary");
    if (!chain) {
      summary.textContent = "No disappearance-linked chain yet.";
      ol.innerHTML = "";
      return;
    }
    summary.textContent = chain.summary || "";
    ol.innerHTML = (chain.steps || []).map((s) =>
      `<li class="${s.active ? "on" : "off"}" data-step='${escapeAttr(JSON.stringify(s))}'>
        <strong>${escapeHtml(s.label)}</strong>
        <div class="muted small">${escapeHtml(s.detail || "")}</div>
      </li>`
    ).join("");
    ol.querySelectorAll("li").forEach((li) => {
      li.onclick = () => {
        document.getElementById("msgDetailBody").textContent =
          JSON.stringify(JSON.parse(li.dataset.step), null, 2);
      };
    });
  }

  function renderPatterns(patterns) {
    const bar = document.getElementById("patternBar");
    bar.innerHTML = (patterns || []).map((p) =>
      `<span class="pattern-pill" title="${escapeAttr(p.evidence || "")}">${escapeHtml(p.label)}</span>`
    ).join("");
  }

  function renderLife(rows) {
    const feed = document.getElementById("lifeFeed");
    // Replace with latest rolling list from snapshot (authoritative)
    feed.innerHTML = "";
    const list = [...(rows || [])].reverse();
    for (const row of list.slice(0, 30)) {
      const el = document.createElement("div");
      el.className = "life-item";
      el.textContent = `t${row.tick} · ${row.summary}`;
      el.onclick = () => {
        document.getElementById("msgDetailBody").textContent = JSON.stringify(row, null, 2);
      };
      feed.appendChild(el);
    }
  }

  function sparkPath(values, width, height) {
    if (!values.length) return "";
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(1e-6, max - min);
    return values.map((v, i) => {
      const x = values.length === 1 ? 0 : (i / (values.length - 1)) * width;
      const y = height - ((v - min) / span) * (height - 2) - 1;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  }

  function renderSparks(series) {
    const row = document.getElementById("sparkRow");
    if (!row) return;
    const keys = [
      ["social_loss", "Social loss"],
      ["prediction_error", "Pred. error"],
      ["search_pressure", "Search"],
    ];
    const pts = series || [];
    row.innerHTML = keys.map(([key, label]) => {
      const vals = pts.map((p) => Number(p[key] || 0));
      const last = vals.length ? vals[vals.length - 1].toFixed(2) : "—";
      const d = sparkPath(vals, 110, 28);
      return `<div class="spark"><label>${label} ${last}</label>
        <svg viewBox="0 0 110 28" preserveAspectRatio="none">
          <path d="${d}" fill="none" stroke="#3d9b8f" stroke-width="1.5"/>
        </svg></div>`;
    }).join("");
  }

  function renderBehaviourRates(rates) {
    const el = document.getElementById("behaviourRates");
    if (!el) return;
    if (!rates || (!rates.delta_seek && !rates.delta_messages && !rates.seek_attempts)) {
      el.textContent = "";
      return;
    }
    const bits = [];
    if (rates.delta_seek != null) bits.push(`Δseek ${rates.delta_seek >= 0 ? "+" : ""}${rates.delta_seek}`);
    if (rates.delta_messages != null) bits.push(`Δmessages ${rates.delta_messages >= 0 ? "+" : ""}${rates.delta_messages}`);
    if (rates.delta_investigate != null) bits.push(`Δinvestigate ${rates.delta_investigate >= 0 ? "+" : ""}${rates.delta_investigate}`);
    el.textContent = bits.length ? `Behaviour since baseline: ${bits.join(" · ")}` : "";
  }

  function renderContrast(contrast) {
    const el = document.getElementById("contrastPanel");
    if (!el) return;
    if (!contrast || (!contrast.friend && !contrast.stranger)) {
      el.classList.add("hidden");
      el.innerHTML = "";
      return;
    }
    el.classList.remove("hidden");
    const fmtSide = (side, title) => {
      if (!side) return `<div class="contrast-col"><strong>${title}</strong><div class="muted">pending…</div></div>`;
      const m = side.metrics || {};
      const d = side.deltas || {};
      const br = side.behaviour_rates || {};
      const lossD = d.social_loss ? Number(d.social_loss.delta).toFixed(2) : "—";
      const peD = d.prediction_error ? Number(d.prediction_error.delta).toFixed(2) : "—";
      return `<div class="contrast-col">
        <strong>${title}</strong>
        <div>social_loss ${Number(m.social_loss || 0).toFixed(2)} (Δ ${lossD})</div>
        <div>pred_error ${Number(m.prediction_error || 0).toFixed(2)} (Δ ${peD})</div>
        <div>Δseek ${br.delta_seek ?? "—"} · Δmsg ${br.delta_messages ?? "—"}</div>
      </div>`;
    };
    el.innerHTML = `<h3>Friend vs stranger contrast</h3>
      <div class="contrast-grid">
        ${fmtSide(contrast.friend, "Friend removal")}
        ${fmtSide(contrast.stranger, "Stranger removal")}
      </div>
      <p class="muted small">${escapeHtml(contrast.note || "Computational comparison only.")}</p>`;
  }

  function updateChrome(snap) {
    document.getElementById("pop").textContent = String(snap.population ?? "—");
    document.getElementById("tick").textContent = String(snap.tick ?? 0);
    document.getElementById("etime").textContent = fmtTime(snap.elapsed_seconds);
    document.getElementById("speedLabel").textContent = `${snap.speed}×`;
    document.getElementById("phase").textContent = snap.phase || "—";
    document.getElementById("spectating").textContent = snap.spectating || "—";
    document.getElementById("btnPause").textContent = snap.paused ? "Resume" : "Pause";
    document.getElementById("metricsTitle").textContent =
      `${(snap.spectating || "AGENT").toUpperCase()} — LIVE`;

    const cd = snap.countdown || {};
    const cdEl = document.getElementById("countdown");
    if (cd.ticks_remaining != null) {
      cdEl.textContent = `${cd.ticks_remaining} → ${cd.next || ""}`;
      document.getElementById("countdownChip").title = cd.detail || "";
    } else {
      cdEl.textContent = cd.label || "—";
    }

    document.getElementById("partnerLine").textContent = snap.significant_partner
      ? `Focus relationship: ${snap.significant_partner}`
      : "";

    const m = snap.metrics;
    const deltas = snap.deltas || {};
    const grid = document.getElementById("metricGrid");
    if (!m) {
      grid.innerHTML = "";
      return;
    }
    document.getElementById("affectLabel").textContent = m.affect_label
      ? `Computational state: ${m.affect_label}`
      : "";
    document.getElementById("disclaimer").textContent =
      snap.disclaimer || m.disclaimer || "";

    renderSparks(snap.metric_series);
    renderBehaviourRates(snap.behaviour_rates);
    renderContrast(snap.contrast);

    grid.innerHTML = METRIC_KEYS.map(([key, label, fmt]) => {
      const val = m[key];
      const d = deltas[key];
      let cls = "";
      let deltaHtml = "";
      if (d) {
        cls = d.direction === "up" ? "up" : (d.direction === "down" ? "down" : "");
        const arrow = d.direction === "up" ? "↑" : (d.direction === "down" ? "↓" : "·");
        deltaHtml = `<span class="delta">${arrow} ${Number(d.baseline).toFixed(2)} → ${Number(d.current).toFixed(2)}</span>`;
      } else if (state.lastMetrics && typeof val === "number" && typeof state.lastMetrics[key] === "number") {
        if (val > state.lastMetrics[key] + 0.05 && (key === "social_loss" || key === "prediction_error" || key === "search_pressure")) {
          cls = "up";
        }
      }
      return `<div class="metric ${cls}">${label}<strong>${fmt(val)}</strong>${deltaHtml}</div>`;
    }).join("");
    state.lastMetrics = { ...m };

    renderChain(snap.causal_chain);
    renderPatterns(snap.patterns);
    renderLife(snap.life_timeline);

    if (snap.finished) {
      const link = document.getElementById("openLife");
      link.classList.remove("hidden");
      const aid = snap.spectating || "agent_000";
      link.href = `/observatory/${aid}_life.html`;
      link.textContent = `Open ${aid} life`;
    }
  }

  function applyCommFilter() {
    const filter = state.commFilter;
    document.querySelectorAll("#commFeed .comm-item").forEach((el) => {
      const hl = el.dataset.highlight || "other";
      const text = (el.dataset.text || "").toLowerCase();
      const absence = el.dataset.absence === "1";
      let show = true;
      if (filter === "spectated") show = hl === "from_spectated" || hl === "to_spectated";
      if (filter === "food") show = text.includes("food");
      if (filter === "absence") show = absence;
      el.classList.toggle("hidden-filter", !show);
    });
  }

  function pushComm(msg) {
    state.messages.unshift(msg);
    if (state.messages.length > 50) state.messages.length = 50;
    const feed = document.getElementById("commFeed");
    const el = document.createElement("div");
    const absence = !!msg.absence_driven;
    el.className = `comm-item ${msg.highlight || "other"}${absence ? " absence-driven" : ""}`;
    el.dataset.highlight = msg.highlight || "other";
    el.dataset.text = msg.text || "";
    el.dataset.absence = absence ? "1" : "0";
    let tag = "";
    if (absence) tag += '<span class="tag absence">ABSENCE</span>';
    if (msg.highlight === "from_spectated") tag += '<span class="tag from">FROM SPECTATED</span>';
    if (msg.highlight === "to_spectated") tag += '<span class="tag to">TO SPECTATED</span>';
    const why = msg.construction_reason
      ? `<div class="muted small">${escapeHtml(msg.construction_reason)}${msg.about_entity_id ? ` · about ${escapeHtml(msg.about_entity_id)}` : ""}</div>`
      : "";
    el.innerHTML = `${tag}<div>t${msg.tick} · ${msg.sender} → ${msg.receiver}</div><div>"${escapeHtml(msg.text || "")}"</div>${why}`;
    el.addEventListener("click", () => {
      document.getElementById("msgDetailBody").textContent = JSON.stringify(msg, null, 2);
      const agents = (state.snapshot && state.snapshot.agents) || [];
      const target = agents.find((a) => a.id === msg.sender) || agents.find((a) => a.id === msg.receiver);
      if (target) {
        state.follow = false;
        document.getElementById("btnFollow").textContent = "Follow";
        state.camX = target.x + 0.5;
        state.camY = target.y + 0.5;
        draw();
      }
    });
    feed.prepend(el);
    while (feed.children.length > 50) feed.removeChild(feed.lastChild);
    applyCommFilter();

    if (msg.sender_position && msg.receiver_position) {
      state.flashLinks.push({
        a: msg.sender, b: msg.receiver, kind: "communicate", tick: msg.tick,
      });
      if (state.flashLinks.length > 8) state.flashLinks.shift();
      setTimeout(() => {
        state.flashLinks = state.flashLinks.filter((l) => !(l.tick === msg.tick && l.a === msg.sender));
        draw();
      }, 900);
    }
  }

  function pushEvent(ev) {
    const feed = document.getElementById("eventFeed");
    const el = document.createElement("div");
    el.className = "ev-item";
    el.textContent = `t${ev.tick ?? "?"} · ${ev.summary || ev.kind || ev.type}`;
    el.onclick = () => {
      document.getElementById("msgDetailBody").textContent = JSON.stringify(ev, null, 2);
    };
    feed.prepend(el);
    while (feed.children.length > 40) feed.removeChild(feed.lastChild);
  }

  function showToast(text, cinema = false) {
    const t = document.getElementById("toast");
    t.textContent = text;
    t.classList.toggle("cinema", !!cinema);
    t.classList.remove("hidden");
    clearTimeout(showToast._tm);
    showToast._tm = setTimeout(() => t.classList.add("hidden"), 3200);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/`/g, "");
  }

  function handleEvent(ev) {
    if (!ev || !ev.type) return;
    if (ev.type === "world") { state.world = ev; return; }
    if (ev.type === "snapshot") {
      state.snapshot = ev;
      if (typeof ev.cinema_dim === "boolean") state.dim = ev.cinema_dim;
      followSpectated();
      updateChrome(ev);
      draw();
      return;
    }
    if (ev.type === "communication") { pushComm(ev); draw(); return; }
    if (ev.type === "cinema") {
      showToast(ev.summary || ev.kind, true);
      pushEvent(ev);
      return;
    }
    if (ev.type === "event" || ev.type === "population" || ev.type === "spectate") {
      pushEvent(ev);
      if (ev.kind === "disappearance" || ev.type === "population") {
        showToast(ev.summary || `Population: ${ev.population}`);
      }
      if (ev.type === "spectate" && ev.agent_id) {
        document.getElementById("spectating").textContent = ev.agent_id;
      }
      return;
    }
    if (ev.type === "finished") {
      pushEvent({ tick: ev.tick, summary: "Simulation finished", kind: "finished" });
      if (state.snapshot) {
        state.snapshot.finished = true;
        updateChrome({ ...state.snapshot, ...ev, finished: true });
      }
      showToast("Run complete — open life observatory", true);
      return;
    }
    if (ev.type === "camera" && ev.action === "reset") {
      state.zoom = 28;
      followSpectated();
      draw();
    }
  }

  async function postControl(action, extra = {}) {
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...extra }),
    });
    const snap = await res.json();
    if (snap && snap.type === "snapshot") handleEvent(snap);
  }

  function connectSSE() {
    const es = new EventSource("/events");
    es.onmessage = (e) => {
      try { handleEvent(JSON.parse(e.data)); } catch (_) {}
    };
  }

  document.getElementById("btnPause").onclick = () => postControl("toggle_pause");
  document.getElementById("btnStep").onclick = () => postControl("step");
  document.getElementById("speedSelect").onchange = (e) =>
    postControl("speed", { value: Number(e.target.value) });
  document.getElementById("btnFollow").onclick = () => {
    state.follow = !state.follow;
    document.getElementById("btnFollow").textContent = state.follow ? "Unfollow" : "Follow";
    followSpectated();
    draw();
  };
  document.getElementById("btnDim").onclick = () => {
    postControl("toggle_dim");
    state.dim = !state.dim;
    document.getElementById("btnDim").textContent = state.dim ? "Show all" : "Dim others";
    draw();
  };
  document.getElementById("btnResetCam").onclick = () => {
    state.zoom = 28;
    state.follow = true;
    document.getElementById("btnFollow").textContent = "Unfollow";
    followSpectated();
    draw();
  };

  document.querySelectorAll(".filter").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".filter").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.commFilter = btn.dataset.filter;
      applyCommFilter();
    };
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    state.zoom = Math.max(6, Math.min(64, state.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
    draw();
  }, { passive: false });

  canvas.addEventListener("mousedown", (e) => {
    state.dragging = true;
    state.lastMouse = { x: e.offsetX, y: e.offsetY };
    canvas.classList.add("dragging");
  });
  window.addEventListener("mouseup", () => {
    state.dragging = false;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("mousemove", (e) => {
    if (!state.dragging || !state.lastMouse) return;
    state.follow = false;
    document.getElementById("btnFollow").textContent = "Follow";
    state.camX -= (e.offsetX - state.lastMouse.x) / state.zoom;
    state.camY -= (e.offsetY - state.lastMouse.y) / state.zoom;
    state.lastMouse = { x: e.offsetX, y: e.offsetY };
    draw();
  });

  canvas.addEventListener("click", (e) => {
    const wpt = screenToWorld(e.offsetX, e.offsetY);
    const agents = (state.snapshot && state.snapshot.agents) || [];
    let best = null;
    let bestD = 1.15;
    for (const a of agents) {
      const d = Math.hypot(a.x + 0.5 - wpt.x, a.y + 0.5 - wpt.y);
      if (d < bestD) { bestD = d; best = a; }
    }
    if (best) {
      state.follow = true;
      document.getElementById("btnFollow").textContent = "Unfollow";
      postControl("spectate", { agent_id: best.id });
    }
  });

  // Soft pulse for spotlight without needing new frames from server
  setInterval(() => {
    if ((state.snapshot && state.snapshot.spotlight || []).length) draw();
  }, 120);

  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  connectSSE();
  fetch("/api/state").then((r) => r.json()).then(handleEvent).catch(() => {});
  fetch("/api/world").then((r) => r.json()).then(handleEvent).catch(() => {});
})();
