"""Export agent life histories as authoritative JSON + meticulously readable biography."""

from __future__ import annotations

import html
import json
from pathlib import Path

from the_last_ai.observatory.affect import affect_label, relationship_label
from the_last_ai.observatory.history import AgentLifeHistory
from the_last_ai.observatory.summary import generate_life_summary
from the_last_ai.types import JSONDict


def _bar(value: float, width: int = 10) -> str:
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _ascii_series(series: list[tuple[int, float]], *, title: str, lo: float = -1.0, hi: float = 1.0) -> str:
    if not series:
        return f"{title}\n(no samples)\n"
    height = 7
    width = min(60, max(20, len(series)))
    # resample
    step = max(1, len(series) // width)
    pts = series[::step][:width]
    rows = [[" " for _ in pts] for _ in range(height)]
    for x, (_, v) in enumerate(pts):
        norm = (v - lo) / (hi - lo) if hi > lo else 0.5
        y = int(round((1.0 - max(0.0, min(1.0, norm))) * (height - 1)))
        rows[y][x] = "•"
    lines = [title, ""]
    for i, row in enumerate(rows):
        label = f"{hi:4.1f}" if i == 0 else (f"{lo:4.1f}" if i == height - 1 else "    ")
        lines.append(f"{label} │{''.join(row)}")
    t0, t1 = pts[0][0], pts[-1][0]
    lines.append(f"     └{'─' * len(pts)}")
    lines.append(f"      t={t0}{' ' * max(1, len(pts) - 10)}t={t1}")
    return "\n".join(lines)


def build_export_payload(
    history: AgentLifeHistory,
    *,
    mind: JSONDict | None = None,
    active_ids: set[str] | None = None,
    meta: JSONDict | None = None,
) -> JSONDict:
    active_ids = active_ids or set()
    mind = mind or {}
    overview = history.overview(
        active_ids=active_ids,
        final_tick=history.events[-1].tick if history.events else None,
    )
    # Memory cards
    memories = []
    reps = mind.get("entity_representations") or {}
    strengths = mind.get("entity_strengths") or {}
    for eid, rep in reps.items():
        memories.append(
            {
                "entity_id": eid,
                "familiarity": rep.get("familiarity"),
                "memory_strength": strengths.get(eid, rep.get("strength")),
                "last_observed": rep.get("last_seen"),
                "expected_location": rep.get("expected_location"),
                "interaction_value": rep.get("interaction_value"),
                "uncertainty": rep.get("uncertainty"),
                "status": "CURRENTLY PRESENT" if eid in active_ids else "HISTORICAL MEMORY",
            }
        )
    # Relationship cards
    relationships = []
    for oid, traj in sorted(
        history.relationship_trajectories.items(),
        key=lambda kv: -(kv[1].strength_history[-1][1] if kv[1].strength_history else 0),
    ):
        strength = traj.strength_history[-1][1] if traj.strength_history else 0.0
        trust = traj.trust_history[-1][1] if traj.trust_history else 0.5
        reliability = (
            traj.reliability_history[-1][1] if traj.reliability_history else 0.5
        )
        relationships.append(
            {
                **traj.to_dict(),
                "current_strength": strength,
                "current_trust": trust,
                "current_information_reliability": reliability,
                "status": (
                    "HISTORICAL — AGENT NO LONGER PRESENT"
                    if traj.historical or oid not in active_ids
                    else "ACTIVE"
                ),
                "label": relationship_label(
                    interactions=traj.interaction_count, strength=strength, trust=trust
                ),
            }
        )

    affect_latest = history.affect_series[-1][1] if history.affect_series else None
    affect_label_info = affect_label(affect_latest) if affect_latest else ("n/a", [])

    summary = generate_life_summary(
        history,
        active_ids=active_ids,
        memories=mind.get("entity_memory"),
    )

    from the_last_ai.loss.narrative import build_loss_observatory_section

    loss_section = build_loss_observatory_section(
        mind.get("computational_loss"),
        agent_id=history.agent_id,
        affect_series=[[t, s.to_dict()] for t, s in history.affect_series],
    )

    return {
        "schema": "the_last_ai.agent_life.v1",
        "meta": meta or {},
        "overview": overview,
        "summary": summary,
        "affect_latest": {
            "values": affect_latest.to_dict() if affect_latest else {},
            "label": affect_label_info[0],
            "reasons": affect_label_info[1],
        },
        "affect_series": [[t, s.to_dict()] for t, s in history.affect_series],
        "memories": memories,
        "relationships": relationships,
        "messages": history.messages,
        "decisions": history.decisions,
        "events": [e.to_dict() for e in history.events],
        "history": history.to_dict(),
        "mind_snapshot": mind,
        "loss": loss_section,
        "non_claims": [
            "Does not establish subjective emotion or consciousness.",
            "Relationship labels are derived from measurable variables, not literal friendship.",
            "Affective labels are descriptive translations of numerical dimensions.",
            "Computational loss / social_loss is not grief or sadness — it measures unresolved expectations about absent entities.",
        ],
    }


def render_biography_text(payload: JSONDict) -> str:
    """Meticulously readable multi-section biography (TXT)."""
    ov = payload.get("overview") or {}
    aid = ov.get("agent_id", "agent")
    lines: list[str] = []
    lines.append("═" * 72)
    lines.append(f"  INDIVIDUAL AGENT LIFE OBSERVATORY — {aid}")
    lines.append("═" * 72)
    lines.append("")
    lines.append("TECHNICAL DATA IS THE SOURCE OF TRUTH. This document is a presentation layer.")
    lines.append("")

    # OVERVIEW
    lines.append("┌─ OVERVIEW " + "─" * 60)
    lines.append(f"│ Agent:        {aid}")
    lines.append(f"│ Birth tick:   {ov.get('birth_tick')}")
    lines.append(f"│ Age (ticks):  {ov.get('age_ticks')}")
    lines.append(f"│ Experiences:  {ov.get('experience_count')}")
    lines.append(f"│ Messages:     {ov.get('message_count')}")
    lines.append(f"│ Decisions:    {ov.get('decision_count')}")
    lines.append(
        f"│ Relationships: {ov.get('relationships_active')} active / "
        f"{ov.get('relationships_historical')} historical"
    )
    affect = payload.get("affect_latest") or {}
    vals = affect.get("values") or {}
    lines.append(f"│ Affect label: {affect.get('label')}")
    for k in (
        "valence",
        "activation",
        "social_drive",
        "uncertainty",
        "security",
        "prediction_error",
        "social_loss",
    ):
        if k in vals:
            lines.append(f"│   {k:18} {vals[k]:+.3f}" if k == "valence" else f"│   {k:18} {vals[k]:.3f}")
    lines.append("└" + "─" * 71)
    lines.append("")

    # SUMMARY
    lines.append("┌─ LIFE SUMMARY " + "─" * 55)
    for para in str(payload.get("summary") or "").splitlines():
        lines.append(f"│ {para}")
    lines.append("└" + "─" * 71)
    lines.append("")

    # LIFE TIMELINE
    lines.append("┌─ LIFE TIMELINE " + "─" * 54)
    for event in payload.get("events") or []:
        lines.append(
            f"│ TICK {event.get('tick'):<5}  [{event.get('event_type')}]  {event.get('summary')}"
        )
    lines.append("└" + "─" * 71)
    lines.append("")

    # MEMORY
    lines.append("┌─ MEMORY " + "─" * 61)
    for mem in payload.get("memories") or []:
        lines.append(f"│ ENTITY {mem.get('entity_id')}")
        lines.append(f"│   Status:            {mem.get('status')}")
        lines.append(f"│   Familiarity:       {mem.get('familiarity')}")
        lines.append(f"│   Memory strength:   {mem.get('memory_strength')}")
        lines.append(f"│   Last observed:     tick {mem.get('last_observed')}")
        lines.append(f"│   Expected location: {mem.get('expected_location')}")
        lines.append(f"│   Interaction value: {mem.get('interaction_value')}")
        lines.append(f"│   Uncertainty:       {mem.get('uncertainty')}")
        lines.append("│")
    if not payload.get("memories"):
        lines.append("│ (no entity memories in snapshot)")
    lines.append("└" + "─" * 71)
    lines.append("")

    # RELATIONSHIPS / WHO WERE THEIR FRIENDS
    lines.append("┌─ RELATIONSHIPS / SOCIAL HISTORY " + "─" * 37)
    lines.append("│ Strongest historical/active relationships (by strength):")
    lines.append("│")
    for i, rel in enumerate((payload.get("relationships") or [])[:10], start=1):
        lines.append(
            f"│ {i}. {rel.get('other_id')} — {rel.get('label')} [{rel.get('status')}]"
        )
        lines.append(
            f"│     strength {rel.get('current_strength'):.2f}  "
            f"trust {rel.get('current_trust'):.2f}  "
            f"interactions {rel.get('interaction_count')}"
        )
        lines.append(
            f"│     messages ok/fail {rel.get('successful_messages')}/{rel.get('failed_messages')}  "
            f"communicate_count {rel.get('communication_count')}"
        )
        lines.append(
            f"│     first tick {rel.get('first_interaction_tick')}  "
            f"last tick {rel.get('last_interaction_tick')}"
        )
        lines.append("│")
    lines.append("└" + "─" * 71)
    lines.append("")

    # TRUST
    lines.append("┌─ TRUST / INFORMATION RELIABILITY " + "─" * 36)
    for rel in payload.get("relationships") or []:
        r = float(rel.get("current_information_reliability") or 0.5)
        lines.append(
            f"│ {rel.get('other_id'):12} {_bar(r)} {r:.2f}  "
            f"(success {rel.get('successful_messages')} / fail {rel.get('failed_messages')})"
        )
        for ve in (rel.get("verification_events") or [])[-5:]:
            lines.append(
                f"│   tick {ve.get('tick')}: "
                f"{'SUCCESS' if ve.get('verified') else 'FAILURE'} "
                f"({ve.get('outcome')})"
            )
    lines.append("└" + "─" * 71)
    lines.append("")

    # COMMUNICATION
    lines.append("┌─ COMMUNICATION HISTORY " + "─" * 46)
    for msg in payload.get("messages") or []:
        tokens = " ".join(msg.get("tokens") or [])
        lines.append(
            f"│ TICK {msg.get('tick')}  {msg.get('sender')} → {msg.get('receiver')}  \"{tokens}\""
        )
        lines.append(
            f"│   reliability={msg.get('sender_reliability')}  "
            f"response={msg.get('receiver_response')}  "
            f"verified={msg.get('verified')}  outcome={msg.get('outcome')}"
        )
    if not payload.get("messages"):
        lines.append("│ (no messages recorded)")
    lines.append("└" + "─" * 71)
    lines.append("")

    # AFFECT SERIES
    lines.append("┌─ AFFECT TRAJECTORY " + "─" * 50)
    series = payload.get("affect_series") or []
    valence = [(t, s.get("valence", 0.0)) for t, s in series]
    social = [(t, s.get("social_drive", 0.0)) for t, s in series]
    pe = [(t, s.get("prediction_error", 0.0)) for t, s in series]
    lines.append(_ascii_series(valence, title="VALENCE", lo=-1, hi=1))
    lines.append("")
    lines.append(_ascii_series(social, title="SOCIAL DRIVE", lo=0, hi=1))
    lines.append("")
    lines.append(_ascii_series(pe, title="PREDICTION ERROR", lo=0, hi=1))
    lines.append("└" + "─" * 71)
    lines.append("")

    # DECISIONS
    lines.append("┌─ DECISIONS (notable) " + "─" * 48)
    for dec in (payload.get("decisions") or [])[-40:]:
        lines.append(f"│ TICK {dec.get('tick')}  selected: {dec.get('selected_action')}")
        info = dec.get("available_information") or {}
        for k, v in list(info.items())[:8]:
            lines.append(f"│   {k}: {v}")
        lines.append("│")
    lines.append("└" + "─" * 71)
    lines.append("")

    # NON-CLAIMS
    lines.append("┌─ SCIENTIFIC CAUTION " + "─" * 49)
    for c in payload.get("non_claims") or []:
        lines.append(f"│ • {c}")
    lines.append("└" + "─" * 71)
    lines.append("")
    return "\n".join(lines)


def render_biography_html(payload: JSONDict) -> str:
    """Self-contained HTML tabbed observatory — meticulously readable, no fake dialogue."""
    aid = html.escape(str((payload.get("overview") or {}).get("agent_id", "agent")))
    payload_json = json.dumps(payload, ensure_ascii=False)

    # Keep CSS/JS brace-heavy; inject JSON via replace to avoid f-string brace clashes.
    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Life Observatory — __AID__</title>
<style>
  :root {
    --bg: #f7f4ef; --ink: #1c1917; --muted: #57534e; --line: #d6d3d1;
    --panel: #ffffff; --accent: #0f766e; --hist: #a8a29e; --warn: #9a3412;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "IBM Plex Sans", "Source Sans 3", system-ui, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.45; }
  header { padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--line); background: var(--panel); }
  header h1 { margin: 0; font-size: 1.35rem; letter-spacing: 0.02em; }
  header p { margin: 0.35rem 0 0; color: var(--muted); font-size: 0.92rem; }
  .tabs { display: flex; flex-wrap: wrap; gap: 0.35rem; padding: 0.75rem 1.5rem;
    border-bottom: 1px solid var(--line); background: #efebe6; }
  .tabs button { border: 1px solid var(--line); background: var(--panel); color: var(--ink);
    padding: 0.4rem 0.75rem; cursor: pointer; font: inherit; border-radius: 0; }
  .tabs button.active { background: var(--accent); color: white; border-color: var(--accent); }
  main { padding: 1.25rem 1.5rem 3rem; max-width: 1100px; }
  .card { background: var(--panel); border: 1px solid var(--line); padding: 1rem 1.1rem; margin: 0 0 1rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
  .metric { font-size: 0.85rem; color: var(--muted); }
  .metric strong { display: block; color: var(--ink); font-size: 1.15rem; font-weight: 600; }
  .timeline { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.84rem; }
  .timeline .ev { padding: 0.35rem 0; border-bottom: 1px solid var(--line); cursor: pointer; }
  .timeline .ev:hover { background: #f0fdfa; }
  .status-hist { color: var(--warn); font-weight: 600; }
  pre { white-space: pre-wrap; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.82rem; }
  .detail { position: sticky; top: 0.5rem; background: #ecfeff; border: 1px solid #99f6e4;
    padding: 1rem; margin-top: 1rem; display: none; }
  .detail.open { display: block; }
  h2 { font-size: 1.05rem; margin: 0 0 0.75rem; }
  .bar { font-family: ui-monospace, monospace; letter-spacing: 0.05em; }
  .loss-hero .lede { font-size: 1.05rem; margin: 0.25rem 0 0.75rem; }
  .disclaimer { color: var(--muted); font-size: 0.9rem; border-left: 3px solid var(--accent); padding-left: 0.75rem; }
  .loss-entity .story p { max-width: 42rem; }
  .loss-entity h3 { font-size: 0.95rem; margin: 1rem 0 0.4rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .analogy { background: #fff7ed; border: 1px solid #fed7aa; padding: 0.75rem 0.9rem; }
  details { margin-top: 0.75rem; }
  details summary { cursor: pointer; color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Individual Agent Life Observatory — __AID__</h1>
  <p>Computational biography explorer. Technical JSON is authoritative. Affective labels are descriptive, not claims of feeling.</p>
</header>
<nav class="tabs" id="tabs"></nav>
<main id="main"></main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById('payload').textContent);
const tabs = [
  ['overview','OVERVIEW'],['life','LIFE'],['loss','LOSS & DISAPPEARANCE'],['memory','MEMORY'],['relationships','RELATIONSHIPS'],
  ['communication','COMMUNICATION'],['affect','AFFECT'],['predictions','PREDICTIONS'],['decisions','DECISIONS']
];
const nav = document.getElementById('tabs');
const main = document.getElementById('main');
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function bar(v){v=Math.max(0,Math.min(1,Number(v)||0)); const n=Math.round(v*10); return '█'.repeat(n)+'░'.repeat(10-n);}
function spark(points, markTick){
  if(!points||!points.length) return '<p class="muted">No trajectory samples yet.</p>';
  const w=520,h=90,pad=8;
  const xs=points.map(p=>Number(p.tick)), ys=points.map(p=>Number(p.value)||0);
  const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=0, ymax=Math.max(1, ...ys);
  const sx=t=> pad + (xmax===xmin?0:((t-xmin)/(xmax-xmin))*(w-2*pad));
  const sy=v=> h-pad - ((v-ymin)/(ymax-ymin||1))*(h-2*pad);
  let d='';
  points.forEach((p,i)=>{const x=sx(p.tick),y=sy(p.value); d+=(i?'L':'M')+x+','+y+' ';});
  let mark='';
  if(markTick!=null){const mx=sx(markTick); mark=`<line x1="${mx}" y1="${pad}" x2="${mx}" y2="${h-pad}" stroke="#9a3412" stroke-dasharray="4 3"/>`;}
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" style="background:#fafaf9;border:1px solid var(--line)">
    <path d="${d}" fill="none" stroke="#0f766e" stroke-width="2"/>${mark}</svg>`;
}
function showDetail(obj){
  let el=document.getElementById('detail');
  if(!el){el=document.createElement('div'); el.id='detail'; el.className='detail open'; main.prepend(el);}
  el.className='detail open';
  el.innerHTML = '<h2>Event inspector</h2><pre>'+esc(JSON.stringify(obj,null,2))+'</pre>';
}
function sectionOverview(){
  const o=data.overview||{}, a=data.affect_latest||{}, v=a.values||{};
  return `<div class="card"><h2>Status</h2>
    <div class="grid">
      <div class="metric">Agent<strong>${esc(o.agent_id)}</strong></div>
      <div class="metric">Age (ticks)<strong>${esc(o.age_ticks)}</strong></div>
      <div class="metric">Experiences<strong>${esc(o.experience_count)}</strong></div>
      <div class="metric">Messages<strong>${esc(o.message_count)}</strong></div>
      <div class="metric">Active relationships<strong>${esc(o.relationships_active)}</strong></div>
      <div class="metric">Historical relationships<strong>${esc(o.relationships_historical)}</strong></div>
      <div class="metric">Affect label<strong>${esc(a.label)}</strong></div>
      <div class="metric">Valence<strong>${Number(v.valence||0).toFixed(3)}</strong></div>
      <div class="metric">Social drive<strong>${Number(v.social_drive||0).toFixed(3)}</strong></div>
      <div class="metric">Uncertainty<strong>${Number(v.uncertainty||0).toFixed(3)}</strong></div>
      <div class="metric">Social loss<strong>${Number(v.social_loss||0).toFixed(3)}</strong></div>
      <div class="metric">Prediction error<strong>${Number(v.prediction_error||0).toFixed(3)}</strong></div>
    </div></div>
    <div class="card"><h2>Computational life summary</h2><pre>${esc(data.summary)}</pre></div>
    <div class="card"><h2>Scientific caution</h2><ul>${(data.non_claims||[]).map(c=>`<li>${esc(c)}</li>`).join('')}</ul></div>`;
}
function sectionLife(){
  const rows=(data.events||[]).map((e,i)=>`<div class="ev" data-i="${i}"><strong>TICK ${esc(e.tick)}</strong> [${esc(e.event_type)}] ${esc(e.summary)}</div>`).join('');
  return `<div class="card"><h2>Life timeline</h2><div class="timeline" id="life-tl">${rows}</div></div>`;
}
function sectionLoss(){
  const L=data.loss||{};
  const ents=L.entities||[];
  const counts=L.counts||{};
  let body=`<div class="card loss-hero">
    <h2>Loss &amp; disappearance</h2>
    <p class="lede">${esc(L.focus_prompt||'What changed after someone this agent knew disappeared?')}</p>
    <p class="disclaimer">${esc(L.disclaimer||'')}</p>
    <div class="grid">
      <div class="metric">Aggregate social_loss<strong>${Number(L.aggregate_social_loss||0).toFixed(3)}</strong></div>
      <div class="metric">Historical entities<strong>${esc(counts.historical||0)}</strong></div>
      <div class="metric">Forgotten<strong>${esc(counts.forgotten||0)}</strong></div>
      <div class="metric">Still active links<strong>${esc(counts.active||0)}</strong></div>
    </div>
  </div>`;
  if(!ents.length){
    body+=`<div class="card"><p>No disappearance-linked computational-loss records for this agent yet.</p></div>`;
    return body;
  }
  ents.forEach((n,idx)=>{
    const d=n.display||{};
    const mark=d.disappeared_tick;
    const tech=n.level3_technical||{};
    body+=`<div class="card loss-entity">
      <h2>${esc(d.entity||n.entity_id)} — what happened</h2>
      <div class="story">
        <h3>In plain language</h3>
        <p>${esc(n.level1_human_summary)}</p>
        <h3>Why (computational chain)</h3>
        <p>${esc(n.level2_why)}</p>
        ${n.analogy_note?`<p class="analogy">${esc(n.analogy_note)}</p>`:''}
      </div>
      <div class="grid facts">
        <div class="metric">Relationship before<strong>${esc(d.relationship_before)}</strong></div>
        <div class="metric">Interactions<strong>${esc(d.interaction_count)}</strong></div>
        <div class="metric">Trust<strong>${esc(d.trust)}</strong></div>
        <div class="metric">Memory before<strong>${esc(d.memory_strength_before)}</strong></div>
        <div class="metric">Disappeared<strong>Tick ${esc(d.disappeared_tick)}</strong></div>
        <div class="metric">Prediction disruption<strong>${esc(d.prediction_disruption)}</strong></div>
        <div class="metric">Social loss<strong>${esc(d.social_loss)}</strong></div>
        <div class="metric">Search attempts<strong>${esc(d.search_attempts)}</strong></div>
        <div class="metric">Communication attempts<strong>${esc(d.communication_attempts)}</strong></div>
        <div class="metric">Memory retrievals<strong>${esc(d.memory_retrievals)}</strong></div>
        <div class="metric">Behaviour notes<strong>${esc((n.behaviour_notes||[]).join(' · ')||'measured in experiment windows')}</strong></div>
        <div class="metric">Adaptation<strong>${d.adaptation!=null?('Tick '+d.adaptation):'not yet'}</strong></div>
        <div class="metric">Current memory<strong>${esc(d.current_memory)}</strong></div>
        <div class="metric">Status<strong class="status-hist">${esc(d.status)}</strong></div>
      </div>
      <h3>Social-loss over time</h3>
      ${spark(tech.loss_trajectory||[], mark)}
      <h3>Prediction disruption over time</h3>
      ${spark((tech.prediction_error_trajectory||[]).map(p=>({tick:p.tick,value:p.prediction_disruption})), mark)}
      <h3>Memory strength over time</h3>
      ${spark((tech.memory_trajectory||[]).map(p=>({tick:p.tick,value:p.memory_strength})), mark)}
      <details><summary>Technical data (Level 3)</summary><pre>${esc(JSON.stringify(tech,null,2))}</pre></details>
      <details><summary>Causal event chain</summary>
        <div class="timeline">${(tech.causal_events||[]).map((c,i)=>
          `<div class="ev" data-loss="${idx}:${i}">TICK ${esc(c.tick)} — ${esc(c.cause)}</div>`).join('')||'<p>None recorded.</p>'}
        </div>
      </details>
    </div>`;
  });
  return body;
}
function sectionMemory(){
  return `<div class="card"><h2>Entity memories</h2>${(data.memories||[]).map(m=>`
    <div class="card" style="margin:0.5rem 0">
      <strong>${esc(m.entity_id)}</strong>
      <div class="${m.status&&m.status.includes('HISTORICAL')?'status-hist':''}">${esc(m.status)}</div>
      <div>Familiarity: ${esc(m.familiarity)} · Strength: ${esc(m.memory_strength)} · Uncertainty: ${esc(m.uncertainty)}</div>
      <div>Last observed: tick ${esc(m.last_observed)} · Expected: ${esc(JSON.stringify(m.expected_location))}</div>
      <div>Interaction value: ${esc(m.interaction_value)}</div>
    </div>`).join('')||'<p>No entity memories in snapshot.</p>'}</div>`;
}
function sectionRelationships(){
  return `<div class="card"><h2>Social history</h2>${(data.relationships||[]).map(r=>`
    <div class="card" style="margin:0.5rem 0">
      <strong>${esc(r.other_id)}</strong> — ${esc(r.label)}
      <div class="${String(r.status).includes('HISTORICAL')?'status-hist':''}">${esc(r.status)}</div>
      <div>strength ${Number(r.current_strength||0).toFixed(2)} · trust ${Number(r.current_trust||0).toFixed(2)} · interactions ${esc(r.interaction_count)}</div>
      <div>first tick ${esc(r.first_interaction_tick)} · last tick ${esc(r.last_interaction_tick)}</div>
      <div>communicate ${esc(r.communication_count)} · info reliability ${Number(r.current_information_reliability||0.5).toFixed(2)}</div>
      <div class="bar">${bar(r.current_information_reliability||0.5)} ${Number(r.current_information_reliability||0.5).toFixed(2)}</div>
    </div>`).join('')||'<p>No relationships recorded.</p>'}</div>`;
}
function sectionCommunication(){
  return `<div class="card"><h2>Exact recorded messages</h2>
    <div class="timeline">${(data.messages||[]).map((m,i)=>`
      <div class="ev" data-m="${i}">TICK ${esc(m.tick)} · ${esc(m.sender)} → ${esc(m.receiver)} · "${esc((m.tokens||[]).join(' '))}"</div>`).join('')||'<p>No messages.</p>'}
    </div></div>`;
}
function sectionAffect(){
  const series=data.affect_series||[];
  const last=series.slice(-30);
  const rows=last.map(([t,s])=>`TICK ${t}  valence=${Number(s.valence).toFixed(3)}  social=${Number(s.social_drive).toFixed(3)}  pe=${Number(s.prediction_error).toFixed(3)}  loss=${Number(s.social_loss).toFixed(3)}`).join('\\n');
  const changes=(data.events||[]).filter(e=>e.event_type==='affective_change');
  return `<div class="card"><h2>Affective state</h2>
    <p>Label: <strong>${esc((data.affect_latest||{}).label)}</strong></p>
    <p>Reasons: ${esc(((data.affect_latest||{}).reasons||[]).join('; '))}</p>
    <pre>${esc(JSON.stringify((data.affect_latest||{}).values||{},null,2))}</pre>
    <h2>Recent samples</h2><pre>${esc(rows)}</pre>
    <h2>Recorded affective changes (with causes)</h2>
    <div class="timeline">${changes.map((e,i)=>`<div class="ev" data-a="${i}">TICK ${esc(e.tick)} — ${esc(e.summary)}</div>`).join('')}</div>
  </div>`;
}
function sectionPredictions(){
  const pe=(data.events||[]).filter(e=>e.event_type==='prediction_error'||e.event_type==='entity_absent_expected');
  return `<div class="card"><h2>Predictions & errors</h2>
    <div class="timeline">${pe.map((e,i)=>`<div class="ev" data-p="${i}">TICK ${esc(e.tick)} [${esc(e.event_type)}] ${esc(e.summary)}</div>`).join('')||'<p>No prediction events.</p>'}</div></div>`;
}
function sectionDecisions(){
  return `<div class="card"><h2>Notable decisions</h2>
    <div class="timeline">${(data.decisions||[]).map((d,i)=>`<div class="ev" data-d="${i}">TICK ${esc(d.tick)} → ${esc(d.selected_action)}</div>`).join('')||'<p>No decisions recorded.</p>'}</div></div>`;
}
const renderers={overview:sectionOverview,life:sectionLife,loss:sectionLoss,memory:sectionMemory,relationships:sectionRelationships,communication:sectionCommunication,affect:sectionAffect,predictions:sectionPredictions,decisions:sectionDecisions};
function activate(id){
  [...nav.querySelectorAll('button')].forEach(b=>b.classList.toggle('active',b.dataset.id===id));
  main.innerHTML = renderers[id]();
  main.querySelectorAll('.ev').forEach(el=>{
    el.addEventListener('click',()=>{
      if(el.dataset.i!=null) showDetail(data.events[el.dataset.i]);
      if(el.dataset.m!=null) showDetail(data.messages[el.dataset.m]);
      if(el.dataset.d!=null) showDetail(data.decisions[el.dataset.d]);
      if(el.dataset.a!=null){
        const changes=(data.events||[]).filter(e=>e.event_type==='affective_change');
        showDetail(changes[el.dataset.a]);
      }
      if(el.dataset.p!=null){
        const pe=(data.events||[]).filter(e=>e.event_type==='prediction_error'||e.event_type==='entity_absent_expected');
        showDetail(pe[el.dataset.p]);
      }
      if(el.dataset.loss!=null){
        const [ei,ci]=el.dataset.loss.split(':').map(Number);
        const tech=((data.loss||{}).entities||[])[ei];
        showDetail(((tech||{}).level3_technical||{}).causal_events[ci]);
      }
    });
  });
}
tabs.forEach(([id,label],i)=>{
  const b=document.createElement('button'); b.textContent=label; b.dataset.id=id;
  if(i===0) b.classList.add('active');
  b.onclick=()=>activate(id); nav.appendChild(b);
});
activate('overview');
</script>
</body>
</html>
"""
    # Escape </script> in JSON so the browser does not terminate the payload early.
    safe_json = payload_json.replace("</", "<\\/")
    return template.replace("__AID__", aid).replace("__PAYLOAD__", safe_json)


def export_agent_life(
    history: AgentLifeHistory,
    output_dir: str | Path,
    *,
    mind: JSONDict | None = None,
    active_ids: set[str] | None = None,
    meta: JSONDict | None = None,
) -> dict[str, str]:
    """Write agent_XXX_life.json / .txt / .html. JSON is authoritative."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_export_payload(history, mind=mind, active_ids=active_ids, meta=meta)
    stem = f"{history.agent_id}_life"
    json_path = output_dir / f"{stem}.json"
    txt_path = output_dir / f"{stem}.txt"
    html_path = output_dir / f"{stem}.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(render_biography_text(payload), encoding="utf-8")
    html_path.write_text(render_biography_html(payload), encoding="utf-8")
    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "html": str(html_path),
    }


def write_agent_biography(*args, **kwargs):
    return export_agent_life(*args, **kwargs)
