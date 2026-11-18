/* JobMatch landing page interactions. All demo data in /static/demo/*.json comes from the real
   index and real agent runs (see scripts/build_demo.py). Honours prefers-reduced-motion. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  const raf2 = (fn) => requestAnimationFrame(() => requestAnimationFrame(fn));
  const sleep = (ms) => new Promise((r) => setTimeout(r, reduced ? Math.min(ms, 30) : ms));
  const getJSON = (u) => fetch(u).then((r) => (r.ok ? r.json() : Promise.reject(new Error(u))));

  const landing = $("#landing");
  if (!landing) return;

  /* ------------------------------------------------ headline word reveal */
  function splitWords(root) {
    let n = 0;
    const walk = (node) => {
      [...node.childNodes].forEach((c) => {
        if (c.nodeType === 3) {
          const frag = document.createDocumentFragment();
          c.textContent.split(/(\s+)/).forEach((part) => {
            if (!part) return;
            if (/^\s+$/.test(part)) { frag.appendChild(document.createTextNode(" ")); return; }
            const w = document.createElement("span"); w.className = "word";
            const inner = document.createElement("span"); inner.textContent = part; inner.style.setProperty("--w", n++);
            w.appendChild(inner); frag.appendChild(w);
          });
          c.replaceWith(frag);
        } else if (c.nodeType === 1 && c.tagName !== "BR") walk(c);
      });
    };
    walk(root);
  }
  $$("[data-split]").forEach((el) => { el.setAttribute("aria-label", el.textContent.replace(/\s+/g, " ").trim()); splitWords(el); $$(".word", el).forEach((w) => w.setAttribute("aria-hidden", "true")); });
  const reveal = () => { landing.classList.add("loaded"); $$("[data-split]").forEach((el) => el.classList.add("in")); };
  if (reduced) reveal();
  else {
    const t = setTimeout(reveal, 1200);
    (document.fonts?.ready ?? Promise.resolve()).then(() => { clearTimeout(t); raf2(reveal); });
  }

  /* ------------------------------------------------ scroll reveals (once) */
  const revealEls = $$(".reveal");
  if (reduced || !("IntersectionObserver" in window)) revealEls.forEach((el) => el.classList.add("in"));
  else {
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
    revealEls.forEach((el) => io.observe(el));
  }

  /* ------------------------------------------------------ result rows */
  const chip = (s, cls = "", icon = "") => `<span class="chip caption-4 ${cls}">${icon}${esc(s)}</span>`;
  function matchRow(r, i) {
    const have = r.matched.map((s) => chip(s, "have", '<i class="ph ph-check" aria-hidden="true"></i>')).join("");
    const gap = r.missing.map((s) => chip("To add: " + s, "gap")).join("");
    return `<article class="res" style="--d:${i * 90}ms;--v:${r.overlap / 100}"><div style="min-width:0">
      <div class="title-3">${esc(r.title)}</div>
      <div class="caption-2 muted">${esc(r.company)}${r.location ? " · " + esc(r.location) : ""}${r.remote ? " · Remote" : ""}</div>
      <div class="chip-row" style="margin-top:var(--s-75)">${have}${gap}</div></div>
      <div class="score"><div class="num">${r.overlap}<small>%</small></div><div class="caption-5 muted">skill overlap</div><div class="bar" aria-hidden="true"><i></i></div></div></article>`;
  }
  function liveRow(j, i) {
    return `<article class="res" style="--d:${i * 90}ms"><div style="min-width:0">
      <div class="title-3">${esc(j.title)}</div>
      <div class="caption-2 muted">${esc(j.company || "Company not listed")}${j.location ? " · " + esc(j.location) : ""}${j.remote ? " · Remote" : ""}</div>
      <div class="chip-row" style="margin-top:var(--s-75)">${(j.skills || []).slice(0, 5).map((s) => chip(s)).join("")}</div></div>
      <div class="score"><div class="num">${i + 1}</div><div class="caption-5 muted">rank</div></div></article>`;
  }
  async function swapRows(container, html) {
    const old = $$(".res", container);
    if (old.length && !reduced) { old.forEach((o) => { o.style.transitionDelay = "0ms"; o.classList.remove("in"); o.style.opacity = "0"; o.style.transform = "translateY(-8px)"; }); await sleep(260); }
    container.innerHTML = html;
    raf2(() => $$(".res", container).forEach((r) => r.classList.add("in")));
  }

  /* --------------------------------------------------- live preview demo */
  let personas = [], activePersona = 0, interacted = false, autoTimer = null;
  const personaBar = $("#personas"), side = $("#demoSide"), main = $("#demoMain");
  function renderSide(p) {
    side.innerHTML = `<span class="caption-3 muted">Sample resume</span><span class="big">${esc(p.label)}</span>
      <span class="caption-2 muted">${esc(p.years)} years of experience</span><span class="caption-3 muted" style="margin-top:var(--s-100)">Skills</span>
      <div class="chip-row">${p.skills.map((s) => chip(s)).join("")}</div>
      <span class="caption-3 muted" style="margin-top:var(--s-100)">Target role</span><span class="body-5">${esc(p.role)}</span>`;
  }
  function selectPersona(i, user = false) {
    if (user) { interacted = true; clearInterval(autoTimer); }
    activePersona = i;
    $$(".persona", personaBar).forEach((b, k) => b.setAttribute("aria-selected", String(k === i)));
    const p = personas[i]; renderSide(p);
    swapRows(main, p.results.map(matchRow).join(""));
  }
  async function initDemo() {
    try { personas = await getJSON("/static/demo/personas.json"); } catch (e) { $("#demo").hidden = true; return; }
    personaBar.innerHTML = personas.map((p, i) => `<button class="persona body-5" type="button" role="tab" aria-selected="false" data-i="${i}">${esc(p.label)}</button>`).join("");
    personaBar.addEventListener("click", (e) => { const b = e.target.closest(".persona"); if (b) selectPersona(+b.dataset.i, true); });
    selectPersona(0);
    if (!reduced && "IntersectionObserver" in window) {
      new IntersectionObserver((es) => es.forEach((en) => {
        clearInterval(autoTimer);
        if (en.isIntersecting && !interacted) autoTimer = setInterval(() => selectPersona((activePersona + 1) % personas.length), 7000);
      }), { threshold: 0.4 }).observe($("#demo"));
    }
    initStoryData(); initBento();
  }
  $("#liveForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#liveQ").value.trim(); if (q.length < 2) return;
    interacted = true; clearInterval(autoTimer);
    $$(".persona", personaBar).forEach((b) => b.setAttribute("aria-selected", "false"));
    side.innerHTML = `<span class="caption-3 muted">Live search</span><span class="big">${esc(q)}</span><span class="caption-2 muted">Hybrid search over the whole index. No resume needed.</span>`;
    main.innerHTML = Array.from({ length: 3 }, () => '<div class="skeleton" aria-hidden="true"><i></i><i></i><i></i><i></i></div>').join("");
    try {
      const r = await fetch("/search?" + new URLSearchParams({ q, limit: 12, mode: "hybrid" }));
      const d = await r.json();
      if (!r.ok) throw new Error(d.error?.message || "Search failed");
      const rows = d.results.slice(0, 4);
      main.innerHTML = rows.length ? rows.map(liveRow).join("") : '<p class="body-6 muted">No postings matched. Try different words.</p>';
      raf2(() => $$(".res", main).forEach((x) => x.classList.add("in")));
    } catch (err) {
      main.innerHTML = `<div class="inline-error body-6" role="alert"><i class="ph ph-warning-circle" aria-hidden="true"></i><span>${esc(err.message)} The first search after a quiet period can take up to a minute.</span></div>`;
    }
  });

  /* ----------------------------------------------------------- marquees */
  async function initMarquee() {
    const skills = ["Python", "SQL", "React", "TypeScript", "Docker", "Kubernetes", "AWS", "Power BI", "Tableau", "Excel", "Figma", "Salesforce", "Machine learning", "Data analysis", "Project management", "Accounting", "Customer service", "Marketing", "Terraform", "Linux", "Java", "Go", "Rust", "FastAPI", "PostgreSQL", "Snowflake", "Spark", "Airflow"];
    let titles = [];
    try { titles = (await getJSON("/static/demo/ticker.json")).titles; } catch (e) { /* decorative */ }
    const fill = (el, items, mono) => { const html = items.map((t) => `<span class="tag body-5${mono ? " mono" : ""}">${esc(t)}</span>`).join(""); el.innerHTML = html + html; };
    if (titles.length) fill($("#trackTitles"), titles, false);
    fill($("#trackSkills"), skills, true);
  }

  /* ------------------------------------------------------------- bento */
  function flip(container, mutate) {
    const first = new Map([...container.children].map((k) => [k, k.getBoundingClientRect().top]));
    mutate();
    [...container.children].forEach((k) => {
      const dy = (first.get(k) ?? k.getBoundingClientRect().top) - k.getBoundingClientRect().top;
      if (dy && !reduced) { k.style.transition = "none"; k.style.transform = `translateY(${dy}px)`; requestAnimationFrame(() => { k.style.transition = ""; k.style.transform = ""; }); }
    });
  }
  function initBento() {
    const jb = personas[0].results[1] || personas[0].results[0];
    $("#gapViz").innerHTML = jb.matched.map((s) => chip(s, "have", '<i class="ph ph-check" aria-hidden="true"></i>')).join("") + jb.missing.map((s) => chip("To add: " + s, "gap")).join("");
    const rows = $("#tasteRows"), data = personas[2].results.slice(0, 3);
    const render = () => {
      rows.innerHTML = data.map((r, i) => `<div class="mini-row" data-i="${i}"><span class="grow body-5">${esc(r.title)}</span>
        <button class="btn btn-ghost button-xsm" type="button" data-like="1" aria-label="Like ${esc(r.title)}"><i class="ph ph-thumbs-up" aria-hidden="true"></i></button>
        <button class="btn btn-ghost button-xsm" type="button" data-like="0" aria-label="Dislike ${esc(r.title)}"><i class="ph ph-thumbs-down" aria-hidden="true"></i></button></div>`).join("")
        + '<button class="btn btn-ghost button-xsm" type="button" data-reset="1" style="align-self:flex-start;margin-top:var(--s-75)"><i class="ph ph-arrow-counter-clockwise" aria-hidden="true"></i>Reset</button>';
    };
    render();
    rows.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      if (b.dataset.reset) { render(); return; }
      const row = b.closest(".mini-row"); if (!row) return;
      if (b.dataset.like === "1") flip(rows, () => rows.prepend(row));
      else { row.style.transition = "opacity 500ms var(--ease), transform 500ms var(--ease)"; row.style.opacity = "0"; row.style.transform = "translateX(24px)"; setTimeout(() => flip(rows, () => row.remove()), reduced ? 0 : 450); }
    });
  }

  /* --------------------------------------------------------- sticky story */
  function initStoryData() {
    const p = personas[0];
    $("#sceneSkills").innerHTML = p.skills.map((s, k) => `<span class="chip have caption-4" style="--k:${k}">${esc(s)}</span>`).join("") + `<span class="chip caption-4" style="--k:${p.skills.length}">${esc(p.role)}</span>`;
    $("#sceneYears").textContent = `${p.years} years of experience found. Target role: ${p.role}.`;
    $("#sceneRes").innerHTML = p.results.slice(0, 3).map(matchRow).join("");
  }
  const steps = $$(".story-step"), scenes = $$(".scene");
  function setStep(i) { steps.forEach((s, k) => s.classList.toggle("on", k === i)); scenes.forEach((s, k) => s.classList.toggle("on", k === i)); }
  if ("IntersectionObserver" in window) {
    const sio = new IntersectionObserver((es) => es.forEach((en) => { if (en.isIntersecting) setStep($$(".triggers i").indexOf(en.target)); }), { rootMargin: "-50% 0px -50% 0px" });
    $$(".triggers i").forEach((t) => sio.observe(t));
  }

  /* -------------------------------------------------------------- tagline */
  const tag = $("#tagline");
  if (tag) {
    const hl = new Set((tag.dataset.hl || "").split(",").filter(Boolean));
    const words = tag.textContent.trim().split(/\s+/);
    tag.setAttribute("aria-label", words.join(" "));
    tag.innerHTML = words.map((w) => `<span class="w${hl.has(w.replace(/[.,]/g, "")) ? " hl" : ""}" aria-hidden="true">${esc(w)}</span>`).join(" ");
    const spans = $$(".w", tag);
    if (reduced || !("IntersectionObserver" in window)) spans.forEach((s) => s.classList.add("on"));
    else {
      const wio = new IntersectionObserver((es) => {
        es.filter((e) => e.isIntersecting).map((e) => e.target).sort((a, b) => spans.indexOf(a) - spans.indexOf(b))
          .forEach((el, i) => { el.style.transitionDelay = `${Math.min(i * 40, 400)}ms`; el.classList.add("on"); wio.unobserve(el); });
      }, { rootMargin: "0px 0px -42% 0px", threshold: 0 });
      spans.forEach((s) => wio.observe(s));
    }
  }

  /* ---------------------------------------------------------- agent theatre */
  let agent = [], runToken = 0, activeScenario = 0;
  const body = $("#termBody");
  const scroll = () => { body.scrollTop = body.scrollHeight; };
  async function type(el, text, speed, tok) {
    for (const ch of text) { if (tok !== runToken) return false; el.textContent += ch; if (!reduced) await sleep(speed); }
    return true;
  }
  async function stream(el, text, tok) {
    const words = text.split(/(\s+)/);
    for (const w of words) { if (tok !== runToken) return false; el.textContent += w; scroll(); if (!reduced && w.trim()) await sleep(28); }
    return true;
  }
  async function play(idx) {
    const tok = ++runToken; activeScenario = idx;
    $$(".scenario").forEach((b, k) => b.setAttribute("aria-selected", String(k === idx)));
    body.innerHTML = "";
    for (const turn of agent[idx].turns) {
      const u = document.createElement("div"); u.className = "t-user body-6"; body.appendChild(u);
      if (!(await type(u, turn.user, 22, tok))) return;
      await sleep(350);
      const bot = document.createElement("div"); bot.className = "t-bot"; body.appendChild(bot);
      const dots = document.createElement("span"); dots.className = "typing"; dots.innerHTML = "<i></i><i></i><i></i>"; bot.appendChild(dots); scroll();
      await sleep(650); if (tok !== runToken) return; dots.remove();
      for (const t of turn.tools) {
        const row = document.createElement("div"); row.className = "t-tool";
        row.innerHTML = '<i class="ph ph-circle-notch spin" aria-hidden="true"></i><span></span>';
        $("span", row).textContent = `${t.name}(${Object.keys(t.args).length ? JSON.stringify(t.args) : ""})`;
        bot.appendChild(row); raf2(() => row.classList.add("in")); scroll();
        await sleep(Math.min(t.ms, 1800) * 0.5 + 400); if (tok !== runToken) return;
        const ic = $("i", row); ic.className = `ph ph-${t.ok ? "check-circle" : "x-circle"}`; $("span", row).textContent += `  ${t.ms} ms`;
      }
      const reply = document.createElement("div"); reply.className = "t-reply body-6"; bot.appendChild(reply);
      const caret = document.createElement("i"); caret.className = "caret"; reply.after(caret);
      if (!(await stream(reply, turn.reply, tok))) return;
      caret.remove();
      const meta = document.createElement("div"); meta.className = "caption-5 muted"; meta.textContent = `${turn.llm_calls} model calls, ${(turn.total_ms / 1000).toFixed(1)} s in total`; bot.appendChild(meta); scroll();
      await sleep(900);
    }
  }
  async function initTheatre() {
    try { agent = await getJSON("/static/demo/agent.json"); } catch (e) { $("#agent").hidden = true; return; }
    $("#scenarios").innerHTML = agent.map((s, i) => `<button class="scenario" type="button" role="tab" aria-selected="false" data-i="${i}"><span class="title-3">${esc(s.label)}</span><span class="caption-2 muted">${esc(s.turns.map((t) => t.user).join(" → "))}</span></button>`).join("");
    $("#scenarios").addEventListener("click", (e) => { const b = e.target.closest(".scenario"); if (b) play(+b.dataset.i); });
    $("#replay").addEventListener("click", () => play(activeScenario));
    let started = false;
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((es, o) => es.forEach((en) => { if (en.isIntersecting && !started) { started = true; play(0); o.disconnect(); } }), { threshold: 0.35 }).observe($(".terminal"));
    } else play(0);
  }

  /* ------------------------------------------------------------ count up */
  const ease = (t) => 1 - Math.pow(1 - t, 4);
  function countUp(el) {
    const to = +el.dataset.count, dec = +el.dataset.dec || 0;
    if (reduced) { el.textContent = to.toFixed(dec); return; }
    const t0 = performance.now(), dur = 1600;
    const tick = (now) => { const p = Math.min(1, (now - t0) / dur); el.textContent = (to * ease(p)).toFixed(dec); if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  const stats = $$(".stat");
  if (reduced || !("IntersectionObserver" in window)) { stats.forEach((s) => { s.classList.add("in"); $$("[data-count]", s).forEach(countUp); }); }
  else {
    const cio = new IntersectionObserver((es) => es.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); $$("[data-count]", en.target).forEach(countUp); cio.unobserve(en.target); } }), { threshold: 0.4 });
    stats.forEach((s) => { $$("[data-count]", s).forEach((c) => { c.textContent = (0).toFixed(+c.dataset.dec || 0); }); cio.observe(s); });
  }

  /* ------------------------------------------- pointer effects (fine pointers) */
  if (finePointer && !reduced) {
    // spotlight: a blurred flat shape follows the pointer inside each card
    $$("[data-spot]").forEach((card) => {
      const spot = $(".spot", card); let queued = false, x = 0, y = 0;
      card.addEventListener("pointermove", (e) => {
        const r = card.getBoundingClientRect(); x = e.clientX - r.left; y = e.clientY - r.top;
        if (!queued) { queued = true; requestAnimationFrame(() => { spot.style.setProperty("--x", x + "px"); spot.style.setProperty("--y", y + "px"); queued = false; }); }
      });
    });
    // tilt on the agent window and the story scenes
    $$(".terminal, .scenes").forEach((el) => {
      el.classList.add("tilt");
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect(), px = (e.clientX - r.left) / r.width - 0.5, py = (e.clientY - r.top) / r.height - 0.5;
        el.classList.add("moving"); el.style.setProperty("--ry", (px * 4).toFixed(2) + "deg"); el.style.setProperty("--rx", (-py * 4).toFixed(2) + "deg");
      });
      el.addEventListener("pointerleave", () => { el.classList.remove("moving"); el.style.setProperty("--rx", "0deg"); el.style.setProperty("--ry", "0deg"); });
    });
    // magnetic primary buttons
    $$(".magnetic").forEach((b) => {
      b.addEventListener("pointermove", (e) => { const r = b.getBoundingClientRect(); b.style.transform = `translate(${((e.clientX - (r.left + r.width / 2)) * 0.18).toFixed(1)}px, ${((e.clientY - (r.top + r.height / 2)) * 0.28).toFixed(1)}px)`; });
      b.addEventListener("pointerleave", () => { b.style.transform = ""; });
    });
    // ambient orb that eases toward the pointer
    const orb = $("#cursorOrb");
    if (orb) {
      let tx = innerWidth / 2, ty = innerHeight / 3, cx = tx, cy = ty, running = false;
      const loop = () => { cx += (tx - cx) * 0.06; cy += (ty - cy) * 0.06; orb.style.transform = `translate3d(${cx}px, ${cy}px, 0)`; if (Math.abs(tx - cx) + Math.abs(ty - cy) > 0.5) requestAnimationFrame(loop); else running = false; };
      window.addEventListener("pointermove", (e) => { tx = e.clientX; ty = e.clientY; if (!running) { running = true; requestAnimationFrame(loop); } }, { passive: true });
    }
  }

  initMarquee(); initDemo(); initTheatre();
})();
