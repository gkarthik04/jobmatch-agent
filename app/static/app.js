/* JobMatch front end: landing interactions + the signed in app. No framework. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let token = null;
  try { token = localStorage.getItem("jm_token"); } catch (e) { /* storage blocked */ }
  const saveToken = (t) => { token = t; try { t ? localStorage.setItem("jm_token", t) : localStorage.removeItem("jm_token"); } catch (e) { /* ignore */ } };

  /* ------------------------------------------------------------------ api */
  async function api(path, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    if (token) headers.Authorization = "Bearer " + token;
    if (opts.json) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.json); }
    let r;
    try { r = await fetch(path, { ...opts, headers }); }
    catch (e) { throw new Error("Could not reach the server. Check your connection and try again."); }
    if (r.status === 204) return null;
    const data = await r.json().catch(() => ({}));
    if (r.status === 401 && token) { logout(); throw new Error("Your session expired. Log in again."); }
    if (!r.ok) { const err = new Error(data.error?.message || `Request failed (${r.status})`); err.code = data.error?.code; throw err; }
    return data;
  }
  const setError = (el, msg) => { el.innerHTML = msg ? `<i class="ph ph-warning-circle" aria-hidden="true"></i><span>${esc(msg)}</span>` : ""; };

  /* ------------------------------------------------------------ navigation */
  const LANDING_ITEMS = [
    { label: "How it works", href: "#how" },
    { label: "Results", href: "#results" },
    { label: "FAQ", href: "#faq" },
    { label: "Log in", auth: "login" },
  ];
  const APP_ITEMS = [
    { label: "Profile", tab: "profile" },
    { label: "Matches", tab: "matches" },
    { label: "Chat", tab: "chat" },
    { label: "Search", tab: "search" },
    { label: "Saved", tab: "saved" },
    { label: "Log out", action: "logout" },
  ];
  let currentTab = "profile";

  function itemAttrs(it) {
    if (it.href) return `href="${it.href}"`;
    if (it.tab) return `data-tab="${it.tab}" type="button"`;
    if (it.auth) return `data-auth="${it.auth}" type="button"`;
    return `data-action="${it.action}" type="button"`;
  }
  const tagFor = (it) => (it.href ? "a" : "button");

  function renderNav() {
    const items = token ? APP_ITEMS : LANDING_ITEMS;
    const mid = token ? items.slice(0, -1) : items.slice(0, -1);
    const end = items[items.length - 1];
    $("#navLinks").innerHTML = mid.map((it) => `<${tagFor(it)} class="nav-link body-5" ${itemAttrs(it)}${it.tab && it.tab === currentTab ? ' aria-current="page"' : ""}>${it.label}</${tagFor(it)}>`).join("");
    $("#navEnd").innerHTML = `<${tagFor(end)} class="nav-link body-5" ${itemAttrs(end)}>${end.label}</${tagFor(end)}>`;
    $("#mobileMenu").innerHTML = items.map((it, i) => `<div class="mm-item" style="--i:${i}"><${tagFor(it)} class="mm-link heading-5" ${itemAttrs(it)}${it.tab && it.tab === currentTab ? ' aria-current="page"' : ""}>${it.label}</${tagFor(it)}></div>`).join("");
  }

  const menu = $("#mobileMenu"), toggle = $("#menuToggle");
  function setMenu(open) {
    menu.dataset.open = String(open);
    menu.setAttribute("aria-hidden", String(!open));
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    document.body.style.overflow = open ? "hidden" : "";
    if (open) setTimeout(() => $(".mm-link", menu)?.focus(), 50);
  }
  toggle.addEventListener("click", () => setMenu(menu.dataset.open !== "true"));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && menu.dataset.open === "true") { setMenu(false); toggle.focus(); } });

  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-tab],[data-auth],[data-action],a[href^='#']");
    if (!t) return;
    if (t.dataset.tab) { setMenu(false); setTab(t.dataset.tab); }
    else if (t.dataset.auth) { setMenu(false); openAuth(t.dataset.auth); }
    else if (t.dataset.action === "logout") { setMenu(false); logout(); }
    else if (t.matches("a[href^='#']")) setMenu(false);
  });

  /* ---------------------------------------------------------------- views */
  const landing = $("#landing"), appRoot = $("#appRoot");
  function showView() {
    const signedIn = !!token;
    landing.hidden = signedIn;
    appRoot.hidden = !signedIn;
    renderNav();
    window.scrollTo(0, 0);
    if (signedIn) setTab(currentTab);
  }
  function logout() {
    saveToken(null);
    matchCache = null;
    currentTab = "profile";
    showView();
  }
  function setTab(name) {
    currentTab = name;
    $$(".tab").forEach((s) => { s.hidden = s.id !== "t-" + name; });
    renderNav();
    window.scrollTo(0, 0);
    if (name === "profile") loadProfile();
    if (name === "matches") renderMatchesIdle();
    if (name === "chat") loadChat();
    if (name === "saved") loadSaved();
  }

  /* ----------------------------------------------------------------- auth */
  const dlg = $("#authDialog");
  let authMode = "signup";
  function openAuth(mode) {
    authMode = mode;
    const signup = mode === "signup";
    $("#authTitle").textContent = signup ? "Create your account" : "Log in";
    $("#authSub").textContent = signup ? "Then upload your resume to see matches." : "Welcome back.";
    $("#authBtn").textContent = signup ? "Create account" : "Log in";
    $("#authSwitch").textContent = signup ? "I already have an account" : "Create an account";
    $("#password").autocomplete = signup ? "new-password" : "current-password";
    ["#emailErr", "#passErr", "#authErr"].forEach((s) => setError($(s), ""));
    $("#email").removeAttribute("aria-invalid"); $("#password").removeAttribute("aria-invalid");
    if (!dlg.open) dlg.showModal();
    $("#email").focus();
  }
  $("#authSwitch").addEventListener("click", () => openAuth(authMode === "signup" ? "login" : "signup"));
  $("#authClose").addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });

  $("#authForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = $("#email").value.trim(), password = $("#password").value;
    let ok = true;
    const emailOk = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    setError($("#emailErr"), emailOk ? "" : "Enter a valid email address, like name@example.com.");
    $("#email").setAttribute("aria-invalid", String(!emailOk));
    const passOk = password.length >= 8 && password.length <= 72;
    setError($("#passErr"), passOk ? "" : "Use between 8 and 72 characters.");
    $("#password").setAttribute("aria-invalid", String(!passOk));
    ok = emailOk && passOk;
    setError($("#authErr"), "");
    if (!ok) return;
    const btn = $("#authBtn"); btn.disabled = true; const label = btn.textContent; btn.textContent = "One moment";
    try {
      const d = await api(authMode === "signup" ? "/auth/signup" : "/auth/login", { method: "POST", json: { email, password } });
      saveToken(d.access_token);
      $("#password").value = "";
      dlg.close();
      currentTab = "profile";
      showView();
    } catch (err) { setError($("#authErr"), err.message); }
    btn.disabled = false; btn.textContent = label;
  });

  /* -------------------------------------------------------------- helpers */
  const chips = (arr, cls = "", icon = "") => (arr || []).map((s) => `<span class="chip caption-4 ${cls}">${icon}${esc(s)}</span>`).join("");
  const skeleton = (n = 3) => Array.from({ length: n }, () => '<div class="skeleton" aria-hidden="true"><i></i><i></i><i></i><i></i></div>').join("");
  const emptyState = (icon, title, text, btn = "") => `<div class="empty"><i class="ph ph-${icon}" aria-hidden="true"></i><h2 class="title-2">${esc(title)}</h2><p class="body-6">${esc(text)}</p>${btn}</div>`;
  const gotoBtn = (tab, label) => `<button class="btn btn-primary button-md" type="button" data-tab="${tab}">${esc(label)}</button>`;

  /* -------------------------------------------------------------- profile */
  const fileIn = $("#file"), dz = $("#dropzone"), upBtn = $("#upload");
  function pickFile(f) {
    setError($("#upErr"), "");
    if (!f) { upBtn.disabled = true; return; }
    if (!/\.pdf$/i.test(f.name) && f.type !== "application/pdf") { setError($("#upErr"), "That file is not a PDF. Choose a PDF resume."); upBtn.disabled = true; return; }
    if (f.size > 5 * 1024 * 1024) { setError($("#upErr"), "The file is over 5 MB. Export a smaller PDF and try again."); upBtn.disabled = true; return; }
    $("#dropTitle").textContent = f.name;
    $("#dropHint").textContent = `${(f.size / 1024).toFixed(0)} KB, ready to read`;
    upBtn.disabled = false;
  }
  fileIn.addEventListener("change", () => pickFile(fileIn.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.dataset.drag = "true"; }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.dataset.drag = "false"; }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) { fileIn.files = e.dataTransfer.files; pickFile(f); } });

  upBtn.addEventListener("click", async () => {
    const f = fileIn.files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    upBtn.disabled = true; $("#upStatus").textContent = "Reading your resume. This usually takes 5 to 30 seconds.";
    $("#profileView").innerHTML = skeleton(2); setError($("#upErr"), "");
    try { await api("/resume", { method: "POST", body: fd }); matchCache = null; $("#upStatus").textContent = ""; await loadProfile(true); }
    catch (err) { $("#upStatus").textContent = ""; setError($("#upErr"), err.message); $("#profileView").innerHTML = ""; upBtn.disabled = false; }
  });

  async function loadProfile(fresh = false) {
    const view = $("#profileView");
    if (!fresh) view.innerHTML = skeleton(1);
    try {
      const p = await api("/profile");
      view.innerHTML = `
        <div class="card" style="margin-top:var(--s-300)">
          <div style="display:flex;justify-content:space-between;gap:var(--s-200);flex-wrap:wrap;align-items:flex-start">
            <div><h2 class="heading-5">${esc(p.name)}</h2><p class="caption-2 muted">${esc(p.experience_years)} years of experience</p></div>
            <button class="btn btn-primary button-md" type="button" data-tab="matches"><i class="ph ph-arrow-right" aria-hidden="true"></i>Find my matches</button>
          </div>
          <div class="profile-grid">
            <div class="full"><span class="label caption-3">Skills</span><div class="chip-row">${chips(p.skills)}</div></div>
            <div><span class="label caption-3">Target roles</span><div class="chip-row">${chips(p.preferred_roles) || '<span class="caption-2 muted">None found</span>'}</div></div>
            <div><span class="label caption-3">Locations</span><div class="chip-row">${chips(p.preferred_locations) || '<span class="caption-2 muted">None stated in the resume</span>'}</div></div>
            <div><span class="label caption-3">Education</span>${p.education.map((e) => `<p class="body-6">${esc([e.degree, e.institution, e.year].filter(Boolean).join(", "))}</p>`).join("") || '<span class="caption-2 muted">None found</span>'}</div>
            <div><span class="label caption-3">Projects</span>${p.projects.map((x) => `<p class="body-6"><span class="body-5">${esc(x.name)}</span> <span class="muted">${esc(x.description)}</span></p>`).join("") || '<span class="caption-2 muted">None found</span>'}</div>
          </div>
        </div>`;
    } catch (e) {
      view.innerHTML = e.code === "no_profile"
        ? emptyState("file-text", "No profile yet", "Choose a PDF above and select Read my resume. Your profile appears here for review before anything is matched.")
        : "";
      if (e.code !== "no_profile") setError($("#upErr"), e.message);
    }
  }

  /* ------------------------------------------------------------- job cards */
  const saved = new Set();
  const fb = new Map();
  function jobCard(j, m, opts = {}) {
    const posted = j.posted_at ? new Date(j.posted_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) : "";
    const fit = m ? `<div class="fit"><div class="heading-5">${m.fit_score}<span class="caption-2 muted"> / 100</span></div><div class="bar" aria-hidden="true"><i data-fit="${m.fit_score}"></i></div></div>` : "";
    const body = m
      ? `<p class="body-6">${esc(m.why_it_matches)}</p>
         <div class="chip-row">${chips(m.matched_skills, "have", '<i class="ph ph-check" aria-hidden="true"></i>')}${(m.missing_skills || []).map((s) => `<span class="chip gap caption-4">To add: ${esc(s)}</span>`).join("")}</div>`
      : `<p class="body-6 muted">${esc(j.snippet)}</p><div class="chip-row">${chips(j.skills)}</div>`;
    const f = fb.get(j.id);
    const actions = [
      m ? `<button class="btn btn-ghost button-sm" type="button" data-fb="1" aria-pressed="${f === true}"><i class="ph ph-thumbs-up" aria-hidden="true"></i>Like</button>
           <button class="btn btn-ghost button-sm" type="button" data-fb="0" aria-pressed="${f === false}"><i class="ph ph-thumbs-down" aria-hidden="true"></i>Dislike</button>` : "",
      opts.removable
        ? `<button class="btn btn-ghost button-sm" type="button" data-unsave="1"><i class="ph ph-trash" aria-hidden="true"></i>Remove</button>`
        : `<button class="btn btn-ghost button-sm" type="button" data-save="1" aria-pressed="${saved.has(j.id)}"><i class="ph ph-bookmark-simple" aria-hidden="true"></i>${saved.has(j.id) ? "Saved" : "Save"}</button>`,
      !m && j.ranks ? `<span class="rank-note caption-2">${esc(Object.entries(j.ranks).map(([k, v]) => `${k} rank ${v}`).join(", "))}${j.rerank_score != null ? `, rerank ${j.rerank_score}` : ""}</span>` : "",
    ].join("");
    return `<article class="card job" data-id="${j.id}">
      <div class="job-top"><div><h3 class="title-2">${esc(j.title)}</h3>
        <div class="job-meta caption-2"><span>${esc(j.company || "Company not listed")}</span><span>${esc(j.location || "")}</span>${j.remote ? '<span class="chip caption-4">Remote</span>' : ""}${posted ? `<span>${posted}</span>` : ""}</div></div>${fit}</div>
      ${body}<div class="job-actions">${actions}</div></article>`;
  }
  function paintBars(root) {
    requestAnimationFrame(() => $$(".bar i[data-fit]", root).forEach((i) => { i.style.transform = `scaleX(${Math.min(100, +i.dataset.fit) / 100})`; }));
  }

  async function jobAction(e) {
    const btn = e.target.closest("button"); if (!btn) return;
    const card = btn.closest(".job"); if (!card) return;
    const id = +card.dataset.id;
    try {
      if (btn.dataset.fb != null) {
        const liked = btn.dataset.fb === "1";
        await api("/feedback", { method: "POST", json: { job_id: id, liked } });
        fb.set(id, liked);
        $$("[data-fb]", card).forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.fb === "1") === liked)));
        $("#matchNote").innerHTML = `<div class="inline-note body-6"><i class="ph ph-check-circle" aria-hidden="true" style="color:var(--accent)"></i><span>Feedback saved. Run the matches again to see the new order.</span><button class="btn btn-ghost button-sm" type="button" data-refresh="1">Refresh matches</button></div>`;
      }
      if (btn.dataset.save) {
        await api("/saved", { method: "POST", json: { job_id: id } });
        saved.add(id); btn.setAttribute("aria-pressed", "true"); btn.innerHTML = '<i class="ph ph-bookmark-simple" aria-hidden="true"></i>Saved';
      }
      if (btn.dataset.unsave) {
        await api("/saved/" + id, { method: "DELETE" });
        saved.delete(id); card.remove();
        if (!$("#savedList .job")) loadSaved();
      }
    } catch (err) {
      let note = card.querySelector(".inline-error");
      if (!note) { note = document.createElement("div"); note.className = "inline-error body-6"; note.setAttribute("role", "alert"); card.appendChild(note); }
      setError(note, err.message);
    }
  }
  ["#matchList", "#searchList", "#savedList"].forEach((s) => $(s).addEventListener("click", jobAction));
  $("#matchNote").addEventListener("click", (e) => { if (e.target.closest("[data-refresh]")) loadMatches(); });

  /* -------------------------------------------------------------- matches */
  let matchCache = null;
  function renderMatchesIdle() {
    if (matchCache) { $("#matchList").innerHTML = matchCache.map((m) => jobCard(m.job, m)).join(""); paintBars($("#matchList")); return; }
    $("#matchList").innerHTML = emptyState("magnifying-glass", "Your ranked shortlist appears here",
      "Select Find my matches. It searches the index, reorders the top results with a language model, and explains each one. This can take up to a minute on the free tier.");
  }
  $("#loadMatches").addEventListener("click", loadMatches);
  async function loadMatches() {
    const b = $("#loadMatches"); b.disabled = true;
    $("#matchNote").innerHTML = ""; $("#matchList").innerHTML = skeleton(3);
    try {
      const rows = await api("/matches?limit=10" + ($("#mRemote").checked ? "&remote=true" : ""));
      matchCache = rows;
      $("#matchList").innerHTML = rows.length ? rows.map((m) => jobCard(m.job, m)).join("")
        : emptyState("tray", "No matches found", "Try turning off Remote only, or upload a resume that lists more skills.");
      paintBars($("#matchList"));
    } catch (e) {
      $("#matchList").innerHTML = e.code === "no_profile"
        ? emptyState("file-text", "Upload your resume first", "Matches are built from your profile.", gotoBtn("profile", "Go to your profile"))
        : `<div class="inline-error body-6" role="alert"><i class="ph ph-warning-circle" aria-hidden="true"></i><span>${esc(e.message)}</span></div>`;
    }
    b.disabled = false;
  }

  /* --------------------------------------------------------------- search */
  $("#searchForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#q").value.trim(); if (q.length < 2) return;
    const p = new URLSearchParams({ q, mode: $("#mode").value, limit: 10 });
    if ($("#loc").value.trim()) p.set("location", $("#loc").value.trim());
    if ($("#sRemote").checked) p.set("remote", "true");
    $("#searchList").innerHTML = skeleton(3);
    try {
      const d = await api("/search?" + p);
      $("#searchList").innerHTML = d.results.length ? d.results.map((j) => jobCard(j, null)).join("")
        : emptyState("tray", "No results", "Try different words or remove a filter.");
    } catch (err) { $("#searchList").innerHTML = `<div class="inline-error body-6" role="alert"><i class="ph ph-warning-circle" aria-hidden="true"></i><span>${esc(err.message)}</span></div>`; }
  });

  /* ---------------------------------------------------------------- saved */
  async function loadSaved() {
    const list = $("#savedList"); list.innerHTML = skeleton(2);
    try {
      const rows = await api("/saved");
      saved.clear(); rows.forEach((j) => saved.add(j.id));
      list.innerHTML = rows.length ? rows.map((j) => jobCard(j, null, { removable: true })).join("")
        : emptyState("bookmark-simple", "Nothing saved yet", "Save jobs from your matches or search results, or ask the agent to save one.", gotoBtn("matches", "Open matches"));
    } catch (e) { list.innerHTML = `<div class="inline-error body-6" role="alert"><i class="ph ph-warning-circle" aria-hidden="true"></i><span>${esc(e.message)}</span></div>`; }
  }

  /* ----------------------------------------------------------------- chat */
  const chatBox = $("#chat");
  const STARTERS = ["Find remote data analyst jobs", "Find python backend developer jobs", "What jobs have I saved?"];
  function addMsg(role, text, extra) {
    const d = document.createElement("div");
    d.className = "msg body-6 " + (role === "user" ? "user" : "bot");
    d.textContent = text;
    if (extra) d.appendChild(extra);
    chatBox.appendChild(d); chatBox.scrollTop = chatBox.scrollHeight; return d;
  }
  function toolPanel(res) {
    const tools = res.steps.filter((s) => s.kind === "tool"), u = res.usage;
    const plan = res.steps.find((s) => s.kind === "llm" && s.text && tools.length);
    const det = document.createElement("details"); det.className = "tools";
    det.innerHTML = `<summary class="caption-2"><i class="ph ph-wrench" aria-hidden="true"></i>${tools.length ? "Tools used: " + esc(tools.map((t) => t.name).join(", ")) : "No tools used"}</summary>
      <div class="caption-2 muted" style="margin:var(--s-75) 0">${u.llm_calls} model call${u.llm_calls === 1 ? "" : "s"}, ${u.prompt_tokens + u.output_tokens} tokens, ${(u.latency_ms / 1000).toFixed(1)} s</div>
      ${plan ? `<div class="caption-2 muted" style="margin-bottom:var(--s-75)">Plan: ${esc(plan.text)}</div>` : ""}
      ${tools.map((t) => `<div class="tool-row caption-2 ${t.ok ? "" : "fail"}"><i class="ph ph-${t.ok ? "check-circle" : "x-circle"}" aria-hidden="true"></i><span>Step ${t.step}: <span class="mono">${esc(t.name)}(${esc(JSON.stringify(t.args))})</span>, ${Math.round(t.latency_ms)} ms${t.error ? ". " + esc(t.error) : ""}</span></div>`).join("")}`;
    return det;
  }
  function renderStarters(show) {
    $("#starters").innerHTML = show ? STARTERS.map((s) => `<button class="btn btn-ghost button-sm" type="button" data-starter="${esc(s)}">${esc(s)}</button>`).join("") : "";
  }
  async function loadChat() {
    try {
      const h = await api("/chat/history");
      chatBox.innerHTML = "";
      h.forEach((m) => addMsg(m.role, m.content));
      if (!h.length) addMsg("bot", "I can search the jobs index, open a job, compare it with your resume, and save it for you. What are you looking for?");
      renderStarters(!h.length);
    } catch (e) { setError($("#chatErr"), e.message); }
  }
  $("#starters").addEventListener("click", (e) => { const b = e.target.closest("[data-starter]"); if (b) { $("#chatIn").value = b.dataset.starter; $("#chatForm").requestSubmit(); } });
  $("#chatForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const t = $("#chatIn").value.trim(); if (!t) return;
    $("#chatIn").value = ""; setError($("#chatErr"), ""); renderStarters(false);
    addMsg("user", t);
    const wait = document.createElement("div"); wait.className = "msg bot"; wait.innerHTML = '<span class="typing" aria-label="The agent is working"><i></i><i></i><i></i></span>';
    chatBox.appendChild(wait); chatBox.scrollTop = chatBox.scrollHeight;
    $("#chatSend").disabled = true;
    try {
      const r = await api("/chat", { method: "POST", json: { message: t } });
      wait.remove(); addMsg("bot", r.reply, toolPanel(r));
    } catch (err) { wait.remove(); setError($("#chatErr"), err.message); }
    $("#chatSend").disabled = false; $("#chatIn").focus();
  });
  $("#chatReset").addEventListener("click", async () => {
    try { await api("/chat/history", { method: "DELETE" }); setError($("#chatErr"), ""); loadChat(); }
    catch (e) { setError($("#chatErr"), e.message); }
  });

  /* ----------------------------------------------------- landing behaviours */
  // scroll reveals: once per element, never on an unthrottled scroll listener
  const revealEls = $$(".reveal");
  if (reduced || !("IntersectionObserver" in window)) revealEls.forEach((el) => el.classList.add("in"));
  else {
    const io = new IntersectionObserver((entries) => entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } }), { rootMargin: "0px 0px -10% 0px", threshold: 0.08 });
    revealEls.forEach((el) => io.observe(el));
  }

  // tagline: each word moves from muted to full colour as it crosses a trigger line
  const tag = $("#tagline");
  if (tag) {
    const words = tag.textContent.trim().split(/\s+/);
    tag.setAttribute("aria-label", tag.textContent.trim());
    tag.innerHTML = words.map((w) => `<span class="w" aria-hidden="true">${esc(w)}</span>`).join(" ");
    const spans = $$(".w", tag);
    if (reduced || !("IntersectionObserver" in window)) spans.forEach((s) => s.classList.add("on"));
    else {
      const wio = new IntersectionObserver((entries) => {
        const hit = entries.filter((en) => en.isIntersecting).map((en) => en.target).sort((a, b) => spans.indexOf(a) - spans.indexOf(b));
        hit.forEach((el, i) => { el.style.transitionDelay = `${Math.min(i * 40, 400)}ms`; el.classList.add("on"); wio.unobserve(el); });
      }, { rootMargin: "0px 0px -45% 0px", threshold: 0 });
      spans.forEach((s) => wio.observe(s));
    }
  }

  // mark the current section in the nav
  if ("IntersectionObserver" in window) {
    const secs = ["how", "results", "faq"].map((id) => document.getElementById(id)).filter(Boolean);
    const nio = new IntersectionObserver((entries) => {
      if (token) return;
      entries.forEach((en) => { if (en.isIntersecting) $$("#navLinks a").forEach((a) => { if (a.getAttribute("href") === "#" + en.target.id) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); }); });
    }, { rootMargin: "-40% 0px -55% 0px" });
    secs.forEach((s) => nio.observe(s));
  }

  showView();
})();
