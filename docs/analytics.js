/*
 * Google Analytics 4 interaction tracking for the final project explorer on GitHub Pages.
 *
 * Hand-maintained -- NOT generated. build_docs.py adds
 * <script src="analytics.js" defer></script> to docs/index.html. The local Flask
 * explorer (run_explorer.py) never loads it, so nothing is tracked when students
 * run it on their own machines.
 *
 * Mirrors docs/analytics.js in neural-network-assignments-ui-explorer: same GA4
 * property, consent setting, opt-out key and generic events, so all explorers can be
 * analysed together. Events from this page carry assignment = 'final'.
 *
 * The explorer code itself is untouched: `defer` runs this file after the page's own
 * scripts, so it can wrap their global functions (toggleModel, goToCompare,
 * switchRawTab, ...) and add document-level listeners. Every hook first checks that
 * its function exists -- renaming one in run_explorer.py only silences that event.
 * The full event list lives in ANALYTICS.md at the repo root.
 *
 * Privacy: no names, IDs or file names are ever sent -- the uploaded my_model.csv
 * file name could contain a student's name, so only its scores are sent.
 *
 * Debugging: add ?ga_debug=1 to the page URL to log every event to the console and
 * show it in GA4's DebugView (works on localhost too). ?notrack=1 opts this browser
 * out for good (use it on your own machines); ?notrack=0 undoes it. The opt-out is
 * shared with the assignment explorers (same site, same storage key).
 */
(() => {
'use strict';

const CONFIG = {
  // GA4 -> Admin -> Data streams -> (your web stream) -> Measurement ID
  measurementId: 'G-T41SB4WQ70',
  // Only these hostnames send data (any host does with ?ga_debug=1).
  hosts: ['ai-sandbox-ai-healthcare.github.io'],
  // 'opt-in' : nothing is sent until the student clicks Allow.
  // 'notice' : tracking is on by default; the banner offers an opt-out.
  // 'none'   : no banner -- only if your IRB / course policy allows it.
  consent: 'none',
  noticeText: 'This explorer can record anonymous usage data (tabs, clicks, model '
    + 'selections and uploaded scores) with Google Analytics cookies, to improve the '
    + 'course and for education research. Your name and file names are never sent.',
  infoUrl: '',            // optional "Learn more" link, e.g. an IRB information sheet
};

const ID = CONFIG.measurementId;
const qs = new URLSearchParams(location.search);
const DEBUG = qs.has('ga_debug');
const KEY = 'nnui_ga_consent';                       // shared with the assignment explorers
const UPLOAD_KEY = 'myModelUpload';                  // written by run_explorer.py's upload code
const REPORTED_KEY = 'nnfp_ga_upload_reported';      // loadedAt of the last upload already sent
const ASSIGNMENT = 'final';
const now = () => Math.round(performance.now());
const warn = e => { if (DEBUG) console.warn('[ga] hook failed', e); };

const store = {
  get(k = KEY) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(v, k = KEY) { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch (e) {} },
};
if (qs.has('notrack')) store.set(qs.get('notrack') === '0' ? null : 'denied');

const live = /^G-[A-Z0-9]+$/.test(ID) && ID !== 'G-XXXXXXXXXX'
  && (DEBUG || CONFIG.hosts.includes(location.hostname));
if (!live && !DEBUG) return;
if (!live) console.info('[ga] measurementId not set -- events are logged here, not sent');

// 'granted' | 'denied' | null (undecided: events wait in `buffer`)
const stored = store.get();
let consent = stored === 'granted' || stored === 'denied' ? stored
  : CONFIG.consent === 'opt-in' ? null : 'granted';
const buffer = [];

// ============================================================
// gtag + consent
// ============================================================
let started = false;
function startGtag() {
  if (started || !live) return;
  started = true;
  window.dataLayer = window.dataLayer || [];
  window.gtag = function () { dataLayer.push(arguments); };
  gtag('js', new Date());
  gtag('config', ID, Object.assign({
    assignment: ASSIGNMENT,
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
  }, DEBUG ? { debug_mode: true } : {}));
  const s = document.createElement('script');
  s.async = true;
  s.src = 'https://www.googletagmanager.com/gtag/js?id=' + ID;
  document.head.appendChild(s);
}

function decide(choice) {
  store.set(choice);
  consent = choice;
  if (choice === 'granted') {
    startGtag();
    if (started) buffer.forEach(([name, ev]) => gtag('event', name, ev));
  } else if (started) {
    window['ga-disable-' + ID] = true;
  }
  buffer.length = 0;
}

function banner() {
  const optIn = CONFIG.consent === 'opt-in';
  const box = document.createElement('div');
  box.id = 'ga-consent';
  box.setAttribute('role', 'region');
  box.setAttribute('aria-label', 'Usage data notice');
  box.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:2147483000;'
    + 'max-width:min(400px,calc(100vw - 32px));background:#2e1f10;color:#f5f0e8;'
    + 'font:13px/1.5 "Segoe UI",system-ui,sans-serif;'
    + 'padding:14px 16px;border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.3)';
  const btn = 'border:0;border-radius:6px;padding:6px 14px;font:inherit;font-weight:600;cursor:pointer;';
  box.innerHTML = '<div>' + CONFIG.noticeText
    + (CONFIG.infoUrl ? ' <a href="' + CONFIG.infoUrl + '" target="_blank" rel="noopener" style="color:#fcd9a0">Learn more</a>' : '')
    + '</div><div style="display:flex;gap:8px;justify-content:flex-end;margin-top:10px">'
    + '<button type="button" data-choice="denied" style="' + btn
    + 'background:transparent;color:#d4b896;box-shadow:inset 0 0 0 1px #7a5c3c">'
    + (optIn ? 'No thanks' : 'Opt out') + '</button>'
    + '<button type="button" data-choice="granted" style="' + btn + 'background:#e8a250;color:#2e1f10">'
    + (optIn ? 'Allow' : 'OK') + '</button></div>';
  box.addEventListener('click', e => {
    const choice = e.target.getAttribute && e.target.getAttribute('data-choice');
    if (!choice) return;
    decide(choice);
    box.remove();
  });
  document.body.appendChild(box);
}

// ============================================================
// send()
// ============================================================
// The tab's own name ("Models", "Deliverables"), without the selection counter or the
// "(Tasks & Submission)" note, so each tab keeps one stable name in GA reports.
const tabLabel = b => ((b.childNodes[0] && b.childNodes[0].textContent) || b.textContent).replace(/\s+/g, ' ').trim();
function currentTab() {
  const b = document.querySelector('#tab-nav .tab-btn.active');
  return b ? tabLabel(b) : ASSIGNMENT;
}

// Random per page load: groups one visit's events (seq orders them) in BigQuery.
const PAGE_LOAD_ID = Math.random().toString(36).slice(2, 10);
let seq = 0;
function send(name, params) {
  const ev = Object.assign({
    assignment: ASSIGNMENT,
    tab_name: currentTab(),
    page_load_id: PAGE_LOAD_ID,
    t_ms: now(),          // ms since page load -- survives buffering
    seq: ++seq,
  }, params);
  // GA4 limits: 25 params per event, 100-character string values.
  for (const k of Object.keys(ev)) {
    const v = ev[k];
    if (v === undefined || v === null || v === '' || (typeof v === 'number' && !isFinite(v))) delete ev[k];
    else if (typeof v === 'number') ev[k] = Number.isInteger(v) ? v : +v.toFixed(4);
    else ev[k] = String(v).slice(0, 100);
  }
  if (DEBUG) { ev.debug_mode = true; console.debug('[ga]', name, ev); }
  if (!live || consent === 'denied') return;
  if (consent === null) { if (buffer.length < 200) buffer.push([name, ev]); return; }
  gtag('event', name, ev);
}

// Wrap a global function of the explorer page. `before(args)` runs first and its
// result is handed to `after(args, returnValue, ctx)`, which runs once the original
// has finished (awaited if it is async).
function hook(name, after, before) {
  const orig = window[name];
  if (typeof orig !== 'function') return;
  window[name] = function (...args) {
    let ctx;
    try { ctx = before && before(args); } catch (e) { warn(e); }
    const r = orig.apply(this, args);
    if (after) Promise.resolve(r).then(v => { try { after(args, v, ctx); } catch (e) { warn(e); } }, () => {});
    return r;
  };
}

// Reads a top-level constant of the explorer page, or `fallback` if it is missing.
function pageConst(read, fallback) {
  try { const v = read(); return v === undefined ? fallback : v; } catch (e) { return fallback; }
}

// ============================================================
// Active time (visible + input within the last 30 s), overall and per tab
// ============================================================
let lastInput = now(), activeMs = 0;
const tabActive = {};
['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart', 'scroll'].forEach(t =>
  document.addEventListener(t, () => { lastInput = now(); }, { capture: true, passive: true }));
setInterval(() => {
  if (document.hidden || now() - lastInput > 30000) return;
  activeMs += 5000;
  const t = currentTab();
  tabActive[t] = (tabActive[t] || 0) + 5000;
}, 5000);

// ============================================================
// Semantic events -- explorer functions
// ============================================================
// Tabs: some switches come from the nav, others from buttons such as "Compare Selected"
// that set the active class directly, so watch the tab buttons' classes instead.
let tab = currentTab(), tabSince = now(), tabActiveAtEnter = 0;
const tabsSeen = new Set([tab]);
const tabNav = document.getElementById('tab-nav');
if (tabNav && window.MutationObserver) new MutationObserver(() => {
  const to = currentTab();
  if (to === tab) return;
  send('tab_view', {
    from_tab: tab,
    from_tab_ms: now() - tabSince,
    from_tab_active_ms: (tabActive[tab] || 0) - tabActiveAtEnter,
    first_time: !tabsSeen.has(to),
  });
  tab = to; tabSince = now(); tabActiveAtEnter = tabActive[to] || 0;
  tabsSeen.add(to);
}).observe(tabNav, { subtree: true, attributes: true, attributeFilter: ['class'] });

// Dataset tab: which raw CSV files students open.
const filesSeen = new Set();
hook('switchRawTab', ([file]) => {
  send('dataset_file_view', { file, first_time: !filesSeen.has(file) });
  filesSeen.add(file);
});

// Models tab -> Compare tab.
const selectedIds = () => pageConst(() => [...state.selectedIds], []);
hook('toggleModel', ([id]) => {
  const ids = selectedIds();
  send('model_select', { model_id: id, selected: ids.includes(id), selected_count: ids.length });
});
let compares = 0;
hook('goToCompare', () => {
  const ids = selectedIds();
  compares++;
  send('compare_view', { models: ids.join(','), model_count: ids.length, includes_my_model: ids.includes('my_model'), compare_index: compares });
});
hook('setMetric', ([, metric]) => send('compare_metric', { metric }));

// Deliverables tab: results file.
hook('downloadMyModelTemplate', () => send('template_download'));
hook('showUploadErrors', ([, problems]) => send('my_model_upload_failed', {
  problem_count: (problems || []).length,
  first_problem: (problems || [])[0],
}));

// A successful upload reloads the page, so it is reported here on the next page view:
// once per upload (keyed by its loadedAt), with the scores and the milestones reached.
function f1Band(f1) {
  const lo = Math.floor(f1 * 20) / 20;
  return lo.toFixed(2) + '-' + (lo + 0.05).toFixed(2);
}
function reportUpload() {
  let up = null;
  try { up = JSON.parse(store.get(UPLOAD_KEY)); } catch (e) { return; }
  if (!up || !up.stats || !up.loadedAt || store.get(REPORTED_KEY) === up.loadedAt) return;
  const s = up.stats, f1 = s.f1;
  const cpuBest = pageConst(() => CPU_BEST_F1, null);
  const gpuBest = pageConst(() => GPU_BEST_F1, null);
  const goal = pageConst(() => GOOD_F1, 0.75);
  const beatCpu = cpuBest !== null && f1 > cpuBest;
  const beatGpu = gpuBest !== null && f1 > gpuBest;
  const good = f1 > goal;
  send('my_model_uploaded', {
    f1, auc: s.auc, accuracy: s.accuracy, precision: s.precision, recall: s.recall,
    runs: s.n_iterations, f1_band: f1Band(f1), elapsed_hr: s.elapsed_hr, gpu_hrs: s.gpu_hrs,
    gpu_tier_unlocked: beatCpu, task_complete: beatGpu, above_goal: good,
  });
  if (beatCpu) send('milestone_gpu_tier_unlocked', { f1, threshold: cpuBest });
  if (beatGpu) send('milestone_task_complete', { f1, threshold: gpuBest });
  if (good) send('milestone_good_performance', { f1, threshold: goal });
  if (live) store.set(up.loadedAt, REPORTED_KEY);
}

// ============================================================
// Generic events -- any element, including ones added later
// ============================================================
const CLICKABLE = 'button, a[href], [onclick], [role="button"], summary, label, input[type="checkbox"], input[type="radio"]';
const FIELD = 'input[type="range"], input[type="number"], input[type="checkbox"], input[type="radio"], select';

let clicks = 0, lastClick = { el: null, t: 0, n: 0 };
document.addEventListener('click', e => {
  const el = e.target.closest && e.target.closest(CLICKABLE);
  if (!el || el.closest('#ga-consent')) return;
  const t = now();
  lastClick = el === lastClick.el && t - lastClick.t < 800
    ? { el, t, n: lastClick.n + 1 } : { el, t, n: 1 };
  const d = {
    el_id: el.id,
    el_tag: el.tagName.toLowerCase(),
    el_text: (el.classList.contains('tab-btn') ? tabLabel(el)
      : (el.getAttribute('aria-label') || el.textContent || '').replace(/\s+/g, ' ').trim()).slice(0, 80),
    el_action: (el.getAttribute('onclick') || '').trim(),
    link_url: el.href,
  };
  clicks++;
  send('ui_click', d);
  if (lastClick.n === 3) send('rage_click', d);
  // "Clear upload" is wired with a direct function reference, so it is caught here.
  if (el.classList.contains('mm-link')) send('my_model_upload_cleared');
}, true);

// Sliders / selects / checkboxes: one event per committed change, with the value
// before the drag or keypress and how long the adjustment took.
const fieldValue = el => (el.type === 'checkbox' || el.type === 'radio' ? String(el.checked) : el.value);
const armed = new WeakMap();
const arm = e => {
  const el = e.target;
  if (el.matches && el.matches(FIELD)) armed.set(el, { v: fieldValue(el), t: now() });
};
document.addEventListener('pointerdown', arm, true);
document.addEventListener('keydown', arm, true);
document.addEventListener('change', e => {
  const el = e.target;
  if (!el.matches || !el.matches(FIELD)) return;
  const a = armed.get(el);
  armed.delete(el);
  send('ui_change', {
    el_id: el.id || el.name, el_type: el.type, from_value: a && a.v,
    to_value: fieldValue(el), adjust_ms: a ? now() - a.t : undefined,
  });
}, true);

// Scroll depth per tab and per scrolling box (page, tables, column guides, ...).
const depthSent = {};
const scrollTimers = new Map();
document.addEventListener('scroll', e => {
  const el = e.target === document ? document.scrollingElement : e.target;
  if (!el || !el.classList || scrollTimers.has(el)) return;
  scrollTimers.set(el, setTimeout(() => {
    scrollTimers.delete(el);
    const room = el.scrollHeight - el.clientHeight;
    if (room < 50) return;
    const pct = el.scrollTop / room * 100;
    const scroller = el === document.scrollingElement ? 'page' : el.id || el.classList[0] || el.tagName.toLowerCase();
    const key = currentTab() + '|' + scroller;
    const sent = depthSent[key] || (depthSent[key] = new Set());
    [25, 50, 75, 100].forEach(p => {
      if (pct >= p - 1 && !sent.has(p)) { sent.add(p); send('scroll_depth', { scroller, percent: p }); }
    });
  }, 250));
}, { capture: true, passive: true });

document.addEventListener('copy', () => send('text_copy', { text_length: String(getSelection()).length }));

// page_hidden doubles as the running summary of the visit: the last one sent is the
// final tally when the student closes the tab.
let hiddenAt = null;
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    hiddenAt = now();
    send('page_hidden', {
      active_ms: activeMs, tabs_seen: tabsSeen.size, files_seen: filesSeen.size,
      compare_count: compares, click_count: clicks,
    });
  } else if (hiddenAt !== null) {
    send('page_visible', { away_ms: now() - hiddenAt });
  }
});

let errors = 0;
window.addEventListener('error', e => {
  if (errors++ < 10) send('js_error', {
    message: e.message, source: (e.filename || '').split('?')[0].split('/').pop() || 'index.html', line: e.lineno });
});
window.addEventListener('unhandledrejection', e => {
  if (errors++ < 10) send('js_error', { message: (e.reason && e.reason.message) || e.reason, source: 'promise' });
});

function clientInfo() {
  const nav = performance.getEntriesByType ? performance.getEntriesByType('navigation')[0] : null;
  send('client_info', {
    viewport_w: innerWidth, viewport_h: innerHeight, pixel_ratio: devicePixelRatio,
    touch: navigator.maxTouchPoints > 0,
    dark_mode: matchMedia('(prefers-color-scheme: dark)').matches,
    load_ms: nav && nav.loadEventEnd ? Math.round(nav.loadEventEnd) : undefined,
  });
}

// ============================================================
// Start
// ============================================================
if (consent === 'granted') startGtag();
if (CONFIG.consent !== 'none' && !stored) banner();
reportUpload();
if (document.readyState === 'complete') setTimeout(clientInfo, 0);
else window.addEventListener('load', () => setTimeout(clientInfo, 0));
})();
