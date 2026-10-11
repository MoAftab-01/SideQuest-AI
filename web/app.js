/* =====================================================================
   SideQuest — front end
   ---------------------------------------------------------------------
   Plain script, no modules, no build step, no CDN. Everything the app
   needs is in this file plus styles.css, so it runs offline after the
   service worker has cached the shell.

   THE ONE RULE THAT SHAPES THIS FILE
   -----------------------------------
   Model output is untrusted text and is never trusted to describe itself.
   The verdict badge is derived *only* from `verdict.is_ai_verified`, which
   the backend sets true only when a model actually ran and produced
   VERIFIED or LIKELY. Everything else — heuristic, unavailable, and every
   failure — is rendered in a visibly different, non-success style. A
   deterministic rule must never be dressed up as a model judgement, and
   neither the markup nor the copy is allowed to imply otherwise.
   ===================================================================== */

(function () {
  'use strict';

  /* ------------------------------------------------------------------
     0. Small helpers
     ------------------------------------------------------------------ */

  var $ = function (id) { return document.getElementById(id); };

  /** Create an element. `text` is assigned with textContent, never HTML. */
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  /** Clear a node without innerHTML (which would be unsafe for model text). */
  function clear(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

  function plural(n, one, many) { return n === 1 ? one : many; }

  var LS_PLAYER = 'sidequest.player_id';
  var LS_SESSION = 'sidequest.session';   // active mission, survives a reload
  var LS_PREF = 'sidequest.prefs';

  /* ------------------------------------------------------------------
     1. API layer
     ------------------------------------------------------------------
     Same-origin when the app is served by the backend (the normal case).
     Opening index.html straight off disk falls back to localhost so the
     demo still works from file://.
     ------------------------------------------------------------------ */

  var API_BASE = (function () {
    var q = new URLSearchParams(location.search).get('api');
    if (q) return q.replace(/\/+$/, '');
    if (location.protocol === 'file:') return 'http://127.0.0.1:8000';
    return '';
  })();

  var TIMEOUT_SHORT = 20000;   // state / health
  var TIMEOUT_LONG  = 60000;   // quest + verify: a model may be thinking

  /** An error that carries a message a player can actually read. */
  function ApiError(message, status, detail) {
    this.name = 'ApiError';
    this.message = message || 'Request failed.';
    this.status = status || 0;
    this.detail = detail || '';
  }
  ApiError.prototype = Object.create(Error.prototype);

  function isOffline() { return typeof navigator !== 'undefined' && navigator.onLine === false; }

  /**
   * fetch with a hard timeout and a real error.
   * Aborts on timeout rather than hanging forever — a silent hang is the
   * one failure mode a demo cannot afford.
   */
  function apiGet(path, timeout) {
    return apiFetch(path, { method: 'GET' }, timeout || TIMEOUT_SHORT);
  }

  function apiPost(path, body, timeout) {
    return apiFetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }, timeout || TIMEOUT_LONG);
  }

  function apiFetch(path, init, timeout) {
    if (isOffline()) {
      return Promise.reject(new ApiError('You are offline. Reconnect and try again.', 0, 'offline'));
    }
    var ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, timeout) : null;

    return fetch(API_BASE + path, Object.assign({}, init, ctrl ? { signal: ctrl.signal } : {}))
      .then(function (res) {
        if (timer) clearTimeout(timer);
        return res.text().then(function (text) {
          var data = null;
          try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }

          if (!res.ok) {
            var detail = (data && (data.detail || data.message)) || '';
            var msg;
            if (res.status === 404) {
              msg = 'That mission is no longer on the server. Generate a new one.';
            } else if (res.status === 422) {
              msg = 'The server rejected the request as invalid.';
            } else if (res.status >= 500) {
              msg = 'The server hit an internal error.';
            } else {
              msg = 'Request failed (HTTP ' + res.status + ').';
            }
            throw new ApiError(msg, res.status, typeof detail === 'string' ? detail : JSON.stringify(detail));
          }
          if (data === null) throw new ApiError('The server sent a response we could not read.', res.status, text.slice(0, 200));
          return data;
        });
      })
      .catch(function (err) {
        if (timer) clearTimeout(timer);
        if (err instanceof ApiError) throw err;
        if (err && err.name === 'AbortError') {
          throw new ApiError('The server took too long to answer and the request was cancelled.', 0, 'timeout');
        }
        // A CORS failure or a dead backend lands here as a bare TypeError.
        throw new ApiError('Could not reach the SideQuest server.', 0, (err && err.message) || 'network');
      });
  }

  /* ------------------------------------------------------------------
     2. Honest vocabulary
     ------------------------------------------------------------------
     One table maps a confidence level to everything the player sees. Keeping
     it in one place is what stops a level from being styled two ways.
     ------------------------------------------------------------------ */

  var LEVELS = {
    verified: {
      variant: 'verified', label: 'AI VERIFIED', kind: 'headline',
      why: 'A model looked at your photo and matched it against this mission’s requirements.'
    },
    likely: {
      variant: 'likely', label: 'VERIFIED — WITH RESERVATIONS', kind: 'headline',
      why: 'A model ran and the photo broadly fits, but it was not a clean match.'
    },
    uncertain: {
      variant: 'uncertain', label: 'UNCERTAIN', kind: 'headline',
      why: 'A model ran but would not commit to a judgement. This is not a pass and not a fail.'
    },
    rejected: {
      variant: 'rejected', label: 'REJECTED', kind: 'headline',
      why: 'A model ran and the photo did not match what this mission asked for.'
    },
    heuristic: {
      variant: 'notchecked', label: 'NOT CHECKED BY AI', kind: 'caveat',
      why: 'No model looked at your evidence. This result came from a fixed rule in the app’s code, ' +
           'so it confirms nothing about your photo.'
    },
    unavailable: {
      variant: 'notchecked', label: 'COULD NOT BE CHECKED', kind: 'caveat',
      why: 'Neither a model nor a fallback could produce a judgement for this submission.'
    }
  };

  function levelInfo(level) {
    return LEVELS[level] || {
      variant: 'notchecked', label: 'UNRECOGNISED RESULT', kind: 'caveat',
      why: 'The server returned a confidence level this build does not recognise, so no claim is being made about it.'
    };
  }

  /* ------------------------------------------------------------------
     3. Badges
     ------------------------------------------------------------------
     A badge is the app's most load-bearing UI element: it is the only
     place a claim about verification is made. Two kinds:
       • headline — describes the verdict level
       • provenance — describes how a piece of work was produced
     ------------------------------------------------------------------ */

  /**
   * @param {object} o
   * @param {string} o.variant  CSS class suffix
   * @param {string} o.label    the claim, in caps
   * @param {string} [o.why]    explanation, always visible (mobile has no hover)
   * @param {string} [o.tip]    short text for the ⓘ tooltip
   */
  function makeBadge(o) {
    var b = el('div', 'badge ' + o.variant);
    b.appendChild(el('span', 'b-dot'));

    var body = el('div');
    var head = el('div');
    head.appendChild(el('b', null, o.label));

    if (o.tip) {
      var info = el('button', 'why', 'i');
      info.type = 'button';
      info.title = o.tip;
      info.setAttribute('aria-label', 'What this badge means: ' + o.tip);
      head.appendChild(info);
    }
    body.appendChild(head);
    if (o.why) body.appendChild(el('span', 'b-why', o.why));
    b.appendChild(body);
    return b;
  }

  function badgesInto(slot, badges) {
    clear(slot);
    if (!badges.length) return;
    var wrap = el('div', 'badges');
    badges.forEach(function (x) { wrap.appendChild(x); });
    slot.appendChild(wrap);
  }

  /**
   * The verdict headline badges.
   * THE HONESTY GATE lives here: `is_ai_verified === true` is the only
   * condition that produces the "AI VERIFIED" badge. Nothing else — not a
   * high confidence number, not a heuristic that agrees, not a model id
   * being configured — can reach it.
   */
  function verdictBadges(v) {
    var info = levelInfo(v && v.level);
    var out = [];

    if (v && v.is_ai_verified === true) {
      // Only this exact branch may say AI VERIFIED.
      out.push(makeBadge({
        variant: (v.level === 'likely') ? 'likely' : 'verified',
        label: 'AI VERIFIED',
        why: info.why,
        tip: 'The server set is_ai_verified=true, which it only does when a model ' +
             'actually ran and returned VERIFIED or LIKELY.'
      }));
    } else {
      // Everything else is a caveat, styled deliberately muted/dashed.
      out.push(makeBadge({
        variant: info.variant,
        label: info.label,
        why: info.why,
        tip: 'No AI judgement was made, so this result cannot confirm your mission.'
      }));
    }

    // If the server claims AI verification, say which model did it.
    if (v && v.is_ai_verified === true && v.model_id) {
      out.push(makeBadge({
        variant: 'gen-cached',
        label: 'JUDGED BY: ' + String(v.model_id).toUpperCase(),
        why: 'The open-weight model named above produced this judgement locally.'
      }));
    }
    return out;
  }

  /**
   * Provenance badge for mission generation.
   * A fallback mission is a real mission, but it is not a model-written one
   * and the player is entitled to know which they are looking at.
   */
  function generationBadge(gen, quest) {
    gen = gen || {};
    var source = gen.source || (quest && quest.source) || 'fallback';

    if (source === 'model') {
      var id = gen.model_id || (quest && quest.model_id);
      if (!id) return null;
      return makeBadge({
        variant: 'gen-cached',
        label: 'MISSION WRITTEN BY ' + String(id).toUpperCase(),
        why: 'Generated by the local open-weight model in ' +
             (typeof gen.latency_ms === 'number' ? gen.latency_ms + ' ms' : 'a model run') + '.'
      });
    }

    if (source === 'cached') {
      return makeBadge({
        variant: 'gen-cached',
        label: 'REUSED FROM CACHE',
        why: 'This mission was returned from a previous run rather than written fresh.'
      });
    }

    // source === 'fallback' — say so, in the muted caveat style.
    var reason = gen.fallback_reason || '';
    var why = 'No model was available, so this mission came from the app’s built-in ' +
              'curated template library. It is a real, playable mission, but no AI wrote it.';
    return makeBadge({
      variant: 'gen-fallback',
      label: 'GENERATED WITHOUT THE AI MODEL',
      why: why,
      tip: reason ? 'Reason: ' + reason.slice(0, 220) : 'No model weights were loaded.'
    });
  }

  /* ------------------------------------------------------------------
     4. Screen router
     ------------------------------------------------------------------ */

  var SCREENS = ['setup', 'mission', 'away', 'return', 'verdict', 'summary'];
  var current = 'setup';

  function show(name) {
    if (SCREENS.indexOf(name) === -1) name = 'setup';
    current = name;
    SCREENS.forEach(function (s) {
      var node = $('screen-' + s);
      if (node) node.hidden = (s !== name);
    });
    document.body.dataset.screen = name;
    // Reset any sticky away-mode state when leaving it.
    if (name !== 'away') document.body.dataset.away = '';

    var h = $('screen-' + name + '-h');
    if (h) { try { h.focus({ preventScroll: true }); } catch (e) { h.focus(); } }
    if (name !== 'away') stopTicker();
  }

  function announce(msg, urgent) {
    var node = urgent ? $('live-alert') : $('live-status');
    if (!node) return;
    node.textContent = '';
    // A tick of delay makes repeated identical messages re-announce.
    setTimeout(function () { node.textContent = msg; }, 40);
  }

  /* ------------------------------------------------------------------
     5. Error surface
     ------------------------------------------------------------------ */

  var lastRetry = null;

  function showError(title, body, retryFn) {
    $('err-title').textContent = title;
    $('err-body').textContent = body;
    $('btn-err-retry').hidden = !retryFn;
    lastRetry = retryFn || null;
    $('err-bar').hidden = false;
    announce(title + '. ' + body, true);
  }

  function hideError() { $('err-bar').hidden = true; lastRetry = null; }

  function describeError(err, what) {
    if (err instanceof ApiError) {
      if (err.status === 0 && err.detail === 'offline') {
        return ['You are offline', 'SideQuest needs the local server to generate and check missions. Reconnect and try again.'];
      }
      if (err.status === 0 && err.detail === 'timeout') {
        return ['That took too long', 'The server did not answer in time. It may still be loading model weights — try again in a moment.'];
      }
      if (err.status === 0) {
        return ['Cannot reach the server', 'Start the backend (uvicorn app.main:app) and try again.'];
      }
      return [what + ' failed', err.message + (err.detail ? ' (' + String(err.detail).slice(0, 120) + ')' : '')];
    }
    return [what + ' failed', (err && err.message) || 'An unexpected error occurred.'];
  }

  function fail(err, what, retryFn) {
    var pair = describeError(err, what);
    showError(pair[0], pair[1], retryFn);
  }

  /* ------------------------------------------------------------------
     6. Player identity
     ------------------------------------------------------------------ */

  var playerId = null;

  function loadPlayerId() {
    try { return localStorage.getItem(LS_PLAYER); } catch (e) { return null; }
  }
  function savePlayerId(id) {
    playerId = id;
    try { if (id) localStorage.setItem(LS_PLAYER, id); } catch (e) { /* private mode */ }
  }

  /* ------------------------------------------------------------------
     7. App state
     ------------------------------------------------------------------ */

  var state = {
    summary: null,
    health: null,
    quest: null,
    generation: null,
    startedAt: null,     // epoch ms, set when GO is pressed
    plannedMinutes: 0,
    photoDataUrl: null
  };

  function saveSession() {
    try {
      if (!state.quest) { localStorage.removeItem(LS_SESSION); return; }
      localStorage.setItem(LS_SESSION, JSON.stringify({
        quest: state.quest,
        generation: state.generation,
        startedAt: state.startedAt,
        planned: state.plannedMinutes
      }));
    } catch (e) { /* storage full or blocked; session is a convenience only */ }
  }

  function loadSession() {
    try {
      var raw = localStorage.getItem(LS_SESSION);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (e) { return null; }
  }

  /**
   * Forget the *stored* session so a reload does not resurrect a finished
   * mission. The in-memory quest is deliberately kept: the verdict screen
   * still needs its title, and `needs_better_photo` must be able to send the
   * player back to the return screen for another attempt against it.
   * Use resetQuest() when the quest is truly over.
   */
  function clearSession() {
    try { localStorage.removeItem(LS_SESSION); } catch (e) { /* ignore */ }
  }

  function resetQuest() {
    clearSession();
    state.quest = null;
    state.generation = null;
    state.startedAt = null;
    state.plannedMinutes = 0;
    state.photoDataUrl = null;
  }

  /* ------------------------------------------------------------------
     8. Setup screen
     ------------------------------------------------------------------ */

  var INTERESTS = [
    'birds', 'photography', 'sketching', 'plants', 'architecture',
    'sound', 'history', 'weather', 'kids', 'dogs'
  ];
  var PARTY = [
    ['solo', 'Solo'], ['with_kids', 'With kids'], ['with_dog', 'With a dog'], ['group', 'Group']
  ];
  var ENERGY = [
    ['low', 'Low'], ['medium', 'Medium'], ['high', 'High']
  ];
  var LOCATIONS = [
    ['unknown', "Wherever I am"], ['neighborhood', 'Neighbourhood'], ['park', 'Park'],
    ['city', 'City streets'], ['trail', 'Trail'], ['waterfront', 'Waterfront'],
    ['indoor_mall', 'Indoor (rain plan)']
  ];
  var WEATHERS = [
    ['unknown', 'No idea'], ['clear', 'Clear'], ['cloudy', 'Cloudy'], ['rain', 'Rain'],
    ['heat', 'Hot'], ['cold', 'Cold'], ['snow', 'Snow'], ['windy', 'Windy']
  ];
  var MOBILITY = [
    ['unrestricted', 'No restrictions'], ['avoid_stairs', 'Avoid stairs'], ['step_free', 'Step-free only']
  ];

  var chosenInterests = [];

  function buildSetup() {
    // --- interest chips (multi-select, capped at the API's max of 8) ---
    var chipWrap = $('interest-chips');
    clear(chipWrap);
    INTERESTS.forEach(function (name) {
      var b = el('button', 'chip');
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.dataset.value = name;
      b.appendChild(el('span', 'tick', '✓'));
      b.appendChild(el('span', null, name));
      b.addEventListener('click', function () {
        var i = chosenInterests.indexOf(name);
        if (i === -1) {
          if (chosenInterests.length >= 8) {
            announce('Eight interests is the maximum. Remove one first.');
            return;
          }
          chosenInterests.push(name);
        } else {
          chosenInterests.splice(i, 1);
        }
        b.setAttribute('aria-pressed', chosenInterests.indexOf(name) === -1 ? 'false' : 'true');
        updateInterestCount();
      });
      chipWrap.appendChild(b);
    });

    // --- segmented controls ---
    buildSeg($('party-seg'), PARTY, 'solo');
    buildSeg($('energy-seg'), ENERGY, 'medium');

    // --- selects ---
    fillSelect($('sel-location'), LOCATIONS);
    fillSelect($('sel-weather'), WEATHERS);
    fillSelect($('sel-mobility'), MOBILITY);

    // --- time slider ---
    var slider = $('time-budget');
    var out = $('time-out');
    function paint() {
      var v = parseInt(slider.value, 10);
      out.textContent = v >= 60
        ? (v % 60 === 0 ? (v / 60) + ' hr' : Math.floor(v / 60) + 'h ' + (v % 60) + 'm')
        : v + ' min';
    }
    slider.addEventListener('input', paint);
    paint();

    restorePrefs();
  }

  function buildSeg(host, pairs, initial) {
    clear(host);
    pairs.forEach(function (p) {
      var b = el('button', 'seg-btn', p[1]);
      b.type = 'button';
      b.dataset.value = p[0];
      b.setAttribute('aria-pressed', p[0] === initial ? 'true' : 'false');
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(host.children, function (c) {
          c.setAttribute('aria-pressed', c === b ? 'true' : 'false');
        });
        savePrefs();
      });
      host.appendChild(b);
    });
  }

  function fillSelect(sel, pairs) {
    clear(sel);
    pairs.forEach(function (p) {
      var o = el('option', null, p[1]);
      o.value = p[0];
      sel.appendChild(o);
    });
  }

  function updateInterestCount() {
    var n = chosenInterests.length;
    $('interest-count').textContent = n + ' of 8 picked.';
  }

  function segValue(id, fallback) {
    var host = $(id);
    var on = host.querySelector('[aria-pressed="true"]');
    return on ? on.dataset.value : fallback;
  }

  function savePrefs() {
    try {
      localStorage.setItem(LS_PREF, JSON.stringify({
        minutes: $('time-budget').value,
        interests: chosenInterests,
        party: segValue('party-seg', 'solo'),
        energy: segValue('energy-seg', 'medium'),
        location: $('sel-location').value,
        weather: $('sel-weather').value,
        mobility: $('sel-mobility').value,
        radius: $('sel-radius').value
      }));
    } catch (e) { /* ignore */ }
  }

  function restorePrefs() {
    var p = null;
    try { p = JSON.parse(localStorage.getItem(LS_PREF) || 'null'); } catch (e) { p = null; }
    if (!p) { updateInterestCount(); return; }

    if (p.minutes) { $('time-budget').value = p.minutes; $('time-out').textContent = formatMinutes(parseInt(p.minutes, 10)); }
    if (Array.isArray(p.interests)) {
      chosenInterests = p.interests.slice(0, 8);
      Array.prototype.forEach.call($('interest-chips').children, function (c) {
        c.setAttribute('aria-pressed', chosenInterests.indexOf(c.dataset.value) === -1 ? 'false' : 'true');
      });
      updateInterestCount();
    }
    if (p.party) setSeg('party-seg', p.party);
    if (p.energy) setSeg('energy-seg', p.energy);
    if (p.location) $('sel-location').value = p.location;
    if (p.weather) $('sel-weather').value = p.weather;
    if (p.mobility) $('sel-mobility').value = p.mobility;
    if (p.radius) $('sel-radius').value = p.radius;
  }

  function setSeg(hostId, value) {
    Array.prototype.forEach.call($(hostId).children, function (c) {
      c.setAttribute('aria-pressed', c.dataset.value === value ? 'true' : 'false');
    });
  }

  function formatMinutes(m) {
    if (m >= 60 && m % 60 === 0) return (m / 60) + ' hr';
    if (m >= 60) return Math.floor(m / 60) + 'h ' + (m % 60) + 'm';
    return m + ' min';
  }

  /** Build the QuestRequest body from the form. Fields match schemas.py exactly. */
  function buildQuestRequest() {
    var avoidRaw = ($('inp-avoid').value || '').split(',');
    var avoid = avoidRaw
      .map(function (s) { return s.trim().toLowerCase(); })
      .filter(function (s) { return s.length > 0; })
      .slice(0, 8);

    return {
      time_budget_minutes: parseInt($('time-budget').value, 10),
      interests: chosenInterests.slice(0, 8),
      party: segValue('party-seg', 'solo'),
      mobility: $('sel-mobility').value,
      energy: segValue('energy-seg', 'medium'),
      location_type: $('sel-location').value,
      weather: $('sel-weather').value,
      max_radius_m: parseInt($('sel-radius').value, 10),
      avoid: avoid,
      context_note: ($('inp-note').value || '').trim().slice(0, 400)
    };
  }

  /* ------------------------------------------------------------------
     9. Generate a mission
     ------------------------------------------------------------------ */

  function generateQuest() {
    var btn = $('btn-generate');
    var req = buildQuestRequest();
    savePrefs();

    btn.disabled = true;
    btn.textContent = 'Generating…';
    hideError();
    announce('Generating a mission.');

    var q = 'player_id=' + encodeURIComponent(playerId || '');
    apiPost('/api/quest?' + q, req, TIMEOUT_LONG)
      .then(function (data) {
        if (!data.quest) throw new ApiError('The server returned no mission.', 0, 'no quest in payload');
        // The server may mint a player id if ours was unknown.
        if (data.player_id) savePlayerId(data.player_id);

        state.quest = data.quest;
        state.generation = data.generation || {};
        state.startedAt = null;
        state.plannedMinutes = data.quest.estimated_minutes || req.time_budget_minutes;
        state.photoDataUrl = null;
        saveSession();
        renderMission();
        show('mission');
        announce('Mission ready: ' + data.quest.title);
      })
      .catch(function (err) {
        fail(err, 'Generating a mission', generateQuest);
      })
      .then(function () {
        btn.disabled = false;
        btn.textContent = 'Generate a mission';
      });
  }

  /* ------------------------------------------------------------------
     10. Mission card
     ------------------------------------------------------------------ */

  function renderMission() {
    var q = state.quest;
    if (!q) return;

    $('mission-time').textContent =
      formatMinutes(q.estimated_minutes) + ' · ' + q.xp_reward + ' XP';
    $('mission-h').textContent = q.title || 'Untitled mission';
    $('mission-hook').textContent = q.hook || '';

    // tags
    var tags = $('mission-tags');
    clear(tags);
    (q.tags || []).forEach(function (t) { tags.appendChild(el('li', null, t)); });

    // generation provenance — shown unconditionally where it matters
    var gb = generationBadge(state.generation, q);
    badgesInto($('gen-badge-slot'), gb ? [gb] : []);

    // steps
    var list = $('mission-steps');
    clear(list);
    (q.steps || []).forEach(function (s, i) {
      var li = el('li', 'step');
      li.appendChild(el('div', 'step-n', String(s.order || (i + 1))));

      var body = el('div', 'step-body');
      body.appendChild(el('p', 'step-instr', s.instruction || ''));
      if (s.detail) body.appendChild(el('p', 'step-detail', s.detail));
      li.appendChild(body);

      li.appendChild(el('div', 'step-min', s.approx_minutes ? s.approx_minutes + 'm' : ''));
      list.appendChild(li);
    });

    // verification requirement — the contract the photo will be judged against
    var v = q.verification || {};
    $('verify-subject').textContent = v.subject || 'No verification criteria were returned.';
    var must = $('verify-must');
    clear(must);
    (v.must_include || []).forEach(function (m) { must.appendChild(el('li', null, m)); });
    if (!(v.must_include || []).length) {
      must.appendChild(el('li', null, 'The server did not list specific requirements.'));
    }

    // safety notes — prominent when present
    var notes = dedupe((q.safety_notes || []).concat(
      ((q.safety && q.safety.findings) || []).map(function (f) { return f.message; })
    ));
    var panel = $('panel-safety');
    var sl = $('safety-list');
    clear(sl);
    if (notes.length) {
      notes.forEach(function (n) { sl.appendChild(el('li', null, n)); });
      panel.hidden = false;
    } else {
      panel.hidden = true;
    }
  }

  function dedupe(arr) {
    var seen = {}, out = [];
    (arr || []).forEach(function (x) {
      if (typeof x === 'string' && x.trim() && !seen[x]) { seen[x] = 1; out.push(x.trim()); }
    });
    return out;
  }

  /* ------------------------------------------------------------------
     11. Away mode — the anti-screen screen
     ------------------------------------------------------------------
     Design intent: after GO the app has one job, which is to stop being
     interesting. There is deliberately no Scroll Wake Lock here — keeping
     the panel on is the opposite of what this product wants. The countdown
     is visible once, then the phone is free to sleep.
     ------------------------------------------------------------------ */

  var ticker = null;

  function renderAway() {
    var q = state.quest;
    if (!q) { show('setup'); return; }

    $('away-kicker').textContent = state.startedAt ? 'Outside now' : 'Your mission · read it, then stop reading';
    $('away-h').textContent = q.title || '';

    var subject = (q.verification && q.verification.subject) || 'what the mission asked for';
    $('away-look-txt').textContent = subject;

    // Before GO the clock shows the plan; after GO it counts up.
    $('clock-num').textContent = String(state.plannedMinutes);
    $('clock-unit').textContent = 'min planned';
    $('clock-fill').style.width = '0%';

    $('btn-go').hidden = !!state.startedAt;
    $('btn-back-early').hidden = !state.startedAt;
    $('away-foot').hidden = !state.startedAt;

    document.body.dataset.away = state.startedAt ? 'gone' : '';
    updateClock();
  }

  function updateClock() {
    if (!state.startedAt || current !== 'away') return;

    var elapsedMin = Math.floor((Date.now() - state.startedAt) / 60000);
    var planned = state.plannedMinutes || 0;
    var num = $('clock-num');
    var unit = $('clock-unit');
    var fill = $('clock-fill');

    num.textContent = String(elapsedMin);

    if (planned > 0 && elapsedMin >= planned) {
      unit.textContent = 'min out · mission budget reached';
      fill.style.width = '100%';
    } else {
      unit.textContent = planned > 0
        ? 'min out · of ' + planned + ' planned'
        : 'min out';
      fill.style.width = planned > 0 ? clamp((elapsedMin / planned) * 100, 0, 100) + '%' : '0%';
    }

    // The timer is a polite live region so it does not interrupt a screen
    // reader every second; the state change is what matters.
    $('clock-sr').textContent = elapsedMin + ' ' + plural(elapsedMin, 'minute', 'minutes') + ' outside.';
  }

  function startTicker() {
    stopTicker();
    ticker = setInterval(updateClock, 1000);
  }
  function stopTicker() {
    if (ticker) { clearInterval(ticker); ticker = null; }
  }

  /* ------------------------------------------------------------------
     12. Return / evidence
     ------------------------------------------------------------------ */

  /** Downscale before upload: keeps the request small and the demo fast. */
  function fileToDataUrl(file) {
    return new Promise(function (resolve, reject) {
      if (!file) { reject(new Error('no file')); return; }
      if (!/^image\//.test(file.type)) {
        reject(new Error('That file is not an image. Pick a photo.'));
        return;
      }
      var reader = new FileReader();
      reader.onerror = function () { reject(new Error('The file could not be read.')); };
      reader.onload = function () {
        var img = new Image();
        img.onerror = function () { resolve(String(reader.result)); }; // send as-is
        img.onload = function () {
          try {
            var MAX = 1280;
            var scale = Math.min(1, MAX / Math.max(img.width, img.height));
            var w = Math.max(1, Math.round(img.width * scale));
            var h = Math.max(1, Math.round(img.height * scale));
            var cv = document.createElement('canvas');
            cv.width = w; cv.height = h;
            cv.getContext('2d').drawImage(img, 0, 0, w, h);
            resolve(cv.toDataURL('image/jpeg', 0.82));
          } catch (e) {
            resolve(String(reader.result));
          }
        };
        img.src = String(reader.result);
      };
      reader.readAsDataURL(file);
    });
  }

  function renderReturn() {
    var q = state.quest;
    if (!q) { show('setup'); return; }

    var mins = elapsedMinutes();
    $('inp-elapsed').value = String(mins);
    $('elapsed-help').textContent = mins > 0
      ? 'Counted from when you pressed GO (' + formatMinutes(mins) + ').'
      : 'Set this by hand if you went out without starting the timer.';

    $('return-eyebrow').textContent = state.startedAt
      ? 'Back from ' + formatMinutes(mins) + ' outside'
      : 'Back from outside';

    var subject = (q.verification && q.verification.subject) || '';
    $('return-need').textContent = subject
      ? 'Your photo is checked against: ' + subject
      : 'Add whatever you brought back.';

    // restore a photo if one was already attached
    if (state.photoDataUrl) showPhoto(state.photoDataUrl);
    else clearPhoto();
  }

  function elapsedMinutes() {
    if (!state.startedAt) return 0;
    return clamp(Math.floor((Date.now() - state.startedAt) / 60000), 0, 1440);
  }

  function showPhoto(dataUrl) {
    var img = $('photo-preview');
    img.src = dataUrl;
    img.hidden = false;
    $('photo-btn').hidden = true;
    $('btn-photo-clear').hidden = false;
    $('photo-status').textContent = 'Photo attached. It is sent only for model checking.';
    announce('Photo attached.');
  }

  function clearPhoto() {
    state.photoDataUrl = null;
    var img = $('photo-preview');
    img.removeAttribute('src');
    img.hidden = true;
    $('photo-btn').hidden = false;
    $('btn-photo-clear').hidden = true;
    $('photo-status').textContent = '';
    try { $('inp-photo').value = ''; } catch (e) { /* ignore */ }
  }

  /* ------------------------------------------------------------------
     13. Submit evidence
     ------------------------------------------------------------------ */

  function submitEvidence() {
    var q = state.quest;
    if (!q) { show('setup'); return; }

    var btn = $('btn-submit');
    var mins = clamp(parseInt($('inp-elapsed').value, 10) || 0, 0, 1440);

    var body = {
      photo_data_url: state.photoDataUrl || null,
      note: ($('inp-note2').value || '').trim().slice(0, 600),
      location_label: ($('inp-place').value || '').trim().slice(0, 120),
      elapsed_minutes: mins
    };

    btn.disabled = true;
    btn.textContent = 'Checking…';
    hideError();
    announce('Checking your evidence.');

    var qs = 'player_id=' + encodeURIComponent(playerId || '') +
             '&quest_id=' + encodeURIComponent(q.id);

    apiPost('/api/verify?' + qs, body, TIMEOUT_LONG)
      .then(function (data) {
        if (!data.verdict) throw new ApiError('The server returned no verdict.', 0, 'no verdict in payload');
        if (data.summary) state.summary = data.summary;
        clearSession();
        renderVerdict(data.verdict, data.verification || {});
        show('verdict');
      })
      .catch(function (err) {
        fail(err, 'Checking your evidence', submitEvidence);
      })
      .then(function () {
        btn.disabled = false;
        btn.textContent = 'Submit for checking';
      });
  }

  /* ------------------------------------------------------------------
     14. Verdict
     ------------------------------------------------------------------ */

  function renderVerdict(v, meta) {
    badgesInto($('verdict-badges'), verdictBadges(v));

    var info = levelInfo(v.level);
    $('verdict-h').textContent = info.label.charAt(0) + info.label.slice(1).toLowerCase();

    var q = state.quest;
    $('verdict-quest').textContent = q ? q.title : 'Your mission';

    $('verdict-summary').textContent = v.summary || 'The server returned no explanation.';

    /* --- XP, stated precisely ---------------------------------------
       The backend awards participation XP on the heuristic path. That is
       not verification and the copy must not read like it is.            */
    var xpLine = $('verdict-xp');
    clear(xpLine);
    var num = el('span', 'xp-num ' + (v.xp_awarded > 0 ? 'gain' : 'zero'),
                 (v.xp_awarded > 0 ? '+' : '') + (v.xp_awarded || 0));
    xpLine.appendChild(num);

    if (v.is_ai_verified === true) {
      xpLine.appendChild(el('span', 'xp-txt',
        'Awarded because a model verified this mission.'));
    } else if (v.xp_awarded > 0) {
      xpLine.appendChild(el('span', 'xp-txt',
        'Participation only. No AI verified this, so the full reward was not earned.'));
    } else {
      xpLine.appendChild(el('span', 'xp-txt', 'No XP awarded for this one.'));
    }

    /* --- matched / missing ------------------------------------------ */
    fillList($('panel-matched'), $('matched-list'), v.matched, 'Nothing was matched.');
    fillList($('panel-missing'), $('missing-list'), v.missing, null);

    /* --- retry affordance ------------------------------------------- */
    var retry = $('panel-retry');
    if (v.needs_better_photo === true) {
      retry.hidden = false;
      $('retry-why').textContent =
        'The checker wants better evidence. A second submission counts as another attempt ' +
        'against this mission, so only resubmit if you genuinely have a better photo.';
    } else {
      retry.hidden = true;
    }

    /* --- provenance: how this was checked --------------------------- */
    var notes = dedupe(meta.notes || []);
    var kv = $('vnotes-kv');
    clear(kv);
    var rows = [];
    rows.push(['Checked by', v.model_id ? v.model_id : (meta.model_id ? meta.model_id : 'no model — fixed rules only')]);
    rows.push(['Method', meta.source || v.source || 'fallback']);
    if (meta.latency_ms || v.latency_ms) {
      rows.push(['Time taken', (meta.latency_ms || v.latency_ms) + ' ms']);
    }
    if (meta.image_bytes) rows.push(['Image received', meta.image_bytes + ' bytes']);
    if (typeof v.confidence === 'number' && v.confidence > 0) {
      rows.push(['Confidence', Math.round(v.confidence * 100) + '%']);
    }
    rows.forEach(function (r) {
      kv.appendChild(el('dt', null, r[0]));
      kv.appendChild(el('dd', null, String(r[1])));
    });
    notes.forEach(function (n) {
      kv.appendChild(el('dt', null, 'note'));
      kv.appendChild(el('dd', null, n));
    });
    $('panel-vnotes').hidden = false;

    announce('Result: ' + info.label + '. ' + (v.summary || ''), true);
  }

  function fillList(panel, list, items, emptyText) {
    var arr = Array.isArray(items) ? items.filter(Boolean) : [];
    clear(list);
    if (!arr.length) {
      if (emptyText) {
        panel.hidden = false;
        var li = el('li', null, emptyText);
        list.appendChild(li);
      } else {
        panel.hidden = true;
      }
      return;
    }
    arr.forEach(function (x) { list.appendChild(el('li', null, x)); });
    panel.hidden = false;
  }

  /* ------------------------------------------------------------------
     15. Summary
     ------------------------------------------------------------------ */

  /** Mirrors xp_for_level() in backend/app/store.py. */
  function xpFloorForLevel(level) {
    if (level <= 1) return 0;
    return 100 * (level - 1) + 50 * ((level - 1) * (level - 2) / 2);
  }

  function renderSummary(s) {
    s = s || state.summary || {
      xp: 0, level: 1, xp_to_next_level: 100, minutes_outdoors: 0,
      verified_count: 0, uncertain_count: 0, rejected_count: 0,
      screen_time_saved_seconds: 0
    };

    $('sum-minutes').textContent = String(s.minutes_outdoors || 0);
    $('sum-level').textContent = String(s.level || 1);
    $('sum-xp').textContent = (s.xp || 0) + ' XP';

    var floor = xpFloorForLevel(s.level || 1);
    var span = Math.max(1, s.xp_to_next_level || 100);
    var pct = clamp(((s.xp - floor) / span) * 100, 0, 100);
    $('lvl-fill').style.width = pct + '%';
    var bar = $('lvl-bar');
    bar.setAttribute('aria-valuenow', String(Math.round(pct)));
    $('lvl-next').textContent = (s.xp_to_next_level || 0) +
      ' XP to level ' + ((s.level || 1) + 1) + '.';

    $('cnt-verified').textContent = String(s.verified_count || 0);
    $('cnt-uncertain').textContent = String(s.uncertain_count || 0);
    $('cnt-rejected').textContent = String(s.rejected_count || 0);

    var verified = s.verified_count || 0;
    $('counts-note').textContent = verified === 0
      ? 'Nothing has been verified by a model yet on this device. ' +
        'Verification requires a loaded vision model — see “How honest is this?”.'
      : 'Only results where a model actually ran are counted as verified.';

    var saved = Math.round((s.screen_time_saved_seconds || 0) / 60);
    $('sum-saved').textContent = saved + ' min';
  }

  /* ------------------------------------------------------------------
     16. About / engine status
     ------------------------------------------------------------------ */

  function renderAbout() {
    var body = $('about-body');
    clear(body);

    var eng = (state.health && state.health.engine) || null;

    body.appendChild(el('p', null,
      'SideQuest gives you one outdoor mission, then gets out of the way. ' +
      'This sheet exists because the app has to be able to tell you when it ' +
      'is not actually doing what its interface implies.'));

    var dl = el('dl', 'about-list');

    function item(label, value) {
      var row = el('div', 'about-item');
      row.appendChild(el('dt', null, label));
      var dd = el('dd');
      dd.textContent = value;
      row.appendChild(dd);
      dl.appendChild(row);
    }

    if (eng && eng.loaded === true) {
      item('Model', eng.model_id || 'unknown');
      item('Running on', eng.device + ' · ' + eng.dtype);
      item('Vision (photo checking)', eng.supports_vision
        ? 'available' : 'NOT available — photos cannot be judged by a model');
    } else {
      item('Model', 'none loaded');
      item('Vision (photo checking)',
        'unavailable — photos cannot be judged by a model, so nothing can be AI verified');
      item('Why',
        (eng && eng.load_error) ? String(eng.load_error).slice(0, 240) : 'Model weights are not installed.');
    }

    item('Where inference runs', 'On this machine. Nothing is sent anywhere else.');
    item('What leaves the machine',
      'Only the photo you attach, and only when you press “Submit for checking”. ' +
      'There is no analytics, no telemetry and no third-party request.');
    item('Your location',
      'Never requested and never stored. “Roughly where?” is a free-text label you type yourself.');

    body.appendChild(dl);

    var p = el('p');
    p.appendChild(el('strong', null, 'The rule this UI follows: '));
    p.appendChild(document.createTextNode(
      'the “AI VERIFIED” badge appears only when the server returns ' +
      'is_ai_verified = true, which happens only when a model actually ran and ' +
      'returned VERIFIED or LIKELY. Heuristic results are labelled “Not checked by AI” ' +
      'and are styled as a caveat, never as a success.'));
    body.appendChild(p);

    var p2 = el('p');
    p2.appendChild(el('strong', null, 'On the screen-time number: '));
    p2.appendChild(document.createTextNode(
      'it is your outdoor minutes restated as screen minutes. It is an estimate ' +
      'by arithmetic, not a measurement of anything.'));
    body.appendChild(p2);

    if (API_BASE) {
      var p3 = el('p');
      p3.appendChild(el('strong', null, 'Server: '));
      p3.appendChild(el('code', null, API_BASE || 'same origin'));
      body.appendChild(p3);
    }
  }

  /* ------------------------------------------------------------------
     17. Wiring
     ------------------------------------------------------------------ */

  function wire() {
    $('btn-generate').addEventListener('click', generateQuest);
    $('btn-reroll').addEventListener('click', generateQuest);

    $('btn-start').addEventListener('click', function () {
      show('away');
      startTicker();
      announce('Mission shown. Press GO to start, then put the phone away.');
    });

    $('btn-go').addEventListener('click', function () {
      state.startedAt = Date.now();
      saveSession();
      renderAway();
      startTicker();
      announce('Timer started. Put the phone away — the screen will sleep on its own.');
    });

    $('btn-back-early').addEventListener('click', function () {
      state.startedAt = null;
      saveSession();
      show('mission');
      announce('Timer reset.');
    });

    $('btn-still-out').addEventListener('click', function () {
      show('away');
      startTicker();
    });

    // photo
    $('inp-photo').addEventListener('change', function (e) {
      var f = e.target.files && e.target.files[0];
      if (!f) return;
      $('photo-status').textContent = 'Processing photo…';
      fileToDataUrl(f)
        .then(function (url) {
          state.photoDataUrl = url;
          showPhoto(url);
        })
        .catch(function (err) {
          $('photo-status').textContent = (err && err.message) || 'That file could not be used.';
          announce('That file could not be used as a photo.', true);
        });
    });
    $('btn-photo-clear').addEventListener('click', function () {
      clearPhoto();
      announce('Photo removed.');
    });

    $('btn-submit').addEventListener('click', submitEvidence);

    $('btn-retry').addEventListener('click', function () {
      // The mission stays on the server, so the player can submit again
      // against it; only the evidence attempt repeats.
      state.photoDataUrl = null;
      renderReturn();
      show('return');
      announce('Add a better photo and submit again.');
    });

    $('btn-summary').addEventListener('click', function () {
      refreshState().then(function () { show('summary'); renderSummary(state.summary); });
    });
    $('btn-again').addEventListener('click', function () { resetQuest(); show('setup'); });
    $('btn-new-mission').addEventListener('click', function () { resetQuest(); show('setup'); });

    // about sheet
    $('btn-about').addEventListener('click', openAbout);
    $('btn-about-2').addEventListener('click', openAbout);
    $('btn-about-close').addEventListener('click', closeAbout);
    // Home navigation: header logo/link and keyboard
    var btnHome = $('btn-home');
    if (btnHome) {
      btnHome.addEventListener('click', function (e) {
        if (e) e.preventDefault();
        show('setup');
        document.body.dataset.screen = 'setup';
        window.scrollTo(0, 0);
        announce('Returned to home.');
      });
    }
    document.addEventListener('keydown', function (e) {
      // Escape closes about sheet; Alt+H returns home from any screen (accessible)
      if (e.key === 'Escape' && !$('about-wrap').hidden) closeAbout();
      if (e.altKey && e.key === 'h') {
        e.preventDefault();
        show('setup');
        document.body.dataset.screen = 'setup';
        window.scrollTo(0, 0);
        announce('Returned to home.');
      }
    });

    // Browser Back support for nested screen URLs (hash persistence) without
    // duplicating navigation bars or breaking active states
    function syncScreenFromHash() {
      var h = location.hash.replace('#', '').trim();
      if (!h) return;
      if (SCREENS.indexOf(h) !== -1 && h !== current) {
        try { show(h); } catch (e) { /* ignore */ }
        // Preserve session on refresh so saved XP / journal entries stay intact
      }
    }
    window.addEventListener('popstate', syncScreenFromHash);
    window.addEventListener('hashchange', syncScreenFromHash);
    // On initial load, respect a hash that points to a valid screen
    syncScreenFromHash();
    // about sheet (continued — no duplicate listeners)
    $('about-scrim').addEventListener('click', closeAbout);

    // error bar
    $('btn-err-close').addEventListener('click', hideError);
    $('btn-err-retry').addEventListener('click', function () {
      var fn = lastRetry;
      hideError();
      if (fn) fn();
    });

    // keep the elapsed field honest if the return screen is left open a while
    document.addEventListener('visibilitychange', function () {
      if (!document.hidden && current === 'return' && state.startedAt) {
        $('inp-elapsed').value = String(elapsedMinutes());
      }
      if (!document.hidden && current === 'away' && state.startedAt) updateClock();
    });
  }

  var aboutOpener = null;

  function openAbout() {
    renderAbout();
    aboutOpener = document.activeElement;
    $('about-wrap').hidden = false;
    $('about-sheet').focus();
  }
  function closeAbout() {
    $('about-wrap').hidden = true;
    if (aboutOpener && aboutOpener.focus) aboutOpener.focus();
  }

  /* ------------------------------------------------------------------
     18. Boot
     ------------------------------------------------------------------ */

  function refreshState() {
    var q = 'player_id=' + encodeURIComponent(playerId || '');
    return apiGet('/api/state?' + q, TIMEOUT_SHORT)
      .then(function (data) {
        if (data.state && data.state.player_id) savePlayerId(data.state.player_id);
        if (data.summary) state.summary = data.summary;
        return data;
      });
  }

  function refreshHealth() {
    // Diagnostics only: never blocks the UI, never blocks a mission.
    return apiGet('/api/health', TIMEOUT_SHORT)
      .then(function (h) { state.health = h; return h; })
      .catch(function () { state.health = { engine: null }; return null; });
  }

  function boot() {
    buildSetup();
    wire();

    playerId = loadPlayerId();

    // Show something immediately; never a blank screen.
    show('setup');

    // Health and state run in parallel and are independent: a failed health
    // probe must not stop the app from working with a fallback generator.
    var p1 = refreshHealth().catch(function () { return null; });
    var p2 = refreshState().catch(function (err) {
      // A first-run failure here is worth surfacing, but the app stays usable
      // because POST /api/quest will mint a player id server-side anyway.
      fail(err, 'Loading your progress', function () { location.reload(); });
      return null;
    });

    Promise.all([p1, p2]).then(function () {
      // Resume an interrupted mission — the common case is the player closed
      // the app to go outside.
      var s = loadSession();
      if (s && s.quest && s.quest.id) {
        state.quest = s.quest;
        state.generation = s.generation || null;
        state.startedAt = s.startedAt || null;
        state.plannedMinutes = s.planned || (s.quest.estimated_minutes || 30);
        renderMission();
        renderAway();
        show(state.startedAt ? 'away' : 'mission');
        if (state.startedAt) startTicker();
        announce(state.startedAt
          ? 'Welcome back. Your mission is still running.'
          : 'Welcome back. You have a mission in progress.');
      }

      // Reflect the real engine state in the copy on the setup screen.
      var note = $('gen-note');
      var eng = state.health && state.health.engine;
      if (eng && eng.loaded === false) {
        note.textContent = 'No model is loaded, so missions will come from the built-in ' +
          'template library and photos cannot be AI verified. The app will say so on every result.';
      } else if (eng && eng.loaded === true) {
        note.textContent = 'Model: ' + (eng.model_id || 'unknown') + '. Everything runs on this machine.';
      }
    });
  }

  /* ------------------------------------------------------------------
     19. Service worker
     ------------------------------------------------------------------
     Caches the shell so the app opens with no network. The API is never
     cached: a stale verdict would be a dishonest verdict.
     ------------------------------------------------------------------ */

  if ('serviceWorker' in navigator && location.protocol !== 'file:') {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('sw.js')
        .catch(function () { /* offline shell is a bonus, not a requirement */ });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();