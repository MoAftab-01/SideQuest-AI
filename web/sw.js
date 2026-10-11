/* =====================================================================
   SideQuest service worker
   ---------------------------------------------------------------------
   Caches the app shell so the game opens with no network at all, which
   matters here: the whole premise is that the phone goes outside, where
   signal is often bad.

   Deliberate omission: /api/* is NEVER cached. A cached verdict would be
   a fabricated verdict, and this project does not tell people that a
   machine checked their photo when it did not. Offline, the app shows an
   explicit "you are offline" error instead of a stale result.
   ===================================================================== */

'use strict';

var VERSION = 'sidequest-v2';
var SHELL = [
  './',
  './index.html',
  './app.js',
  './styles.css',
  './manifest.webmanifest'
];

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(VERSION).then(function (cache) {
      // addAll is atomic: if one file 404s the install fails loudly rather
      // than leaving a half-cached shell that breaks on first offline load.
      return cache.addAll(SHELL);
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) {
        return k === VERSION ? null : caches.delete(k);
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener('fetch', function (event) {
  var req = event.request;

  if (req.method !== 'GET') return;

  var url;
  try { url = new URL(req.url); } catch (e) { return; }

  if (url.origin !== self.location.origin) return;   // no cross-origin caching

  // The API is never served from cache. See the header note.
  if (url.pathname.indexOf('/api/') === 0) return;

  // Navigations: network first so a deployed update is picked up, falling
  // back to the cached shell when the player is genuinely offline.
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req).then(function (res) {
        var copy = res.clone();
        caches.open(VERSION).then(function (c) { c.put(req, copy); });
        return res;
      }).catch(function () {
        return caches.match('./index.html').then(function (hit) {
          return hit || new Response(
            'SideQuest is offline and the app shell is not cached yet. ' +
            'Reconnect once and it will be available offline from then on.',
            { status: 503, headers: { 'Content-Type': 'text/plain' } }
          );
        });
      })
    );
    return;
  }

  // Static shell assets: cache first, they are versioned by VERSION.
  event.respondWith(
    caches.match(req).then(function (hit) {
      if (hit) return hit;
      return fetch(req).then(function (res) {
        if (res && res.ok && res.type === 'basic') {
          var copy = res.clone();
          caches.open(VERSION).then(function (c) { c.put(req, copy); });
        }
        return res;
      });
    })
  );
});