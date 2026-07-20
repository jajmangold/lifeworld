// Deterministic documentary motion-graphics renderer: JSON spec -> per-frame vintage SVG -> resvg PNG.
// Engine: deterministic SVG -> @resvg/resvg-js. No browser. render(t) is a pure function of time (scrubbable,
// parallel-safe). The archival GRAIN/vignette unifier is applied downstream in ffmpeg (media.graphic_segment).
//   usage: node render.js <spec.json>
import fs from "node:fs";
import os from "node:os";
import { Worker, isMainThread, workerData } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { Resvg } from "@resvg/resvg-js";

// spec comes from argv (main) or workerData (worker). Frames are a pure function of t, so we
// split the frame range across worker threads — deterministic + trivially parallel.
const spec = isMainThread ? JSON.parse(fs.readFileSync(process.argv[2], "utf8")) : workerData.spec;
const W = spec.W ?? 1920, H = spec.H ?? 1080, FPS = spec.fps ?? 30;
const DUR = spec.duration ?? 12;
const N = Math.max(1, Math.round(DUR * FPS));
const OUT = spec.out_dir;
fs.mkdirSync(OUT, { recursive: true });

// palette (agent research: cream/parchment, ink brown-black, faded water, route red)
const C = {
  paper: "#efe4cb", paper2: "#e4d4ad", ink: "#3a3226", inkSoft: "#5a4d38",
  water: "#b7c8c0", road: "#8a7857", route: "#7a3323", mark: "#6e2f22", umber: "#54372a",
  cream: "#efe6d2", edge: "#4a3a24",   // route/mark muted to oxblood — bright red reads as modern UI accent
};
const SERIF = "DejaVu Serif";
const TYPE = "DejaVu Sans Mono";

// Source-controlled procedural textures avoid unproven archived bitmap assets.
function textureURI(background, mark, seed) {
  let flecks = "";
  for (let i = 0; i < 180; i++) {
    const x = Math.abs(Math.sin((i + seed) * 12.9898) * 43758.5453) % 100;
    const y = Math.abs(Math.sin((i + seed) * 78.233) * 12345.6789) % 100;
    const r = 0.08 + (i % 5) * 0.035;
    const opacity = 0.025 + (i % 7) * 0.008;
    flecks += `<circle cx="${x.toFixed(2)}%" cy="${y.toFixed(2)}%" r="${r.toFixed(2)}" fill="${mark}" opacity="${opacity.toFixed(3)}"/>`;
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="100%" height="100%" fill="${background}"/>${flecks}</svg>`;
  return "data:image/svg+xml;base64," + Buffer.from(svg).toString("base64");
}
const PARCH = textureURI("#efe4cb", "#6b5437", 11);
const CORK = textureURI("#9a7048", "#3f291b", 29);
// load an image file path -> base64 data URI (cached), so evidence cards can hold real photos
const _imgCache = {};
function imgURI(p) {
  if (!p) return "";
  if (p in _imgCache) return _imgCache[p];
  try {
    const ext = (p.split(".").pop() || "png").toLowerCase();
    const mime = (ext === "jpg" || ext === "jpeg") ? "image/jpeg" : (ext === "gif") ? "image/gif" : "image/png";
    _imgCache[p] = `data:${mime};base64,` + fs.readFileSync(p).toString("base64");
  } catch (e) { _imgCache[p] = ""; }
  return _imgCache[p];
}
// deterministic pseudo-random (no Math.random -> frames stay identical across threads)
const rnd = (i) => { const x = Math.sin(i * 12.9898) * 43758.5453; return x - Math.floor(x); };

// ── easing ─────────────────────────────────────────────
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const easeInOutCubic = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const easeOutCubic = (x) => 1 - Math.pow(1 - x, 3);
const easeOutBack = (x) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// ── city projection: linear over the points' bbox (good at city scale) ──
function makeProjection(points) {
  const lats = points.map(p => p.lat), lngs = points.map(p => p.lng);
  let [minLat, maxLat] = [Math.min(...lats), Math.max(...lats)];
  let [minLng, maxLng] = [Math.min(...lngs), Math.max(...lngs)];
  // pad the bbox so pins aren't on the edge; keep aspect roughly framed
  const padLat = Math.max((maxLat - minLat) * 0.35, 0.01), padLng = Math.max((maxLng - minLng) * 0.35, 0.01);
  minLat -= padLat; maxLat += padLat; minLng -= padLng; maxLng += padLng;
  const m = 0.10; // frame inset
  const x = (lng) => (m + (1 - 2 * m) * (lng - minLng) / (maxLng - minLng)) * W;
  const y = (lat) => (m + (1 - 2 * m) * (maxLat - lat) / (maxLat - minLat)) * H; // north=up
  return { x, y };
}

function segLen(pts) { let L = 0; for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); return L; }

// ── MAP scene ──────────────────────────────────────────
function renderMap(t) {
  const proj = makeProjection(spec.points);
  const P = spec.points.map(p => ({ ...p, px: proj.x(p.lng), py: proj.y(p.lat) }));
  const cx = P.reduce((s, p) => s + p.px, 0) / P.length, cy = P.reduce((s, p) => s + p.py, 0) / P.length;
  // slow ease-in push toward the cluster
  const zoom = 1 + 0.14 * easeInOutCubic(clamp01(t / DUR));
  const tx = (1 - zoom) * cx, ty = (1 - zoom) * cy;

  let g = "";
  // route draw-on (after ~35% of clip), between points in order
  if (spec.route && P.length > 1) {
    const pts = P.map(p => [p.px, p.py]);
    const L = segLen(pts);
    const rp = easeInOutCubic(clamp01((t - DUR * 0.35) / (DUR * 0.4)));
    const d = "M " + pts.map(p => p.join(" ")).join(" L ");
    g += `<path d="${d}" fill="none" stroke="${C.route}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9" stroke-dasharray="${L}" stroke-dashoffset="${L * (1 - rp)}"/>`;
  }
  // markers: staggered drop with overshoot + one ripple + label
  for (let i = 0; i < P.length; i++) {
    const p = P[i];
    const at = p.at ?? (DUR * (0.12 + 0.7 * i / Math.max(1, P.length - 1)));
    const k = clamp01((t - at) / 0.6);
    if (k <= 0) continue;
    const s = easeOutBack(k);                       // mark scale overshoot
    const rr = 15;
    // ripple (single, fades)
    const rk = clamp01((t - at) / 1.4);
    if (rk > 0 && rk < 1) g += `<circle cx="${p.px}" cy="${p.py}" r="${8 + rk * 46}" fill="none" stroke="${C.mark}" stroke-width="1.6" opacity="${(1 - rk) * 0.4}"/>`;
    // hand-drawn ink location mark: two slightly offset rough rings + center dot (period crime-map)
    const jx = (rnd(i) - 0.5) * 2.4, jy = (rnd(i + 7) - 0.5) * 2.4;
    g += `<g transform="translate(${p.px} ${p.py}) scale(${s})">`
      + `<circle r="${rr}" fill="none" stroke="${C.mark}" stroke-width="2.8" opacity="0.92"/>`
      + `<circle cx="${jx}" cy="${jy}" r="${rr * 0.92}" fill="none" stroke="${C.mark}" stroke-width="1.5" opacity="0.5"/>`
      + `<circle r="${rr * 0.3}" fill="${C.mark}"/></g>`;
    // label reveals after the mark settles
    const lk = clamp01((t - at - 0.35) / 0.5);
    if (lk > 0 && p.name) {
      g += `<text x="${p.px}" y="${p.labelBelow ? p.py + rr + 36 : p.py - rr - 14}" font-family="${SERIF}" font-style="italic" font-size="31" fill="${C.ink}" text-anchor="middle" opacity="${lk}">${esc(p.name)}</text>`;
    }
  }

  // REAL city plan: OSM streets + water for the bbox (spec.streets from geo.streets), drawn as
  // engraved ink lines by tier (major roads darker/thicker). Falls back to a jittered grid if absent.
  let grid = "";
  if (spec.streets && spec.streets.length) {
    const proj2 = (c) => c.map(([lng, lat]) => `${proj.x(lng).toFixed(1)} ${proj.y(lat).toFixed(1)}`).join(" L ");
    for (const s of spec.streets) {
      if (s.coords.length < 2) continue;
      const d = "M " + proj2(s.coords);
      if (s.water) grid += `<path d="${d}" fill="none" stroke="${C.water}" stroke-width="10" opacity="0.6" stroke-linejoin="round"/>`;
      else {
        const w = s.tier === 0 ? 2.6 : s.tier === 1 ? 1.4 : 0.8;
        const op = s.tier === 0 ? 0.5 : s.tier === 1 ? 0.32 : 0.2;   // engraved ink, clearly legible
        grid += `<path d="${d}" fill="none" stroke="${C.inkSoft}" stroke-width="${w}" opacity="${op}"/>`;
      }
    }
  } else {                                               // fallback: hand-jittered grid
    const NV = 22, NH = 13;
    for (let gx = 1; gx < NV; gx++) { const x = gx / NV * W, j = (rnd(gx) - 0.5) * 10; grid += `<path d="M ${x + j} 40 Q ${x - j} ${H / 2} ${x + j * 0.6} ${H - 40}" fill="none" stroke="${C.road}" stroke-width="0.8" opacity="0.08"/>`; }
    for (let gy = 1; gy < NH; gy++) { const y = gy / NH * H, j = (rnd(gy + 99) - 0.5) * 10; grid += `<path d="M 40 ${y + j} Q ${W / 2} ${y - j} ${W - 40} ${y + j * 0.6}" fill="none" stroke="${C.road}" stroke-width="0.8" opacity="0.08"/>`; }
  }

  // cartouche title (top-left) + compass rose (top-right)
  const titleIn = easeOutCubic(clamp01(t / 0.8));
  // period masthead: title over a double ink rule with end ornaments (no UI box)
  const tw = 30 + spec.title.length * 30;
  const cartouche = spec.title ? `<g opacity="${titleIn}">`
    + `<text x="90" y="112" font-family="${SERIF}" font-weight="bold" font-size="52" fill="${C.ink}">${esc(spec.title)}</text>`
    + `<line x1="90" y1="130" x2="${90 + tw}" y2="130" stroke="${C.ink}" stroke-width="2.4"/>`
    + `<line x1="90" y1="136" x2="${90 + tw}" y2="136" stroke="${C.inkSoft}" stroke-width="1"/>`
    + `<path d="M ${90 + tw} 133 l 10 -5 l 0 10 z" fill="${C.ink}"/>`
    + (spec.subtitle ? `<text x="90" y="168" font-family="${SERIF}" font-style="italic" font-size="27" fill="${C.umber}">${esc(spec.subtitle)}</text>` : "")
    + `</g>` : "";
  const compass = `<g transform="translate(${W - 130} 150)" opacity="${0.5 * titleIn}">`
    + `<circle r="46" fill="none" stroke="${C.ink}" stroke-width="1.5"/><circle r="38" fill="none" stroke="${C.inkSoft}" stroke-width="0.8"/>`
    + `<path d="M 0 -44 L 8 0 L 0 44 L -8 0 Z" fill="${C.ink}"/><path d="M -44 0 L 0 -8 L 44 0 L 0 8 Z" fill="none" stroke="${C.ink}" stroke-width="1"/>`
    + `<text x="0" y="-52" font-family="${SERIF}" font-size="20" fill="${C.ink}" text-anchor="middle">N</text></g>`;

  // double cartographic border
  const border = `<rect x="24" y="24" width="${W - 48}" height="${H - 48}" fill="none" stroke="${C.ink}" stroke-width="3"/>`
    + `<rect x="34" y="34" width="${W - 68}" height="${H - 68}" fill="none" stroke="${C.inkSoft}" stroke-width="1.2"/>`;

  // vignette via radial gradient (grain added later in ffmpeg)
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
    <defs><radialGradient id="vig" cx="50%" cy="50%" r="72%">
      <stop offset="55%" stop-color="${C.edge}" stop-opacity="0"/><stop offset="100%" stop-color="${C.edge}" stop-opacity="0.5"/>
    </radialGradient></defs>
    <rect width="${W}" height="${H}" fill="${C.paper}"/>
    <image href="${PARCH}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="xMidYMid slice" opacity="0.9"/>
    <g transform="translate(${tx} ${ty}) scale(${zoom})">${grid}${g}</g>
    ${cartouche}${compass}${border}
    <rect width="${W}" height="${H}" fill="url(#vig)"/>
  </svg>`;
}

// ── TIMELINE scene ─────────────────────────────────────
function renderTimeline(t) {
  const E = spec.events || [];
  const n = E.length;
  const spineY = H * 0.56, x0 = W * 0.11, x1 = W * 0.89;
  const xOf = (i) => x0 + (x1 - x0) * (n <= 1 ? 0.5 : i / (n - 1));
  // spine draws on first — a thick, slightly-wobbling INK line (hand-drawn, not a vector rule)
  const spineK = easeInOutCubic(clamp01(t / 1.2));
  const endX = x0 + (x1 - x0) * spineK;
  // faint ruled ledger lines behind everything (period logbook paper, fills the empty field)
  let g = "";
  for (let ly = 1; ly < 14; ly++) g += `<line x1="70" y1="${ly / 14 * H}" x2="${W - 70}" y2="${ly / 14 * H}" stroke="${C.inkSoft}" stroke-width="0.8" opacity="0.07"/>`;
  // spine: thick, wobbling INK line with varying weight (hand-drawn, not a vector rule)
  let sp = `M ${x0} ${spineY}`;
  for (let s = 1; s <= 44; s++) { const px = x0 + (endX - x0) * s / 44, jy = (rnd(s * 3.1) - 0.5) * 5.5; sp += ` L ${px.toFixed(1)} ${(spineY + jy).toFixed(1)}`; }
  g += `<path d="${sp}" fill="none" stroke="${C.ink}" stroke-width="3" stroke-linecap="round" opacity="0.5"/>`
    + `<path d="${sp}" fill="none" stroke="${C.ink}" stroke-width="5.2" stroke-linecap="round" opacity="0.85"/>`;
  // playhead position = latest revealed node
  for (let i = 0; i < n; i++) {
    const e = E[i];
    const at = e.at ?? (DUR * (0.14 + 0.78 * i / Math.max(1, n - 1)));
    const k = clamp01((t - at) / 0.5);
    if (k <= 0) continue;
    const x = xOf(i), up = i % 2 === 0;                 // alternate above/below
    const active = clamp01((t - at) / 1.3) < 1 && t >= at;
    const dim = t > at + 1.6 ? 0.45 : 1;                // past nodes dim
    const s = easeOutBack(k);
    // node
    g += `<circle cx="${x}" cy="${spineY}" r="${8 * s}" fill="${C.mark}" stroke="${C.ink}" stroke-width="1.5" opacity="${dim}"/>`;
    // connector + card
    const cy = up ? spineY - 60 : spineY + 60, cardY = up ? cy - 74 : cy;
    const lk = clamp01((t - at - 0.15) / 0.45);
    if (lk > 0) {
      g += `<line x1="${x}" y1="${spineY}" x2="${x}" y2="${cy}" stroke="${C.inkSoft}" stroke-width="1.2" opacity="${dim * lk}"/>`;
      g += `<g opacity="${dim * lk}">`
        + `<text x="${x}" y="${up ? cy - 34 : cy + 30}" font-family="${SERIF}" font-weight="bold" font-size="30" fill="${C.ink}" text-anchor="middle">${esc(e.date || "")}</text>`
        + `<text x="${x}" y="${up ? cy - 4 : cy + 60}" font-family="${SERIF}" font-style="italic" font-size="25" fill="${C.ink}" text-anchor="middle">${esc(e.label || "")}</text></g>`;
    }
  }
  const titleIn = easeOutCubic(clamp01(t / 0.8));
  const title = spec.title ? `<text x="${W / 2}" y="120" font-family="${SERIF}" font-weight="bold" font-size="46" fill="${C.ink}" text-anchor="middle" letter-spacing="2" opacity="${titleIn}">${esc(spec.title)}</text>` : "";
  const border = `<rect x="24" y="24" width="${W - 48}" height="${H - 48}" fill="none" stroke="${C.ink}" stroke-width="3"/><rect x="34" y="34" width="${W - 68}" height="${H - 68}" fill="none" stroke="${C.inkSoft}" stroke-width="1.2"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
    <rect width="${W}" height="${H}" fill="${C.paper}"/>
    <image href="${PARCH}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="xMidYMid slice" opacity="0.9"/>
    ${title}${g}${border}
    <rect width="${W}" height="${H}" fill="url(#vig0)"/>
    <defs><radialGradient id="vig0" cx="50%" cy="50%" r="72%"><stop offset="55%" stop-color="${C.edge}" stop-opacity="0"/><stop offset="100%" stop-color="${C.edge}" stop-opacity="0.5"/></radialGradient></defs>
  </svg>`;
}

// ── EVIDENCE BOARD scene ───────────────────────────────
function renderEvidence(t) {
  const nodes = spec.nodes || [];
  const links = spec.links || [];
  const n = nodes.length;
  // deterministic loose layout: spread across a padded field, jittered (auto unless x/y given)
  const cols = Math.min(n, Math.ceil(Math.sqrt(n * 1.6)));
  const pos = nodes.map((nd, i) => {
    if (typeof nd.x === "number") return [nd.x * W, nd.y * H];
    const col = i % cols, row = Math.floor(i / cols), rows = Math.ceil(n / cols);
    const x = W * (0.16 + 0.68 * (cols === 1 ? 0.5 : col / (cols - 1))) + (rnd(i) - 0.5) * 90;
    const y = H * (0.26 + 0.5 * (rows <= 1 ? 0.5 : row / (rows - 1))) + (rnd(i + 5) - 0.5) * 70;
    return [x, y];
  });
  const cw = 250, ch = 208;
  // the red string is pinned pin-to-pin — anchor at each card's PUSH-PIN (top-centre), rotated
  // with the card, not at the card centre (which made strings sprout from the card faces).
  const rotOf = (i) => (rnd(i + 3) - 0.5) * 14 * Math.PI / 180;
  const pinPos = (i) => {
    const rot = rotOf(i), lx = 0, ly = -ch / 2 + 6;
    return [pos[i][0] + (lx * Math.cos(rot) - ly * Math.sin(rot)),
            pos[i][1] + (lx * Math.sin(rot) + ly * Math.cos(rot))];
  };
  // background clutter — scattered pinned scraps/photos + loose pins so the board reads as a
  // real, busy investigation wall (not a sparse diagram). Deterministic, drawn on the cork.
  let clutter = "";
  const NC = 12;
  for (let c = 0; c < NC; c++) {
    const cx = W * (0.05 + 0.90 * rnd(c * 7 + 1)), cy = H * (0.17 + 0.76 * rnd(c * 7 + 2));
    const rr = (rnd(c * 7 + 3) - 0.5) * 26, typ = rnd(c * 7 + 4);
    clutter += `<g transform="translate(${cx.toFixed(1)} ${cy.toFixed(1)}) rotate(${rr.toFixed(1)})" opacity="0.72">`;
    if (typ < 0.4) {                       // small blank photo
      const w = 74 + 46 * rnd(c + 9), h = w * 0.82;
      clutter += `<rect x="${-w / 2 + 4}" y="${-h / 2 + 5}" width="${w}" height="${h}" fill="#000" opacity="0.22"/>`
        + `<rect x="${-w / 2}" y="${-h / 2}" width="${w}" height="${h}" fill="#e7dcc4" stroke="#3a3226" stroke-width="1"/>`
        + `<rect x="${-w / 2 + 5}" y="${-h / 2 + 5}" width="${w - 10}" height="${h - 10}" fill="#bba483"/>`;
    } else if (typ < 0.78) {               // torn note with ruled lines
      const w = 84 + 52 * rnd(c + 2), h = 56 + 30 * rnd(c + 4);
      clutter += `<rect x="${-w / 2 + 3}" y="${-h / 2 + 4}" width="${w}" height="${h}" fill="#000" opacity="0.18"/>`
        + `<rect x="${-w / 2}" y="${-h / 2}" width="${w}" height="${h}" fill="#efe7d2" stroke="#cbbfa0" stroke-width="0.5"/>`
        + `<line x1="${-w / 2 + 8}" y1="-5" x2="${w / 2 - 8}" y2="-5" stroke="#9a8d70" stroke-width="1.3"/>`
        + `<line x1="${-w / 2 + 8}" y1="7" x2="${w / 2 - 20}" y2="7" stroke="#9a8d70" stroke-width="1.3"/>`;
    }
    const pc = rnd(c + 30) < 0.4 ? "#b0201d" : "#8c8c8c";   // some red pins, some brass/steel
    clutter += `<circle cx="0" cy="-13" r="6" fill="${pc}" stroke="#555" stroke-width="0.7"/>`
      + `<circle cx="-2" cy="-15" r="2" fill="#fff" opacity="0.6"/></g>`;
  }
  let g = "";
  // strings first (behind cards): sagging yarn drawn on when the link is revealed
  for (let li = 0; li < links.length; li++) {
    const [a, b] = links[li];
    if (a >= n || b >= n) continue;
    const at = links[li][2]?.at ?? (DUR * (0.5 + 0.4 * li / Math.max(1, links.length - 1)));
    const k = easeInOutCubic(clamp01((t - at) / 0.7));
    if (k <= 0) continue;
    const [x1, y1] = pinPos(a), [x2, y2] = pinPos(b);
    const dist = Math.hypot(x2 - x1, y2 - y1);
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 + (0.13 * dist + 16);
    // perpendicular unit vector, to waver the string off the ideal curve like real yarn
    const nx = -(y2 - y1) / (dist || 1), ny = (x2 - x1) / (dist || 1);
    // approximate the quadratic with a partial draw via sampling to (k), with fibre waver
    let d = `M ${x1.toFixed(1)} ${y1.toFixed(1)}`;
    const S = 40, end = Math.max(1, Math.round(S * k));
    for (let s = 1; s <= end; s++) {
      const u = s / S;
      let qx = (1 - u) ** 2 * x1 + 2 * (1 - u) * u * mx + u * u * x2;
      let qy = (1 - u) ** 2 * y1 + 2 * (1 - u) * u * my + u * u * y2;
      const taper = Math.sin(Math.PI * u);                 // 0 at pinned ends, max mid-span
      const w = (Math.sin(u * 17 + li * 2.3) * 1.6 + (rnd(li * 50 + s) - 0.5) * 1.5) * taper;
      qx += nx * w; qy += ny * w;
      d += ` L ${qx.toFixed(1)} ${qy.toFixed(1)}`;
    }
    // three strokes = soft shadow + body + a thin uneven highlight, so it reads as round yarn
    g += `<path d="${d}" fill="none" stroke="#4a0c0c" stroke-width="5" opacity="0.5" stroke-linecap="round"/>`
      + `<path d="${d}" fill="none" stroke="#8b1a1a" stroke-width="${(2.6 + 0.5 * Math.sin(li)).toFixed(2)}" opacity="0.92" stroke-linecap="round"/>`
      + `<path d="${d}" fill="none" stroke="#b83a34" stroke-width="1" opacity="0.5" stroke-linecap="round"/>`;
  }
  // cards on top
  for (let i = 0; i < n; i++) {
    const nd = nodes[i];
    const at = nd.at ?? (DUR * (0.1 + 0.5 * i / Math.max(1, n - 1)));
    const k = clamp01((t - at) / 0.5);
    if (k <= 0) continue;
    const s = 0.9 + 0.1 * easeOutBack(k), rot = (rnd(i + 3) - 0.5) * 14;   // more obvious tilt
    const [x, y] = pos[i];
    const wx = -cw / 2 + 12, wy = -ch / 2 + 12, ww = cw - 24, wh = ch - 60;
    const photo = imgURI(nd.img);
    g += `<g transform="translate(${x} ${y}) rotate(${rot}) scale(${s})" opacity="${clamp01(k * 1.4)}">`
      + `<rect x="${-cw / 2 + 6}" y="${-ch / 2 + 8}" width="${cw}" height="${ch}" fill="#000" opacity="0.28"/>`   // drop shadow = depth
      + `<rect x="${-cw / 2}" y="${-ch / 2}" width="${cw}" height="${ch}" fill="#efe6d0" stroke="#3a3226" stroke-width="1.5"/>`
      // photo well — a real period image if the node carries one, else an empty photographic tone
      + (photo
        ? `<image href="${photo}" x="${wx}" y="${wy}" width="${ww}" height="${wh}" preserveAspectRatio="xMidYMid slice"/>`
          + `<rect x="${wx}" y="${wy}" width="${ww}" height="${wh}" fill="none" stroke="#2a231a" stroke-width="1" opacity="0.5"/>`
        : `<rect x="${wx}" y="${wy}" width="${ww}" height="${wh}" fill="#cdbb96"/>`)
      + (nd.label ? `<text x="0" y="${ch / 2 - 20}" font-family="${TYPE}" font-size="22" fill="${C.ink}" text-anchor="middle">${esc(nd.label)}</text>` : "")
      + `<circle cx="0" cy="${-ch / 2 + 6}" r="9" fill="#b0201d" stroke="#7a1512" stroke-width="1"/><circle cx="-3" cy="${-ch / 2 + 3}" r="3" fill="#e88" opacity="0.8"/>`   // push-pin
      + `</g>`;
  }
  const titleIn = easeOutCubic(clamp01(t / 0.8));
  const title = spec.title ? `<text x="${W / 2}" y="96" font-family="${SERIF}" font-weight="bold" font-size="46" fill="#efe0c0" text-anchor="middle" opacity="${titleIn}">${esc(spec.title)}</text>` : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
    <image href="${CORK}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="xMidYMid slice"/>
    ${clutter}${title}${g}
    <rect width="${W}" height="${H}" fill="url(#vigE)"/>
    <defs><radialGradient id="vigE" cx="50%" cy="50%" r="72%"><stop offset="52%" stop-color="#1c1206" stop-opacity="0"/><stop offset="100%" stop-color="#1c1206" stop-opacity="0.6"/></radialGradient></defs>
  </svg>`;
}

// ── STAT card scene ────────────────────────────────────
function renderStat(t) {
  const val = spec.value ?? "";
  const isNum = /^\d[\d,]*$/.test(String(val));
  const shown = isNum ? Math.round(parseInt(String(val).replace(/,/g, "")) * easeOutCubic(clamp01((t - 0.4) / 1.6))).toLocaleString() : (clamp01((t - 0.4) / 0.6) > 0 ? val : "");
  const inK = easeOutCubic(clamp01((t - 0.3) / 0.7));
  const y0 = H * 0.5;
  // dominant value (~40% of frame height) with stroke weight. Size adapts to length so a
  // short number is huge but a WORD (e.g. "NONE") still fits — a lone "0" reads as a ring,
  // so zero-stats are better phrased as a word.
  const fs = isNum ? 320 : Math.min(300, Math.floor(1150 / Math.max(3, String(shown).length)));
  const g = `<g opacity="${inK}">`
    + `<line x1="${W * 0.30}" y1="${y0 - 210}" x2="${W * 0.70}" y2="${y0 - 210}" stroke="${C.ink}" stroke-width="2"/>`
    + `<text x="${W / 2}" y="${y0 + fs * 0.34}" font-family="${SERIF}" font-weight="bold" font-size="${fs}" fill="${C.ink}" stroke="${C.ink}" stroke-width="3" paint-order="stroke" text-anchor="middle">${esc(shown)}</text>`
    + (spec.label ? `<text x="${W / 2}" y="${y0 + 165}" font-family="${SERIF}" font-style="italic" font-size="44" fill="${C.umber}" text-anchor="middle">${esc(spec.label)}</text>` : "")
    + `<line x1="${W * 0.30}" y1="${y0 + 205}" x2="${W * 0.70}" y2="${y0 + 205}" stroke="${C.ink}" stroke-width="2"/></g>`;
  const border = `<rect x="24" y="24" width="${W - 48}" height="${H - 48}" fill="none" stroke="${C.ink}" stroke-width="3"/><rect x="34" y="34" width="${W - 68}" height="${H - 68}" fill="none" stroke="${C.inkSoft}" stroke-width="1.2"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
    <rect width="${W}" height="${H}" fill="${C.paper}"/>
    <image href="${PARCH}" x="0" y="0" width="${W}" height="${H}" preserveAspectRatio="xMidYMid slice" opacity="0.9"/>
    ${g}${border}
    <rect width="${W}" height="${H}" fill="url(#vigS)"/>
    <defs><radialGradient id="vigS" cx="50%" cy="50%" r="72%"><stop offset="55%" stop-color="${C.edge}" stop-opacity="0"/><stop offset="100%" stop-color="${C.edge}" stop-opacity="0.5"/></radialGradient></defs>
  </svg>`;
}

const RENDERERS = { map: renderMap, timeline: renderTimeline, evidence: renderEvidence, stat: renderStat };
const renderFn = RENDERERS[spec.type];
if (!renderFn) { console.error("unknown type " + spec.type); process.exit(2); }

const resvgOpts = { fitTo: { mode: "width", value: W }, font: { fontDirs: ["/usr/share/fonts"], loadSystemFonts: true, defaultFontFamily: "DejaVu Serif" } };

function renderRange(a, b) {
  for (let i = a; i < b; i++) {
    const png = new Resvg(renderFn(i / FPS), resvgOpts).render().asPng();
    fs.writeFileSync(`${OUT}/f${String(i).padStart(5, "0")}.png`, png);
  }
}

if (!isMainThread) {                       // worker: render our slice
  renderRange(workerData.a, workerData.b);
} else {
  const NW = Math.max(1, Math.min(os.cpus().length - 1, N, 32));
  const per = Math.ceil(N / NW);
  const here = fileURLToPath(import.meta.url);
  const workers = [];
  for (let w = 0; w < NW; w++) {
    const a = w * per, b = Math.min(N, a + per);
    if (a >= b) break;
    workers.push(new Promise((res, rej) => {
      const wk = new Worker(here, { workerData: { spec, a, b } });
      wk.on("exit", (c) => (c === 0 ? res() : rej(new Error("worker exit " + c))));
      wk.on("error", rej);
    }));
  }
  await Promise.all(workers);
  console.log(`DOCGFX_OK ${spec.type} ${N} frames -> ${OUT} (${workers.length} workers)`);
}
