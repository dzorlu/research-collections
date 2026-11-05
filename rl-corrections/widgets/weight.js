// The weight widget: the far-engine trade of section 6 with q, w_2 and B under the reader's control.
// Everything is closed form (section 4 and 6.2), nothing is sampled.
//   row = w_2 v with probability q, else 0;  E[x] = q w_2 v;  ||bias||^2 = ||v||^2 (q w_2 - p_2)^2;
//   Var(x) = w_2^2 ||v||^2 q (1 - q);  error = ||bias||^2 + Var(x) / B.
// A batch of B rows holds k ~ Binomial(B, q) token-2 rows and estimates w_2 k / B v.
window.widgets = window.widgets || {};
window.widgets.weight = function (el) {
  const P2 = 0.2, V2 = -0.8, VN2 = 0.98, G2 = P2 * V2;   // p_2; token 2's coordinate of v; ||v||^2; g's token-2 coordinate
  const BLUE = '#2a78d6', ORANGE = '#eb6834', AQUA = '#1baf7a', INK = '#1d1d1b', MUTED = '#5f5e5a', GRID = '#e4e3df';
  const state = { q: 0.01, w: 1, B: 16 };
  const PRESETS = [['REINFORCE', () => 1], ['TIS, cap 2', () => Math.min(2, P2 / state.q)],
                   ['CISPO, cap 4', () => Math.min(4, P2 / state.q)], ['IS', () => P2 / state.q], ['PPO, zeroed', () => 0]];

  el.innerHTML = `
    <div class="w-controls">
      <label>sampler's rate for token 2, q <b class="w-q"></b><input class="w-qs" type="range" min="0" max="100" value="0"></label>
      <label>weight on the row, w₂ <b class="w-w"></b><input class="w-ws" type="range" min="0" max="40" step="0.1" value="1"></label>
      <label>batch size, B <b class="w-B"></b><input class="w-Bs" type="range" min="0" max="10" value="4"></label>
      <div class="w-presets">${PRESETS.map((p, i) => `<button data-i="${i}">${p[0]}</button>`).join('')}</div>
    </div>
    <div class="w-readout"></div>
    <div class="w-panels">
      <figure><svg class="w-hist" viewBox="0 0 300 200"></svg><figcaption>one batch estimate, token 2's coordinate</figcaption></figure>
      <figure><svg class="w-bars" viewBox="0 0 300 200"></svg><figcaption>error at this B, by weight</figcaption></figure>
      <figure><svg class="w-cap" viewBox="0 0 300 200"></svg><figcaption>error against the cap on w<sub>2</sub></figcaption></figure>
    </div>`;
  const $ = s => el.querySelector(s);
  const svgEl = (tag, attrs, text) => {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const fmt = x => Math.abs(x) >= 0.01 || x === 0 ? x.toFixed(3) : x.toExponential(1);

  // The closed forms.
  const stats = (q, w, B) => {
    const bias2 = VN2 * (q * w - P2) ** 2, variance = w * w * VN2 * q * (1 - q);
    return { bias2, variance, error: bias2 + variance / B };
  };
  const binomial = (B, q) => {                            // P(k) for k = 0..B
    const out = [(1 - q) ** B];
    for (let k = 0; k < B; k++) out.push(out[k] * (B - k) / (k + 1) * q / (1 - q));
    return out;
  };

  // Axes: a frame with a y scale on the left and an x scale at the bottom.
  const frame = (svg, xlo, xhi, ylo, yhi, xlog) => {
    svg.innerHTML = '';
    const L = 44, R = 8, T = 10, Bm = 28, W = 300, H = 200;
    const sx = x => L + (W - L - R) * ((xlog ? Math.log(x) : x) - (xlog ? Math.log(xlo) : xlo)) / ((xlog ? Math.log(xhi) : xhi) - (xlog ? Math.log(xlo) : xlo));
    const sy = y => H - Bm - (H - Bm - T) * (y - ylo) / (yhi - ylo);
    svg.appendChild(svgEl('line', { x1: L, y1: H - Bm, x2: W - R, y2: H - Bm, stroke: MUTED }));
    svg.appendChild(svgEl('line', { x1: L, y1: T, x2: L, y2: H - Bm, stroke: MUTED }));
    for (const t of [ylo, (ylo + yhi) / 2, yhi]) {
      svg.appendChild(svgEl('line', { x1: L, y1: sy(t), x2: W - R, y2: sy(t), stroke: GRID }));
      svg.appendChild(svgEl('text', { x: L - 4, y: sy(t) + 3, 'text-anchor': 'end', 'font-size': 8, fill: MUTED }, fmt(t)));
    }
    return { sx, sy, L, R, T, Bm, W, H };
  };
  const tick = (svg, f, x, label) => {
    svg.appendChild(svgEl('line', { x1: f.sx(x), y1: f.H - f.Bm, x2: f.sx(x), y2: f.H - f.Bm + 3, stroke: MUTED }));
    svg.appendChild(svgEl('text', { x: f.sx(x), y: f.H - f.Bm + 12, 'text-anchor': 'middle', 'font-size': 8, fill: MUTED }, label));
  };

  const drawHist = () => {
    const { q, w, B } = state, svg = $('.w-hist');
    const pk = binomial(B, q), est = pk.map((_, k) => w * k / B * V2);
    const shown = pk.map((p, k) => [est[k], p]).filter(([, p]) => p > 1e-4);
    const xlo = Math.min(G2, ...shown.map(s => s[0])) * 1.15 - 0.01, xhi = 0.02;
    const f = frame(svg, xlo, xhi, 0, 1, false);
    const bw = Math.max(1.5, Math.min(10, 0.8 * Math.abs(f.sx(est[1]) - f.sx(est[0]))));
    for (const [x, p] of shown) svg.appendChild(svgEl('rect', { x: f.sx(x) - bw / 2, y: f.sy(p), width: bw, height: f.sy(0) - f.sy(p), fill: AQUA }));
    svg.appendChild(svgEl('line', { x1: f.sx(G2), y1: f.T, x2: f.sx(G2), y2: f.H - f.Bm, stroke: INK, 'stroke-dasharray': '4 3' }));
    const mean = q * w * V2;
    svg.appendChild(svgEl('path', { d: `M${f.sx(mean) - 5},${f.H - f.Bm + 7} l5,-7 l5,7 z`, fill: INK }));
    svg.appendChild(svgEl('text', { x: f.L + 6, y: f.T + 2, 'font-size': 8, fill: INK }, '- - g = −0.16'));
    svg.appendChild(svgEl('text', { x: f.L + 6, y: f.T + 12, 'font-size': 8, fill: INK }, '▲ mean of the estimate'));
    svg.appendChild(svgEl('text', { x: f.L + 6, y: f.T + 22, 'font-size': 8, fill: MUTED }, 'y: probability of the batch'));
    for (const x of [Math.ceil(xlo * 20) / 20, 0]) tick(svg, f, x, x.toFixed(2));
  };

  const drawBars = () => {
    const { q, w, B } = state, svg = $('.w-bars');
    const rho = P2 / q;
    const bars = [['w = 1', 1], ['cap 2', Math.min(2, rho)], ['cap 4', Math.min(4, rho)], ['IS', rho], ['yours', w]].map(([n, wi]) => [n, wi, stats(q, wi, B)]);
    const ymax = Math.max(...bars.map(b => b[2].error)) * 1.1 || 1;
    const f = frame(svg, 0, bars.length, 0, ymax, false);
    bars.forEach(([name, , s], i) => {
      const x = f.sx(i + 0.2), bw = f.sx(0.8) - f.sx(0.2);
      svg.appendChild(svgEl('rect', { x, y: f.sy(s.bias2), width: bw, height: f.sy(0) - f.sy(s.bias2), fill: BLUE }));
      svg.appendChild(svgEl('rect', { x, y: f.sy(s.error), width: bw, height: f.sy(s.bias2) - f.sy(s.error), fill: ORANGE }));
      svg.appendChild(svgEl('text', { x: x + bw / 2, y: f.sy(s.error) - 3, 'text-anchor': 'middle', 'font-size': 8, fill: INK }, fmt(s.error)));
      svg.appendChild(svgEl('text', { x: x + bw / 2, y: f.H - f.Bm + 12, 'text-anchor': 'middle', 'font-size': 8, fill: name === 'yours' ? INK : MUTED, 'font-weight': name === 'yours' ? 600 : 400 }, name));
    });
    svg.appendChild(svgEl('text', { x: f.L + 6, y: f.T + 2, 'font-size': 8, fill: BLUE }, '■ ||bias||²'));
    svg.appendChild(svgEl('text', { x: f.L + 6, y: f.T + 12, 'font-size': 8, fill: ORANGE }, '■ Var(x) / B'));
  };

  const drawCap = () => {
    const { q, w, B } = state, svg = $('.w-cap');
    const rho = P2 / q, caps = [];
    for (let i = 0; i <= 200; i++) caps.push(Math.exp(Math.log(rho) * i / 200));     // 1 .. rho, log spaced
    const rows = caps.map(c => [c, stats(q, c, B)]);
    const ymax = Math.max(...rows.map(r => r[1].error)) * 1.1 || 1;
    const f = frame(svg, 1, Math.max(rho, 1.01), 0, ymax, true);
    const path = (key, color) => svg.appendChild(svgEl('path', { d: rows.map(([c, s], i) => `${i ? 'L' : 'M'}${f.sx(c)},${f.sy(s[key])}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2 }));
    path('bias2', BLUE); svg.appendChild(svgEl('path', { d: rows.map(([c, s], i) => `${i ? 'L' : 'M'}${f.sx(c)},${f.sy(s.variance / B)}`).join(' '), fill: 'none', stroke: ORANGE, 'stroke-width': 2 }));
    path('error', AQUA);
    const cStar = Math.min(rho, P2 / (q + (1 - q) / B)), best = [cStar, stats(q, cStar, B)];   // d error / d cap = 0
    svg.appendChild(svgEl('circle', { cx: f.sx(best[0]), cy: f.sy(best[1].error), r: 3.5, fill: AQUA, stroke: '#fff' }));
    svg.appendChild(svgEl('text', { x: f.sx(best[0]), y: f.sy(best[1].error) - 7, 'text-anchor': 'middle', 'font-size': 8, fill: INK }, `best cap ${best[0].toFixed(1)}`));
    if (w >= 1 && w <= rho) {
      svg.appendChild(svgEl('line', { x1: f.sx(w), y1: f.T, x2: f.sx(w), y2: f.H - f.Bm, stroke: INK, 'stroke-dasharray': '2 3' }));
      svg.appendChild(svgEl('text', { x: f.sx(w) + 3, y: f.H - f.Bm - 4, 'font-size': 8, fill: INK }, 'yours'));
    }
    for (const c of [1, 2, 4, 10, 20, 40]) if (c <= rho) tick(svg, f, c, String(c));
    svg.appendChild(svgEl('text', { x: f.L + 6, y: f.T + 2, 'font-size': 8, fill: BLUE }, '— ||bias||²'));
    svg.appendChild(svgEl('text', { x: f.L + 6, y: f.T + 12, 'font-size': 8, fill: ORANGE }, '— Var(x) / B'));
    svg.appendChild(svgEl('text', { x: f.L + 6, y: f.T + 22, 'font-size': 8, fill: AQUA }, '— error'));
  };

  const render = () => {
    const { q, w, B } = state, s = stats(q, w, B), rho = P2 / q;
    $('.w-q').textContent = q.toFixed(3); $('.w-w').textContent = w.toFixed(1); $('.w-B').textContent = B;
    $('.w-readout').innerHTML = `ρ = ${rho.toFixed(1)}, w₂ = ${w.toFixed(1)}. error <b>${fmt(s.error)}</b> = bias² ${fmt(s.bias2)} + Var/B ${fmt(s.variance / B)}.`;
    drawHist(); drawBars(); drawCap();
  };
  $('.w-qs').oninput = e => { state.q = 0.005 * 100 ** (e.target.value / 100); render(); };
  $('.w-ws').oninput = e => { state.w = +e.target.value; render(); };
  $('.w-Bs').oninput = e => { state.B = 2 ** +e.target.value; render(); };
  el.querySelectorAll('.w-presets button').forEach(b => b.onclick = () => {
    state.w = Math.round(PRESETS[+b.dataset.i][1]() * 10) / 10; $('.w-ws').value = state.w; render();
  });
  $('.w-qs').value = Math.round(100 * Math.log(state.q / 0.005) / Math.log(100));
  render();
};
