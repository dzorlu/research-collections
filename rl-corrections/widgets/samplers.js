// The samplers widget: section 4's two error terms against the sampler's q, with the model's p_2, the engine's q
// and B under the reader's control. Closed form (section 4), nothing is sampled.
//   p = ((1 - p_2) 5/8, (1 - p_2) 3/8, p_2),  v = p - e_2,  ||v||^2 = (1 - p_2)^2 * 98/64   (0.98 at p_2 = 0.2)
//   ||bias||^2 = ||v||^2 (q - p_2)^2;  Var(x) = ||v||^2 q (1 - q);  error = ||bias||^2 + Var(x) / B.
window.widgets = window.widgets || {};
window.widgets.samplers = function (el) {
  const BLUE = '#2a78d6', ORANGE = '#eb6834', AQUA = '#1baf7a', INK = '#1d1d1b', MUTED = '#5f5e5a', GRID = '#e4e3df';
  const state = { p2: 0.2, q: 0.1, B: 16 };
  const QMAX = 0.5, BMAX = 1024;

  el.innerHTML = `
    <div class="w-controls">
      <label>model's rate for token 2, p₂ <b class="w-p"></b><input class="w-ps" type="range" min="0.05" max="0.5" step="0.01" value="0.2"></label>
      <label>engine's rate for token 2, q <b class="w-q"></b><input class="w-qs" type="range" min="0" max="0.5" step="0.005" value="0.1"></label>
      <label>batch size, B <b class="w-B"></b><input class="w-Bs" type="range" min="0" max="10" value="4"></label>
    </div>
    <div class="w-readout"></div>
    <div class="w-panels">
      <figure><svg class="w-qplot" viewBox="0 0 300 200"></svg><figcaption>error terms against the sampler's q, at this B</figcaption></figure>
      <figure><svg class="w-bplot" viewBox="0 0 300 200"></svg><figcaption>error of each sampler against B</figcaption></figure>
    </div>`;
  const $ = s => el.querySelector(s);
  const svgEl = (tag, attrs, text) => {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const fmt = x => Math.abs(x) >= 0.01 || x === 0 ? x.toFixed(3) : x.toFixed(4);

  // The closed forms.
  const vn2 = p2 => (1 - p2) ** 2 * 98 / 64;
  const stats = (q, p2, B) => {
    const n = vn2(p2), bias2 = n * (q - p2) ** 2, variance = n * q * (1 - q);
    return { bias2, variance, error: bias2 + variance / B };
  };

  // Axes: a frame with a y scale on the left and an x scale at the bottom.
  const frame = (svg, xlo, xhi, ylo, yhi, xlog) => {
    svg.innerHTML = '';
    const L = 44, R = 8, T = 10, Bm = 28, W = 300, H = 200;
    const tx = x => xlog ? Math.log(x) : x;
    const sx = x => L + (W - L - R) * (tx(x) - tx(xlo)) / (tx(xhi) - tx(xlo));
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
  const curve = (svg, f, pts, color) =>
    svg.appendChild(svgEl('path', { d: pts.map(([x, y], i) => `${i ? 'L' : 'M'}${f.sx(x)},${f.sy(y)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': 2, 'clip-path': 'url(#w-qclip)' }));
  const vline = (svg, f, x, label, dy, top) => {
    svg.appendChild(svgEl('line', { x1: f.sx(x), y1: f.T, x2: f.sx(x), y2: f.H - f.Bm, stroke: INK, 'stroke-dasharray': '4 3' }));
    svg.appendChild(svgEl('text', { x: f.sx(x) + 3, y: top ? f.T + 8 : f.H - f.Bm - 4 - dy, 'font-size': 8, fill: INK }, label));
  };

  // Left: the two terms and their sum against q, at the current B. Dashed: the model at p_2 and the engine at q.
  const drawQ = () => {
    const { p2, q, B } = state, svg = $('.w-qplot'), qs = [];
    for (let i = 0; i <= 200; i++) qs.push(QMAX * i / 200);
    const rows = qs.map(x => [x, stats(x, p2, B)]);
    const ymax = Math.max(stats(p2, p2, B).error, stats(q, p2, B).error) * 2.5 || 1;   // the marked points, with room
    const f = frame(svg, 0, QMAX, 0, ymax, false);
    const clip = svgEl('clipPath', { id: 'w-qclip' }); clip.appendChild(svgEl('rect', { x: f.L, y: f.T, width: f.W - f.L - f.R, height: f.H - f.T - f.Bm }));
    svg.appendChild(clip);
    curve(svg, f, rows.map(([x, s]) => [x, s.bias2]), BLUE);
    curve(svg, f, rows.map(([x, s]) => [x, s.variance / B]), ORANGE);
    curve(svg, f, rows.map(([x, s]) => [x, s.error]), AQUA);
    vline(svg, f, p2, 'model', 0); vline(svg, f, q, 'engine', Math.abs(p2 - q) < 0.06 ? 10 : 0);
    for (const [x, s] of [[p2, stats(p2, p2, B)], [q, stats(q, p2, B)]])
      svg.appendChild(svgEl('circle', { cx: f.sx(x), cy: f.sy(s.error), r: 3.5, fill: AQUA, stroke: '#fff' }));
    for (const x of [0, 0.1, 0.2, 0.3, 0.4, 0.5]) tick(svg, f, x, x.toFixed(1));
    svg.appendChild(svgEl('text', { x: f.W - f.R - 4, y: f.T + 2, 'text-anchor': 'end', 'font-size': 8, fill: BLUE }, '— ||bias||²'));
    svg.appendChild(svgEl('text', { x: f.W - f.R - 4, y: f.T + 12, 'text-anchor': 'end', 'font-size': 8, fill: ORANGE }, '— Var(x) / B'));
    svg.appendChild(svgEl('text', { x: f.W - f.R - 4, y: f.T + 22, 'text-anchor': 'end', 'font-size': 8, fill: AQUA }, '— error'));
  };

  // Right: each sampler's error against B. The model's falls as 1/B; the engine's floors at its ||bias||^2.
  const drawB = () => {
    const { p2, q, B } = state, svg = $('.w-bplot'), Bs = [];
    for (let i = 0; i <= 100; i++) Bs.push(BMAX ** (i / 100));
    const m = Bs.map(b => [b, stats(p2, p2, b).error]), e = Bs.map(b => [b, stats(q, p2, b).error]);
    const ymax = Math.max(m[0][1], e[0][1]) * 1.1 || 1;
    const f = frame(svg, 1, BMAX, 0, ymax, true);
    curve(svg, f, m, INK); curve(svg, f, e, MUTED);
    const sm = stats(p2, p2, B), se = stats(q, p2, B);
    const cross = (sm.variance - se.variance) / se.bias2;             // where Var_m / B = bias_e^2 + Var_e / B
    if (cross > 1 && cross < BMAX) {
      svg.appendChild(svgEl('circle', { cx: f.sx(cross), cy: f.sy(stats(p2, p2, cross).error), r: 3.5, fill: '#fff', stroke: INK }));
      svg.appendChild(svgEl('text', { x: f.sx(cross) + 5, y: f.sy(stats(p2, p2, cross).error) - 5, 'font-size': 8, fill: INK }, `cross at B = ${cross.toFixed(1)}`));
    }
    vline(svg, f, B, `B = ${B}`, 0, true);
    svg.appendChild(svgEl('circle', { cx: f.sx(B), cy: f.sy(sm.error), r: 3, fill: INK }));
    svg.appendChild(svgEl('circle', { cx: f.sx(B), cy: f.sy(se.error), r: 3, fill: MUTED }));
    for (const b of [1, 4, 16, 64, 256, 1024]) tick(svg, f, b, String(b));
    svg.appendChild(svgEl('text', { x: f.W - f.R - 4, y: f.T + 2, 'text-anchor': 'end', 'font-size': 8, fill: INK }, '— model: Var(x) / B'));
    svg.appendChild(svgEl('text', { x: f.W - f.R - 4, y: f.T + 12, 'text-anchor': 'end', 'font-size': 8, fill: MUTED }, '— engine: ||bias||² + Var(x) / B'));
  };

  const render = () => {
    const { p2, q, B } = state, sm = stats(p2, p2, B), se = stats(q, p2, B), n = vn2(p2);
    const cross = (sm.variance - se.variance) / se.bias2;
    $('.w-p').textContent = p2.toFixed(2); $('.w-q').textContent = q.toFixed(3); $('.w-B').textContent = B;
    const verdict = se.bias2 === 0 ? 'The engine is the model.'
      : cross <= 1 ? 'The engine loses at every B.'
      : `The engine wins below B = ${cross.toFixed(1)} and loses above it.`;
    $('.w-readout').innerHTML =
      `||v||² = ${n.toFixed(3)}. Model: bias 0, Var(x) = ${fmt(sm.variance)}, error Var(x)/B = <b>${fmt(sm.error)}</b>. ` +
      `Engine: ||bias||² = ${n.toFixed(3)} (${q.toFixed(3)} − ${p2.toFixed(2)})² = ${fmt(se.bias2)}, Var(x)/B = ${fmt(se.variance / B)}, error <b>${fmt(se.error)}</b>. ${verdict}`;
    drawQ(); drawB();
  };
  $('.w-ps').oninput = e => { state.p2 = +e.target.value; render(); };
  $('.w-qs').oninput = e => { state.q = +e.target.value; render(); };
  $('.w-Bs').oninput = e => { state.B = 2 ** +e.target.value; render(); };
  render();
};
