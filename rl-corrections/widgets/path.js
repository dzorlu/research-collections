// The path widget (section 7.1): how often the current model reaches token t's context, relative to the batch.
// If every earlier token's probability changed by the same fraction c, the context of token t is (1 + c)^(t - 1)
// times as common as when the batch was written. The loss counts every token's row at the batch's rate, 1.
window.widgets = window.widgets || {};
window.widgets.path = function (el) {
  const BLUE = '#2a78d6', INK = '#1d1d1b', MUTED = '#5f5e5a', GRID = '#e4e3df';
  const state = { T: 2, c: -20 };
  const PRESETS = [['the toy: 2 tokens, −20%', 2, -20], ['500 tokens, −1%', 500, -1], ['500 tokens, +1%', 500, 1]];
  const lenOf = s => Math.round(2 * 500 ** (s / 100)), sliderOf = T => Math.round(100 * Math.log(T / 2) / Math.log(500));

  el.innerHTML = `
    <div class="w-controls">
      <label>response length, tokens <b class="w-T"></b><input class="w-Ts" type="range" min="0" max="100" value="0"></label>
      <label>change in each earlier token's probability <b class="w-c"></b><input class="w-cs" type="range" min="-20" max="20" step="0.5" value="-20"></label>
      <div class="w-presets">${PRESETS.map((p, i) => `<button data-i="${i}">${p[0]}</button>`).join('')}</div>
    </div>
    <div class="w-readout"></div>
    <div class="w-panels">
      <figure><svg class="w-plot" viewBox="0 0 600 220"></svg><figcaption>how often the current model reaches each token's context, relative to the batch (log scale)</figcaption></figure>
    </div>`;
  const $ = s => el.querySelector(s);
  const svgEl = (tag, attrs, text) => {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const fmt = v => v >= 1e4 || v < 1e-3 ? v.toExponential(1).replace('e+', 'e') : String(Number(v.toPrecision(3)));

  const draw = () => {
    const { T, c } = state, svg = $('.w-plot'), rho = 1 + c / 100;
    svg.innerHTML = '';
    const L = 56, R = 96, Tp = 16, Bm = 34, W = 600, H = 220;
    const end = rho ** (T - 1);
    const lo = Math.log10(Math.min(1, end)), hi = Math.log10(Math.max(1, end));
    const pad = Math.max(0.05, (hi - lo) * 0.12), ylo = lo - pad, yhi = hi + pad;
    const sx = t => L + (W - L - R) * (t - 1) / Math.max(1, T - 1);
    const sy = v => H - Bm - (H - Bm - Tp) * (Math.log10(v) - ylo) / (yhi - ylo);
    // axes
    svg.appendChild(svgEl('line', { x1: L, y1: H - Bm, x2: W - R, y2: H - Bm, stroke: MUTED }));
    svg.appendChild(svgEl('line', { x1: L, y1: Tp, x2: L, y2: H - Bm, stroke: MUTED }));
    for (const t of [...new Set([1, Math.round((T + 1) / 2), T])]) {
      svg.appendChild(svgEl('line', { x1: sx(t), y1: H - Bm, x2: sx(t), y2: H - Bm + 3, stroke: MUTED }));
      svg.appendChild(svgEl('text', { x: sx(t), y: H - Bm + 13, 'text-anchor': 'middle', 'font-size': 9, fill: MUTED }, `token ${t}`));
    }
    for (const v of [...new Set([1, end])]) {
      svg.appendChild(svgEl('line', { x1: L, y1: sy(v), x2: W - R, y2: sy(v), stroke: GRID }));
      svg.appendChild(svgEl('text', { x: L - 5, y: sy(v) + 3, 'text-anchor': 'end', 'font-size': 9, fill: MUTED }, `${fmt(v)}×`));
    }
    // what the loss assumes: every context as common as in the batch
    svg.appendChild(svgEl('line', { x1: L, y1: sy(1), x2: W - R, y2: sy(1), stroke: INK, 'stroke-dasharray': '4 3' }));
    svg.appendChild(svgEl('text', { x: W - R + 5, y: sy(1) + 3, 'font-size': 9, fill: INK }, 'the loss: 1×'));
    // the current model
    const pts = [];
    for (let i = 0; i <= 200; i++) { const t = 1 + (T - 1) * i / 200; pts.push([t, rho ** (t - 1)]); }
    svg.appendChild(svgEl('path', { d: pts.map(([t, v], i) => `${i ? 'L' : 'M'}${sx(t)},${sy(v)}`).join(' '), fill: 'none', stroke: BLUE, 'stroke-width': 2 }));
    svg.appendChild(svgEl('circle', { cx: sx(T), cy: sy(end), r: 4, fill: BLUE, stroke: '#fff', 'stroke-width': 2 }));
    if (Math.abs(sy(end) - sy(1)) > 10) svg.appendChild(svgEl('text', { x: W - R + 5, y: sy(end) + 3, 'font-size': 9, fill: INK }, `actual: ${fmt(end)}×`));
  };

  const render = () => {
    const { T, c } = state, end = (1 + c / 100) ** (T - 1);
    $('.w-T').textContent = T; $('.w-c').textContent = `${c > 0 ? '+' : ''}${c}%`;
    const verdict = c === 0 ? 'the loss counts its row correctly' :
      end < 1 ? `the loss over-counts its row <b>${fmt(1 / end)}×</b>` : `the loss under-counts its row <b>${fmt(end)}×</b>`;
    $('.w-readout').innerHTML = `The context of token ${T} is <b>${fmt(end)}×</b> as common as in the batch; ${verdict}.`;
    draw();
  };
  $('.w-Ts').oninput = e => { state.T = lenOf(+e.target.value); render(); };
  $('.w-cs').oninput = e => { state.c = +e.target.value; render(); };
  el.querySelectorAll('.w-presets button').forEach(b => b.onclick = () => {
    [, state.T, state.c] = PRESETS[+b.dataset.i]; $('.w-Ts').value = sliderOf(state.T); $('.w-cs').value = state.c; render();
  });
  render();
};
