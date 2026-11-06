// The batches widget: where one batch of 16 lands under REINFORCE, TIS (cap 2) and IS, far engine (q = 0.01), b = 0.
// The estimate is w_2 k / 16 v with k ~ Binomial(16, 0.01) token-2 rows; drawn as a multiple of g = 0.2 v.
window.widgets = window.widgets || {};
window.widgets.batches = function (el) {
  const P2 = 0.2, Q = 0.01, B = 16, RHO = P2 / Q;
  const BLUE = '#2a78d6', ORANGE = '#eb6834', AQUA = '#1baf7a', INK = '#1d1d1b', MUTED = '#5f5e5a';
  const SERIES = [['REINFORCE, w₂ = 1', 1, BLUE], ['TIS, w₂ = 2', 2, ORANGE], ['IS, w₂ = 20', RHO, AQUA]];

  el.innerHTML = `
    <div class="w-panels">
      <figure><svg class="w-plot" viewBox="0 0 600 220"></svg><figcaption>bars: share of batches of 16 from the far engine landing at each value, one color per weight</figcaption></figure>
    </div>`;
  const svgEl = (tag, attrs, text) => {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const binomial = (n, q) => {
    const out = [(1 - q) ** n];
    for (let k = 0; k < n; k++) out.push(out[k] * (n - k) / (k + 1) * q / (1 - q));
    return out;
  };

  const svg = el.querySelector('.w-plot');
  const L = 36, R = 12, T = 30, Bm = 34, W = 600, H = 220, XMAX = 7;
  const sx = x => L + (W - L - R) * x / XMAX, sy = y => H - Bm - (H - Bm - T) * y;
  const pk = binomial(B, Q);
  // axes
  svg.appendChild(svgEl('line', { x1: L, y1: H - Bm, x2: W - R, y2: H - Bm, stroke: MUTED }));
  for (let x = 0; x <= XMAX; x++) {
    svg.appendChild(svgEl('line', { x1: sx(x), y1: H - Bm, x2: sx(x), y2: H - Bm + 3, stroke: MUTED }));
    svg.appendChild(svgEl('text', { x: sx(x), y: H - Bm + 13, 'text-anchor': 'middle', 'font-size': 9, fill: MUTED }, x === 1 ? 'g' : `${x}g`));
  }
  svg.appendChild(svgEl('text', { x: (L + W - R) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 9, fill: MUTED }, 'batch estimate, token 2 coordinate, as a multiple of the truth g'));
  // truth
  svg.appendChild(svgEl('line', { x1: sx(1), y1: T - 8, x2: sx(1), y2: H - Bm, stroke: INK, 'stroke-dasharray': '4 3' }));
  svg.appendChild(svgEl('text', { x: sx(1) + 4, y: T, 'font-size': 9, fill: INK }, 'truth g'));
  // bars: one group per token-2 count, the three weights side by side; triangles: each weight's average over batches
  const bw = 6, n = SERIES.length;
  SERIES.forEach(([name, wi, color], i) => {
    pk.forEach((p, k) => {
      const e = wi * k / B / P2;
      if (p < 0.002 || e > XMAX) return;
      const x = sx(e) - (n * bw) / 2 + i * bw;
      svg.appendChild(svgEl('rect', { x, y: sy(p), width: bw - 1, height: sy(0) - sy(p), fill: color }));
      if (p >= 0.05 && i === 0) svg.appendChild(svgEl('text', { x: sx(e), y: sy(p) - 4, 'text-anchor': 'middle', 'font-size': 9, fill: INK }, `${Math.round(p * 100)}%`));
    });
    const mean = Q * wi / P2;
    svg.appendChild(svgEl('path', { d: `M${sx(mean) - 5},${H - Bm + 8} l5,-8 l5,8 z`, fill: color }));
    svg.appendChild(svgEl('rect', { x: W - R - 170, y: T - 8 + i * 12, width: 8, height: 8, fill: color }));
    svg.appendChild(svgEl('text', { x: W - R - 158, y: T + i * 12, 'font-size': 9, fill: INK }, `${name}: averages ${mean.toFixed(2)} g`));
  });
  svg.appendChild(svgEl('text', { x: W - R - 170, y: T + n * 12 + 2, 'font-size': 9, fill: MUTED }, '▲ average over batches'));
};
