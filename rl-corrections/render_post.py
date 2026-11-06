"""
Render a post (markdown) to a local HTML page: contents on the left, references as side notes on the right.
Citations in the markdown are links to `#ref-N`; the reference list has `<a id="ref-N"></a>` anchors.

  python render_post.py bias_variance.md out.html            # local preview: images by absolute path
  python render_post.py bias_variance.md out.html --web      # for the site: images relative, link home
"""
import html
import os
import re
import sys

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
:root {{ --ink:#1d1d1b; --muted:#5f5e5a; --line:#e4e3df; --bg:#fcfcfb; --code:#f2f1ee; --accent:#2a78d6; --th:#f0efec; --widget:#fff; }}
html.dark {{ --ink:#e8e6e1; --muted:#9c9a92; --line:#2c2c2a; --bg:#151514; --code:#1f1f1d; --accent:#7fb2f0; --th:#242422; --widget:#1a1a19; }}
html.dark img {{ opacity:0.9; }}
/* widget charts hard-code the light palette in SVG attributes; remap them in dark mode */
html.dark .w-panels svg [fill="#1d1d1b"] {{ fill:var(--ink); }}  html.dark .w-panels svg [stroke="#1d1d1b"] {{ stroke:var(--ink); }}
html.dark .w-panels svg [fill="#5f5e5a"] {{ fill:var(--muted); }} html.dark .w-panels svg [stroke="#5f5e5a"] {{ stroke:var(--muted); }}
html.dark .w-panels svg [stroke="#e4e3df"] {{ stroke:var(--line); }}
html.dark .w-panels svg [fill="#fff"] {{ fill:var(--widget); }}   html.dark .w-panels svg [stroke="#fff"] {{ stroke:var(--widget); }}
.mode {{ float:right; font-size:13px; color:var(--muted); background:none; border:0; cursor:pointer; padding:0; }} .mode:hover {{ color:var(--accent); }}
.mode::before {{ content:'\2600\FE0E dark'; }} html.dark .mode::before {{ content:'\263E\FE0E light'; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.65 -apple-system,system-ui,sans-serif; }}
a {{ color:var(--accent); }}
#toc {{ position:fixed; top:0; left:0; width:230px; height:100vh; overflow-y:auto; padding:40px 16px 40px 24px;
        box-sizing:border-box; border-right:1px solid var(--line); font-size:13px; }}
#toc a {{ display:block; color:var(--muted); text-decoration:none; padding:3px 0; }}
#toc a.h3 {{ padding-left:14px; }}
#toc a.active {{ color:var(--ink); font-weight:600; }}
#toc .title {{ font-weight:600; color:var(--ink); margin-bottom:10px; }}
.home {{ font-size:13px; color:var(--muted); text-decoration:none; }} .home:hover {{ color:var(--accent); }}
main {{ margin-left:270px; max-width:820px; padding:40px 300px 80px 0; box-sizing:content-box; }}
pre {{ background:var(--code); padding:12px; overflow-x:auto; border-radius:6px; font-size:13px; }}
code {{ font:13px Menlo,monospace; }}
p code, li code, td code {{ background:var(--code); padding:1px 4px; border-radius:3px; }}
table {{ border-collapse:collapse; margin:12px 0; display:block; overflow-x:auto; font-size:14px;
         width:max-content; max-width:calc(100% + 300px); }}
th, td {{ border:1px solid var(--line); padding:6px 10px; white-space:nowrap; }} th {{ background:var(--th); }}
img {{ max-width:100%; }} em {{ color:var(--muted); }}
sup.cite a {{ color:var(--accent); text-decoration:none; font-size:11px; }}
sup.cite + sup.cite::before {{ content:','; font-size:11px; color:var(--accent); }}
.sidenote {{ float:right; clear:right; margin-right:-300px; width:270px; font-size:12px; line-height:1.45;
             color:var(--muted); margin-top:4px; margin-bottom:10px; }}
.sidenote b {{ color:var(--accent); }}
.sidenote.inline {{ float:none; display:block; margin:-6px 0 12px; width:auto; max-width:820px; border-left:2px solid var(--line); padding-left:8px; }}
#refs li {{ font-size:13px; color:var(--muted); }}
@media (max-width: 1400px) {{
  #toc {{ display:none; }} main {{ margin:0 auto; padding:24px 16px 60px; }}
  .sidenote {{ float:none; display:block; margin:6px 0 10px; width:auto; border-left:2px solid var(--line); padding-left:8px; }}
  table {{ max-width:100%; }}
}}
.katex-display {{ overflow-x:auto; overflow-y:hidden; padding:2px 0; }}
details.note {{ margin:4px 0 14px; }}
details.note summary {{ list-style:none; cursor:pointer; display:inline-block; font-size:12px; color:var(--accent); }}
details.note summary::-webkit-details-marker {{ display:none; }}
details.note summary::before {{ content:'?'; display:inline-block; width:16px; height:16px; line-height:16px; text-align:center;
         border:1px solid var(--accent); border-radius:50%; margin-right:6px; font-weight:600; }}
details.note[open] summary::before {{ content:'×'; }}
.note-body {{ margin-top:8px; padding:12px 16px; border-left:3px solid var(--accent); background:var(--code); border-radius:0 6px 6px 0;
         font-size:14px; }}
.note-body p:first-child {{ margin-top:0; }} .note-body p:last-child {{ margin-bottom:0; }}
.tip {{ border-bottom:1px dotted var(--accent); cursor:help; position:relative; }}
.tip:hover::after {{ content:attr(data-tip); position:absolute; left:0; top:1.5em; z-index:10; background:var(--ink); color:var(--bg);
         padding:6px 10px; border-radius:4px; font:12px/1.4 -apple-system,system-ui,sans-serif; width:max-content; max-width:320px; white-space:normal; }}
.widget {{ margin:16px 0 20px; padding:14px 16px; border:1px solid var(--line); border-radius:8px; background:var(--widget); font-size:13px; }}
.w-controls {{ display:flex; flex-wrap:wrap; gap:10px 24px; align-items:flex-end; }}
.w-controls label {{ display:flex; flex-direction:column; gap:2px; color:var(--muted); min-width:180px; }}
.w-controls label b {{ color:var(--ink); }} .w-controls input {{ width:180px; }}
.w-presets {{ display:flex; gap:6px; flex-wrap:wrap; }}
.w-presets button {{ font:12px -apple-system,system-ui,sans-serif; padding:3px 8px; border:1px solid var(--line); border-radius:4px; background:var(--code); cursor:pointer; }}
.w-presets button:hover {{ border-color:var(--accent); }}
.w-readout {{ margin:10px 0 6px; color:var(--muted); line-height:1.6; }} .w-readout b {{ color:var(--ink); }}
.w-panels {{ display:flex; flex-wrap:wrap; gap:8px; }}
.w-panels figure {{ flex:1 1 240px; margin:0; }} .w-panels svg {{ width:100%; height:auto; font-family:-apple-system,system-ui,sans-serif; }}
.w-panels figcaption {{ font-size:11px; color:var(--muted); text-align:center; }}
.c-blue {{ color:#2a78d6; }} .c-orange {{ color:#eb6834; }} .c-aqua {{ color:#1baf7a; }}
</style>
<script>
// Same key the Quarto site writes, so the toggle here and the one in the site navbar stay in step.
(function () {{ try {{ if (localStorage.getItem('quarto-color-scheme') === 'alternate') document.documentElement.classList.add('dark'); }} catch (e) {{}} }})();
window.toggleMode = function () {{
  const dark = document.documentElement.classList.toggle('dark');
  try {{ localStorage.setItem('quarto-color-scheme', dark ? 'alternate' : 'default'); }} catch (e) {{}}
}};
</script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
</head><body>
<nav id="toc"></nav><main id="c">{home}</main>
<script id="md" type="text/plain">{md}</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<script>{widgets}</script>
<script>
const raw = document.createElement('textarea'); raw.innerHTML = document.getElementById('md').textContent;
// Math: swap each $$...$$ / $...$ span (outside code) for a placeholder so marked leaves `_`, `*`, `|` alone,
// then put the escaped source back and let KaTeX render it.
const math = [];
const protectedMd = raw.value.replace(/(```[\s\S]*?```|`[^`\n]*`)|\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g, (m, code) => {{
  if (code) return m;
  math.push(m); return 'KATEXMATH' + (math.length - 1) + 'END';
}});
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
// Notes: a block  :::note Title ... :::  becomes a collapsed box opened by a "?" toggle.
const notes = [];
const withNotes = protectedMd.replace(/^:::note[ \t]+(.+)\n([\s\S]*?)^:::[ \t]*$/gm, (_, title, body) => {{
  notes.push(`<details class="note"><summary>${{title}}</summary><div class="note-body">${{marked.parse(body)}}</div></details>`);
  return 'NOTEBLOCK' + (notes.length - 1) + 'END';
}});
// Tooltips: {{text @@ note}} -> hover note saying where a symbol or number comes from.
const withTips = withNotes.replace(/\{{\{{([\s\S]+?) @@ ([\s\S]+?)\}}\}}/g,
  (_, t, tip) => `<span class="tip" data-tip="${{esc(tip).replace(/"/g, '&quot;')}}">${{t}}</span>`)
// Widgets: a line  :::widget name  becomes a box that widgets/name.js fills in.
  .replace(/^:::widget[ \t]+(\S+)[ \t]*$/gm, '<div class="widget" data-widget="$1"></div>');
const main = document.getElementById('c');
const home = main.innerHTML;
main.innerHTML = home + marked.parse(withTips)
  .replace(/<p>NOTEBLOCK(\d+)END<\/p>/g, (_, i) => notes[+i])
  .replace(/KATEXMATH(\d+)END/g, (_, i) => esc(math[+i]));

// References: text of each numbered entry.
const refs = {{}};
main.querySelectorAll('a[id^="ref-"]').forEach(a => {{
  const li = a.closest('li'); refs[a.id] = li.innerHTML.replace(a.outerHTML, '');
  li.closest('ol').id = 'refs';
}});

// Citations: [[n]](#ref-n) -> superscript, plus a side note the first time n appears in each section.
let seen = new Set();
for (const el of main.children) {{
  if (el.tagName === 'H2') seen = new Set();
  el.querySelectorAll('a[href^="#ref-"]').forEach(a => {{
    if (a.closest('#refs')) return;
    const id = a.getAttribute('href').slice(1), n = id.slice(4);
    const sup = document.createElement('sup'); sup.className = 'cite';
    sup.innerHTML = `<a href="#${{id}}">${{n}}</a>`;
    a.replaceWith(sup);
    if (!seen.has(id) && refs[id]) {{
      seen.add(id);
      const note = document.createElement('span'); note.className = 'sidenote';
      note.innerHTML = `<b>${{n}}</b> ${{refs[id]}}`;
      const table = sup.closest('table');
      if (table) {{                                                     // tables reach into the margin: note goes under the table
        note.classList.add('inline'); let after = table;
        while (after.nextSibling && after.nextSibling.classList && after.nextSibling.classList.contains('inline')) after = after.nextSibling;
        after.parentNode.insertBefore(note, after.nextSibling);
      }}
      else {{ const block = sup.closest('p, li') || sup; block.parentNode.insertBefore(note, block); }}
    }}
  }});
}}
// Tables: cells stay on one line unless the table would overflow the page; then let them wrap.
main.querySelectorAll('table').forEach(t => {{
  if (t.scrollWidth > t.clientWidth) t.querySelectorAll('th, td').forEach(c => c.style.whiteSpace = 'normal');
}});
// Colored columns: a header wrapped in <span class="c-..."> colors its whole column, matching the widgets' series.
main.querySelectorAll('th > span[class^="c-"]').forEach(s => {{
  const th = s.parentNode, i = [...th.parentNode.children].indexOf(th);
  th.closest('table').querySelectorAll('tbody tr').forEach(tr => tr.children[i] && tr.children[i].classList.add(s.className));
}});
// Shaded columns: a header wrapped in <span class="c-shade"> shades its cells by value, lighter is lower, on one
// log scale across every shaded column of the table.
main.querySelectorAll('table').forEach(t => {{
  const cols = [...t.querySelectorAll('th > span.c-shade')].map(s => [...s.parentNode.parentNode.children].indexOf(s.parentNode));
  if (!cols.length) return;
  const cells = [];
  t.querySelectorAll('tbody tr').forEach(tr => cols.forEach(i => {{
    const td = tr.children[i], v = parseFloat(td && td.textContent); if (!isNaN(v) && v > 0) cells.push([td, Math.log(v)]);
  }}));
  const lo = Math.min(...cells.map(c => c[1])), hi = Math.max(...cells.map(c => c[1]));
  cells.forEach(([td, v]) => td.style.background = `rgba(235,104,52,${{(0.06 + 0.5 * (v - lo) / (hi - lo || 1)).toFixed(2)}})`);
}});
// Contents, with the current section highlighted.
const toc = document.getElementById('toc');
const h1 = main.querySelector('h1'); toc.innerHTML = `<div class="title">${{h1 ? h1.textContent : ''}}</div>`;
const heads = [...main.querySelectorAll('h2, h3')];
heads.forEach((h, i) => {{
  h.id = h.id || 'sec-' + i;
  const a = document.createElement('a'); a.href = '#' + h.id; a.textContent = h.textContent;
  a.className = h.tagName.toLowerCase(); toc.appendChild(a);
}});
renderMathInElement(document.body, {{
  delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}],
  throwOnError: false,
}});
main.querySelectorAll('.widget').forEach(el => (window.widgets || {{}})[el.dataset.widget]?.(el));
const links = [...toc.querySelectorAll('a')];
addEventListener('scroll', () => {{
  let cur = 0; heads.forEach((h, i) => {{ if (h.getBoundingClientRect().top < 120) cur = i; }});
  links.forEach((a, i) => a.classList.toggle('active', i === cur));
}});
</script></body></html>'''


def main():
  src, out = sys.argv[1], sys.argv[2]
  web = '--web' in sys.argv[3:]
  folder = os.path.dirname(os.path.abspath(src))
  md = open(src).read()
  if not web:
    md = md.replace('](figs/', f'](file://{folder}/figs/')
  title = md.splitlines()[0].lstrip('# ').strip()
  names = re.findall(r'^:::widget[ \t]+(\S+)', md, flags=re.M)
  widgets = '\n'.join(open(os.path.join(folder, 'widgets', f'{n}.js')).read() for n in names)
  home = '<a class="home" href="../../">Tinkerings</a><button class="mode" onclick="toggleMode()" aria-label="toggle dark mode"></button>' if web else ''
  open(out, 'w').write(PAGE.format(title=html.escape(title), md=html.escape(md), widgets=widgets, home=home))
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
