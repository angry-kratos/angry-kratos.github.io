// "Will it fit?" calculator for GPT-2 small, fp32, AdamW, on one A40.
// Peak memory model, built from this post's measurements (see the post's
// "Will it fit?" section). Allocated MiB at the step-2 peak:
//   fixed            1,494.2                 weights + Adam m,v + masks + workspaces
//   + 12 x T^2 bytes                         bool causal-mask comparison, one per layer
//   + B x per-sequence cost, where per sequence:
//       12 layers x 16 x C x 4 bytes x T     linear-in-T saved tensors (49,152 B/token/layer)
//       12 layers x n_head x T^2 x 4 bytes   attention probabilities
//        2 x C x 4 bytes x T                 embedding output + ln_f output
//        4 x V x 4 bytes x T                 four vocab-sized tensors at the peak
// Checks: B=8,T=1024 -> 17,052 (measured 17,054); B=20 -> 40,372 (measured 40,375).
// Reserved is estimated as allocated x 1.11, the ratio measured at B=16..20, T=1024.
// The usable budget is the 45,440 MiB that nvidia-smi reported free before the run.
(function () {
  const root = document.getElementById('tm-fit');
  if (!root) return;
  const MiB = 2 ** 20, C = 768, NH = 12, L = 12, V = 50257;
  const FIXED = 1494.2, FREE = 45440, CAP = 46068, RES = 1.11;
  const col = { fixed: '#80a8ca', lin: '#1baf7a', att: '#0f7a54', voc: '#b783cf', hw: '#b23a48', ink: '#2b2a27', mute: '#7a776f', grid: '#e6dfd1' };
  const fmt = n => Math.round(n).toLocaleString('en-US');

  function model(B, T) {
    const lin = B * L * 16 * C * 4 * T / MiB;
    const att = B * L * NH * T * T * 4 / MiB + L * T * T / MiB;
    const emb = B * 2 * C * 4 * T / MiB;
    const voc = B * 4 * V * 4 * T / MiB;
    const alloc = FIXED + lin + att + emb + voc;
    return { fixed: FIXED, lin: lin + emb, att, voc, alloc, res: alloc * RES };
  }
  function maxBatch(T) { let b = 0; while (model(b + 1, T).res < FREE && b < 4096) b++; return b; }

  root.innerHTML = `
    <div class="tm-row"><label>batch size <b class="tm-b"></b></label><input aria-label="Batch size" class="tm-in-b" type="range" min="1" max="64" step="1" value="8"></div>
    <div class="tm-row"><label>tokens per sequence <b class="tm-t"></b></label><input aria-label="Tokens per sequence" class="tm-in-t" type="range" min="128" max="1024" step="128" value="1024"></div>
    <div class="tm-bar-svg"></div>
    <div class="tm-verdict"></div>
    <p class="tm-note"></p>`;
  const $ = q => root.querySelector(q);

  function render() {
    const B = +$('.tm-in-b').value, T = +$('.tm-in-t').value, m = model(B, T);
    $('.tm-b').textContent = B; $('.tm-t').textContent = T;
    const W = 680, X0 = 20, X1 = 640, K = (X1 - X0) / Math.max(56000, m.res * 1.08);
    let h = `<svg viewBox="0 0 ${W} 150" width="100%" role="img" aria-label="Estimated peak ${fmt(m.alloc)} MiB allocated">`;
    [0, 10000, 20000, 30000, 40000, 50000].forEach(v => {
      const x = X0 + v * K;
      h += `<line x1="${x}" y1="30" x2="${x}" y2="96" stroke="${col.grid}"/><text x="${x}" y="114" font-size="11" fill="${col.mute}" text-anchor="middle">${v / 1000}k</text>`;
    });
    let x = X0;
    [['fixed', 'model + Adam'], ['lin', 'layer activations'], ['att', 'attention probs (T²)'], ['voc', 'vocab tensors']].forEach(([k]) => {
      const w = m[k] * K; h += `<rect x="${x.toFixed(1)}" y="40" width="${w.toFixed(1)}" height="40" fill="${col[k]}"/>`; x += w;
    });
    const xr = X0 + m.res * K;
    h += `<rect x="${x.toFixed(1)}" y="40" width="${Math.max(0, xr - x).toFixed(1)}" height="40" fill="none" stroke="${col.mute}" stroke-dasharray="3 3"/>`;
    const xf = X0 + FREE * K, xc = X0 + CAP * K;
    h += `<line x1="${xf}" y1="24" x2="${xf}" y2="96" stroke="${col.hw}" stroke-width="2"/>`;
    h += `<text x="${xf - 4}" y="18" font-size="12" fill="${col.hw}" text-anchor="end">free on the A40: 45,440</text>`;
    h += `<line x1="${xc}" y1="30" x2="${xc}" y2="96" stroke="${col.mute}" stroke-dasharray="2 3"/>`;
    h += `<text x="20" y="140" font-size="12" fill="${col.ink}">allocated ≈ ${fmt(m.alloc)} MiB · reserved (dashed) ≈ ${fmt(m.res)} MiB</text>`;
    h += '</svg>';
    $('.tm-bar-svg').innerHTML = h;
    const fits = m.res < FREE, mb = maxBatch(T);
    const v = $('.tm-verdict');
    v.className = 'tm-verdict ' + (fits ? 'ok' : 'oom');
    v.textContent = fits ? `Estimated to fit. The largest batch at ${T.toLocaleString('en-US')} tokens: about ${mb}.` : `Estimated not to fit. The largest batch at ${T.toLocaleString('en-US')} tokens: about ${mb}.`;
    const share = n => Math.round(100 * n / m.alloc);
    $('.tm-note').textContent = `Of the allocated peak: model + Adam ${share(m.fixed)}%, layer activations ${share(m.lin)}%, attention probabilities ${share(m.att)}%, vocab-sized tensors ${share(m.voc)}%.` + (T < 1024 ? ' (Below 1,024 tokens this is extrapolated from the formula: I only measured T = 1,024.)' : '');
  }
  root.querySelectorAll('input').forEach(i => i.addEventListener('input', render));
  render();
})();
