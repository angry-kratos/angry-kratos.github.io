// Step scrubber: walk through every measured point of one training step.
// Data: 55 hook/checkpoint readings from profile_step2.py plus the true peak
// from the allocator trace, decomposed with vocab_ledger.json. Units: MiB.
(function () {
  const P = [{"l": "step 2 start", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 0.0, "v": 0.0, "t": 1494.2, "hw": 1494.2}, {"l": "F00 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 51.0, "v": 0.0, "t": 1545.2, "hw": 1545.2}, {"l": "F00 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 820.1, "v": 0.0, "t": 2314.3, "hw": 2939.3}, {"l": "F01 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 820.1, "v": 0.0, "t": 2314.3, "hw": 2939.3}, {"l": "F01 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 1589.3, "v": 0.0, "t": 3083.4, "hw": 3708.4}, {"l": "F02 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 1589.3, "v": 0.0, "t": 3083.4, "hw": 3708.4}, {"l": "F02 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 2358.4, "v": 0.0, "t": 3852.6, "hw": 4477.5}, {"l": "F03 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 2358.4, "v": 0.0, "t": 3852.6, "hw": 4477.5}, {"l": "F03 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 3127.5, "v": 0.0, "t": 4621.7, "hw": 5246.6}, {"l": "F04 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 3127.5, "v": 0.0, "t": 4621.7, "hw": 5246.6}, {"l": "F04 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 3896.6, "v": 0.0, "t": 5390.8, "hw": 6015.8}, {"l": "F05 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 3896.6, "v": 0.0, "t": 5390.8, "hw": 6015.8}, {"l": "F05 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 4665.8, "v": 0.0, "t": 6159.9, "hw": 6784.9}, {"l": "F06 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 4665.8, "v": 0.0, "t": 6159.9, "hw": 6784.9}, {"l": "F06 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 5434.9, "v": 0.0, "t": 6929.1, "hw": 7554.0}, {"l": "F07 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 5434.9, "v": 0.0, "t": 6929.1, "hw": 7554.0}, {"l": "F07 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 6204.0, "v": 0.0, "t": 7698.2, "hw": 8323.1}, {"l": "F08 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 6204.0, "v": 0.0, "t": 7698.2, "hw": 8323.1}, {"l": "F08 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 6973.1, "v": 0.0, "t": 8467.3, "hw": 9092.3}, {"l": "F09 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 6973.1, "v": 0.0, "t": 8467.3, "hw": 9092.3}, {"l": "F09 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 7742.3, "v": 0.0, "t": 9236.4, "hw": 9861.4}, {"l": "F10 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 7742.3, "v": 0.0, "t": 9236.4, "hw": 9861.4}, {"l": "F10 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 8511.4, "v": 0.0, "t": 10005.6, "hw": 10630.5}, {"l": "F11 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 8511.4, "v": 0.0, "t": 10005.6, "hw": 10630.5}, {"l": "F11 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 9280.5, "v": 0.0, "t": 10774.7, "hw": 11399.6}, {"l": "logits / before loss", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 9304.6, "v": 1570.5, "t": 12369.3, "hw": 12369.3}, {"l": "after loss", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 9277.6, "v": 3141.1, "t": 13912.9, "hw": 13939.9}, {"l": "inside loss backward (trace)", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 9277.6, "v": 6282.1, "t": 17053.9, "hw": 17053.9}, {"l": "B11 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 9400.7, "v": 1570.5, "t": 12465.5, "hw": 17053.9}, {"l": "B11 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 27.0, "act": 8656.4, "v": 1570.5, "t": 11748.1, "hw": 17053.9}, {"l": "B10 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 27.0, "act": 8632.4, "v": 1570.5, "t": 11724.1, "hw": 17053.9}, {"l": "B10 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 54.1, "act": 7888.4, "v": 1570.5, "t": 11007.2, "hw": 17053.9}, {"l": "B09 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 54.1, "act": 7864.4, "v": 1570.5, "t": 10983.2, "hw": 17053.9}, {"l": "B09 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 81.1, "act": 7119.5, "v": 1570.5, "t": 10265.3, "hw": 17053.9}, {"l": "B08 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 81.1, "act": 7095.5, "v": 1570.5, "t": 10241.3, "hw": 17053.9}, {"l": "B08 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 108.2, "act": 6351.1, "v": 1570.5, "t": 9524.0, "hw": 17053.9}, {"l": "B07 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 108.2, "act": 6327.1, "v": 1570.5, "t": 9500.0, "hw": 17053.9}, {"l": "B07 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 135.2, "act": 5582.0, "v": 1570.5, "t": 8781.9, "hw": 17053.9}, {"l": "B06 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 135.2, "act": 5558.0, "v": 1570.5, "t": 8757.9, "hw": 17053.9}, {"l": "B06 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 162.2, "act": 4812.9, "v": 1570.5, "t": 8039.8, "hw": 17053.9}, {"l": "B05 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 162.2, "act": 4788.9, "v": 1570.5, "t": 8015.8, "hw": 17053.9}, {"l": "B05 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 189.3, "act": 4044.0, "v": 1570.5, "t": 7298.0, "hw": 17053.9}, {"l": "B04 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 189.3, "act": 4020.0, "v": 1570.5, "t": 7274.0, "hw": 17053.9}, {"l": "B04 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 216.3, "act": 3274.9, "v": 1570.5, "t": 6555.9, "hw": 17053.9}, {"l": "B03 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 216.3, "act": 3250.9, "v": 1570.5, "t": 6531.9, "hw": 17053.9}, {"l": "B03 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 243.3, "act": 2505.8, "v": 1570.5, "t": 5813.8, "hw": 17053.9}, {"l": "B02 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 243.3, "act": 2481.8, "v": 1570.5, "t": 5789.8, "hw": 17053.9}, {"l": "B02 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 270.4, "act": 1737.4, "v": 1570.5, "t": 5072.5, "hw": 17053.9}, {"l": "B01 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 270.4, "act": 1713.4, "v": 1570.5, "t": 5048.5, "hw": 17053.9}, {"l": "B01 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 297.4, "act": 968.3, "v": 1570.5, "t": 4330.4, "hw": 17053.9}, {"l": "B00 before", "w": 474.7, "a": 949.4, "o": 70.1, "g": 297.4, "act": 944.3, "v": 1570.5, "t": 4306.4, "hw": 17053.9}, {"l": "B00 after", "w": 474.7, "a": 949.4, "o": 70.1, "g": 324.5, "act": 199.1, "v": 1570.5, "t": 3588.3, "hw": 17053.9}, {"l": "after backward", "w": 474.7, "a": 949.4, "o": 70.1, "g": 474.7, "act": 3.9, "v": 1570.5, "t": 3543.3, "hw": 17053.9}, {"l": "after del logits", "w": 474.7, "a": 949.4, "o": 70.1, "g": 474.7, "act": 3.9, "v": 0.0, "t": 1972.8, "hw": 17053.9}, {"l": "after step()", "w": 474.7, "a": 949.4, "o": 70.1, "g": 474.7, "act": 3.9, "v": 0.0, "t": 1972.8, "hw": 17053.9}, {"l": "after zero_grad", "w": 474.7, "a": 949.4, "o": 70.1, "g": 0.0, "act": 0.0, "v": 0.0, "t": 1494.2, "hw": 17053.9}];
  const root = document.getElementById('tm-scrubber');
  if (!root) return;

  const C = {
    w: '#486b9c', a: '#80a8ca', o: '#c7b58a', g: '#dd9761', act: '#1baf7a', v: '#b783cf',
    hw: '#b23a48', ink: '#2b2a27', mute: '#7a776f', grid: '#e6dfd1', paper: '#faf6ee', idle: '#efe8da'
  };
  const fmt = n => Math.round(n).toLocaleString('en-US');
  const blk = l => { const m = l.match(/^([FB])(\d\d) (before|after)$/); return m ? { dir: m[1], k: +m[2], when: m[3] } : null; };

  function state(p) {
    // saved[k]: block k holds saved activations; grad[k]: block k has weight gradients
    const saved = Array(12).fill(false), grad = Array(12).fill(false);
    let cur = -1, head = false, emb = false;
    const b = blk(p.l);
    if (b && b.dir === 'F') {
      emb = true;
      for (let k = 0; k < 12; k++) saved[k] = b.when === 'after' ? k <= b.k : k < b.k;
      cur = b.k;
    } else if (b && b.dir === 'B') {
      emb = true; head = true;
      for (let k = 0; k < 12; k++) {
        saved[k] = b.when === 'before' ? k <= b.k : k < b.k;
        grad[k] = !saved[k];
      }
      cur = b.k;
    } else if (/^(logits|after loss|inside loss)/.test(p.l)) {
      emb = true; head = true; saved.fill(true); cur = 12;
    } else if (/after backward|after del logits|after step/.test(p.l)) {
      grad.fill(true); emb = true;
    }
    return { saved, grad, cur, head, emb };
  }

  function caption(p) {
    const b = blk(p.l);
    if (p.l === 'step 2 start') return 'Step 2 begins. Already on the GPU: the weights (475 MiB), Adam’s two running averages m and v (949 MiB, created by step 1 and kept forever), and 70 MiB of attention masks and library scratch space. No gradients yet. This 1,494 MiB floor never goes away.';
    if (b && b.dir === 'F' && b.k === 0 && b.when === 'before') return 'The token and position embeddings are looked up: 51 MiB. Now the input walks down through the 12 blocks.';
    if (b && b.dir === 'F' && b.when === 'before') return `Entering block ${b.k + 1}. Nothing from the earlier blocks has been thrown away.`;
    if (b && b.dir === 'F') return `Block ${b.k + 1} is done. Besides its output, it kept 769 MiB of in-between results, because backward will need them. ${b.k + 1} block${b.k ? 's' : ''} × 769 MiB = ${fmt((b.k + 1) * 769.1)} MiB saved so far.`;
    if (p.l.startsWith('logits')) return 'lm_head scores all 50,257 words at all 8,192 positions. That is the logits: 1,571 MiB in a single tensor (purple).';
    if (p.l === 'after loss') return 'Cross-entropy keeps its log-softmax output, another 1,571 MiB with the same shape as the logits. Forward is finished at 13,913 MiB. This is the highest point my hooks ever recorded.';
    if (p.l.startsWith('inside')) return 'The real peak, between two hooks. The first thing backward does is compute the gradient of the loss with respect to those 50,257 scores, and that creates two more logits-sized tensors while everything else is still alive. Four vocab-sized tensors at once = 6,282 MiB. Peak: 17,054 MiB. Only the allocator trace sees this moment.';
    if (b && b.dir === 'B' && b.k === 11 && b.when === 'before') return 'The cross-entropy temporaries are freed. The logits are still here, because my Python variable `logits` still points at them. Backward now walks up the model in reverse, starting with block 12.';
    if (b && b.dir === 'B' && b.when === 'before') return `Backward reaches block ${b.k + 1}.`;
    if (b && b.dir === 'B') return `Block ${b.k + 1}’s backward is done. It used its saved 769 MiB to compute gradients, then freed it. Its weight gradients (27 MiB, orange) stay until the optimizer uses them.`;
    if (p.l === 'after backward') return 'Backward is finished and every saved block is gone. The embedding gradient arrived last (150 MiB): it is shared with lm_head, so autograd waited for both halves. Purple is still here, held only by the variable `logits`.';
    if (p.l === 'after del logits') return '`del logits` frees 1,571 MiB in one line. A tensor stays alive as long as any name points at it.';
    if (p.l === 'after step()') return 'optimizer.step() updates the weights, m and v in place. Its persistent state does not grow in step 2, though temporary allocations occur inside the update.';
    if (p.l === 'after zero_grad') return 'zero_grad(set_to_none=True) frees the 475 MiB of gradients. We are back on the 1,494 MiB floor, ready for step 3.';
    return '';
  }

  const W = 680, H = 430, Y0 = 395, K = 345 / 18000;
  const layerNames = ['embeddings'].concat([...Array(12)].map((_, k) => `block ${k + 1}`)).concat(['lm_head + loss']);

  function svg(i) {
    const p = P[i], s = state(p);
    let h = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Memory at point ${i + 1}: ${fmt(p.t)} MiB">`;
    // left: the model
    h += `<text x="30" y="22" font-size="13" fill="${C.mute}">the model</text>`;
    const dirTxt = /^(F|logits|after loss)/.test(p.l) ? 'forward ↓' : /^B|inside/.test(p.l) ? 'backward ↑' : '';
    if (dirTxt) h += `<text x="250" y="22" font-size="13" fill="${C.ink}" text-anchor="end">${dirTxt}</text>`;
    layerNames.forEach((name, r) => {
      const y = 34 + r * 25;
      let fill = C.idle, stroke = '#d9d0bd', txt = C.mute;
      if (r === 0 && s.emb) { fill = '#e3dccd'; }
      if (r >= 1 && r <= 12 && s.saved[r - 1]) { fill = C.act; txt = '#ffffff'; stroke = C.act; }
      if (r === 13 && p.v > 0) { fill = C.v; txt = '#ffffff'; stroke = C.v; }
      h += `<rect x="30" y="${y}" width="220" height="21" rx="4" fill="${fill}" stroke="${stroke}"/>`;
      h += `<text x="140" y="${y + 15}" font-size="12" text-anchor="middle" fill="${txt}">${name}</text>`;
      if (r >= 1 && r <= 12 && s.grad[r - 1]) h += `<rect x="256" y="${y + 3}" width="10" height="15" rx="2" fill="${C.g}"/>`;
      const curRow = s.cur === 12 ? 13 : s.cur + 1;
      if (s.cur >= 0 && r === curRow) h += `<rect x="25" y="${y - 3}" width="230" height="27" rx="6" fill="none" stroke="${C.hw}" stroke-width="2"/>`;
    });
    // right: the memory column
    [0, 6000, 12000, 18000].forEach(m => {
      const y = Y0 - m * K;
      h += `<line x1="350" y1="${y}" x2="490" y2="${y}" stroke="${C.grid}"/><text x="498" y="${y + 4}" font-size="12" fill="${C.mute}">${fmt(m)}</text>`;
    });
    h += `<text x="350" y="22" font-size="13" fill="${C.mute}">GPU memory (MiB)</text>`;
    let y = Y0;
    [['w', C.w], ['a', C.a], ['o', C.o], ['g', C.g], ['act', C.act], ['v', C.v]].forEach(([k, col]) => {
      const hh = p[k] * K; if (hh <= 0) return;
      y -= hh;
      h += `<rect x="370" y="${y.toFixed(1)}" width="100" height="${Math.max(hh, 0.8).toFixed(1)}" fill="${col}"/>`;
    });
    h += `<text x="420" y="${(y - 8).toFixed(1)}" font-size="14" font-weight="600" text-anchor="middle" fill="${C.ink}">${fmt(p.t)}</text>`;
    const hy = Y0 - p.hw * K;
    h += `<line x1="356" y1="${hy}" x2="484" y2="${hy}" stroke="${C.hw}" stroke-width="1.6" stroke-dasharray="5 4"/>`;
    h += `<text x="560" y="${hy + 4}" font-size="12" fill="${C.hw}">high-water ${fmt(p.hw)}</text>`;
    h += '</svg>';
    return h;
  }

  root.innerHTML = `
    <div class="tm-legend">
      <span><i style="background:${C.w}"></i>weights</span><span><i style="background:${C.a}"></i>Adam m, v</span>
      <span><i style="background:${C.o}"></i>masks + scratch</span><span><i style="background:${C.g}"></i>gradients</span>
      <span><i style="background:${C.act}"></i>activations + other residual</span><span><i style="background:${C.v}"></i>vocab-sized tensors</span>
    </div>
    <div class="tm-svg"></div>
    <div class="tm-readout"><span class="tm-label"></span></div>
    <p class="tm-caption"></p>
    <div class="tm-controls">
      <button type="button" data-d="-1" aria-label="Previous point">←</button>
      <button type="button" class="tm-play">play</button>
      <button type="button" data-d="1" aria-label="Next point">→</button>
      <input type="range" min="0" max="${P.length - 1}" step="1" value="0" aria-label="Point in the training step">
    </div>`;
  const $ = q => root.querySelector(q);
  const range = $('input'); let i = 0, timer = null;
  function render() {
    const p = P[i];
    $('.tm-svg').innerHTML = svg(i);
    $('.tm-label').textContent = `point ${i + 1} of ${P.length} · ${p.l}`;
    $('.tm-caption').textContent = caption(p);
    range.value = i;
  }
  function go(n) { i = Math.max(0, Math.min(P.length - 1, n)); render(); }
  root.querySelectorAll('button[data-d]').forEach(b => b.addEventListener('click', () => { stop(); go(i + +b.dataset.d); }));
  range.addEventListener('input', () => { stop(); go(+range.value); });
  function stop() { if (timer) { clearInterval(timer); timer = null; $('.tm-play').textContent = 'play'; } }
  $('.tm-play').addEventListener('click', () => {
    if (timer) return stop();
    if (i === P.length - 1) i = -1;
    $('.tm-play').textContent = 'pause';
    timer = setInterval(() => { if (i >= P.length - 1) return stop(); go(i + 1); }, 650);
  });
  render();
})();
