/*
 * Unit tests for static/js/risk-scale.js.
 *
 * Uses Node's built-in test runner (`node --test`) and assert module -
 * no npm install, no devDependencies, nothing to fall out of date. Run
 * from anywhere with:
 *     node --test 08_webapp/tests/js
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const { PROB_MIN, COLOR_STOPS, colorForProb, radiusForProb } = require('../../static/js/risk-scale.js');

test('PROB_MIN matches the server-side display cutoff (app.py PROB_THRESHOLD)', () => {
  assert.equal(PROB_MIN, 0.3);
});

test('colorForProb at the cutoff returns the first color stop exactly', () => {
  assert.equal(colorForProb(PROB_MIN), 'rgb(145,207,96)');
});

test('colorForProb at p=1.0 returns the last color stop exactly', () => {
  assert.equal(colorForProb(1.0), 'rgb(215,48,39)');
});

test('colorForProb interpolates within the first segment (green -> yellow) monotonically on every channel', () => {
  // Only the green->yellow segment (COLOR_STOPS[0]->[1]) is monotonic on
  // R, G, and B individually - the palette's later segments (yellow->
  // orange->red) have R and B *decrease* even though perceived "redness"
  // keeps rising, which is normal for a ColorBrewer-style palette but
  // means a blanket "every channel rises with p" claim across the whole
  // range is false (caught by an earlier, wrong version of this test).
  const samples = [0.3, 0.35, 0.4, 0.45, 0.5]; // p in [PROB_MIN, PROB_MIN + (1-PROB_MIN)/3]
  const rgbs = samples.map(p => colorForProb(p).match(/\d+/g).map(Number));
  for (let i = 1; i < rgbs.length; i++) {
    for (let ch = 0; ch < 3; ch++) {
      assert.ok(rgbs[i][ch] >= rgbs[i - 1][ch],
        `channel ${ch} should not decrease within the first segment: ${rgbs[i - 1]} -> ${rgbs[i]}`);
    }
  }
});

test('colorForProb\'s red channel is higher at the yellow stop than at the final red stop (non-monotonic red, by design)', () => {
  // Documents the same real property the test above exists to avoid
  // getting wrong again: RGB(254,...) at the yellow stop has a HIGHER red
  // channel than RGB(215,...) at the actual "red" end of the scale.
  const [rYellow] = colorForProb(0.3 + (1 - 0.3) / 3).match(/\d+/g).map(Number);
  const [rFinal] = colorForProb(1.0).match(/\d+/g).map(Number);
  assert.ok(rYellow > rFinal, `expected yellow-stop red (${rYellow}) > final red-stop red (${rFinal})`);
});

test('colorForProb clamps values below the cutoff to the same color as the cutoff itself', () => {
  assert.equal(colorForProb(0.0), colorForProb(PROB_MIN));
  assert.equal(colorForProb(-5), colorForProb(PROB_MIN));
});

test('colorForProb clamps values above 1.0 to the same color as 1.0', () => {
  assert.equal(colorForProb(1.5), colorForProb(1.0));
  assert.equal(colorForProb(100), colorForProb(1.0));
});

test('colorForProb produces a different color for a real day-to-day shift (0.82 vs 0.73)', () => {
  // The concrete bug this module fixed: with the old 4-bucket scheme,
  // 0.82 and 0.73 both landed in the same "red" bucket and looked
  // identical even though the model's output had genuinely changed.
  assert.notEqual(colorForProb(0.82), colorForProb(0.73));
});

test('radiusForProb is 5px at the cutoff and 14px at p=1.0', () => {
  assert.equal(radiusForProb(PROB_MIN), 5);
  assert.equal(radiusForProb(1.0), 14);
});

test('radiusForProb is monotonically non-decreasing as probability rises', () => {
  const samples = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  for (let i = 1; i < samples.length; i++) {
    assert.ok(radiusForProb(samples[i]) >= radiusForProb(samples[i - 1]));
  }
});

test('radiusForProb clamps outside [PROB_MIN, 1.0] the same way colorForProb does', () => {
  assert.equal(radiusForProb(-5), radiusForProb(PROB_MIN));
  assert.equal(radiusForProb(100), radiusForProb(1.0));
});

test('COLOR_STOPS has exactly 4 RGB triples (matches colorForProb\'s interpolation math)', () => {
  assert.equal(COLOR_STOPS.length, 4);
  for (const stop of COLOR_STOPS) {
    assert.equal(stop.length, 3);
    for (const channel of stop) {
      assert.ok(channel >= 0 && channel <= 255);
    }
  }
});
