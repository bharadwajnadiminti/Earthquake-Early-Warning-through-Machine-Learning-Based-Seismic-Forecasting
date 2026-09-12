/*
 * risk-scale.js
 *
 * Continuous color + radius scale for a predicted probability, used by
 * templates/index.html to draw each active cell's forecast marker.
 *
 * Pulled out of the page's inline <script> into its own file for one
 * reason: it's the one piece of this page's JS that's pure logic (no DOM,
 * no Leaflet, no localStorage) - so it's the one piece that can be unit
 * tested directly with plain Node (see tests/js/risk-scale.test.js)
 * instead of only ever being checked by eye in a browser screenshot.
 *
 * Loaded as a plain browser <script> (defines globals) AND as a CommonJS
 * module for the Node test file - the `typeof module` guard at the
 * bottom is the only thing that makes that dual use possible.
 */

// The display cutoff - inference/app.py only ever sends cells with
// predicted probability > 0.3 in the first place, so that's the bottom of
// the range being colored/sized here, not 0.
var PROB_MIN = 0.3;

// ColorBrewer RdYlGn (reversed): green (low end of the shown range) ->
// yellow -> orange -> red (p = 1.0).
var COLOR_STOPS = [
  [145, 207, 96],
  [254, 224, 139],
  [252, 141, 89],
  [215, 48, 39],
];

function colorForProb(p) {
  var t = Math.max(0, Math.min(1, (p - PROB_MIN) / (1 - PROB_MIN)));
  var seg = t * (COLOR_STOPS.length - 1);
  var i = Math.min(COLOR_STOPS.length - 2, Math.floor(seg));
  var f = seg - i;
  var a = COLOR_STOPS[i], b = COLOR_STOPS[i + 1];
  var rgb = [0, 1, 2].map(function (c) { return Math.round(a[c] + (b[c] - a[c]) * f); });
  return 'rgb(' + rgb.join(',') + ')';
}

function radiusForProb(p) {
  var t = Math.max(0, Math.min(1, (p - PROB_MIN) / (1 - PROB_MIN)));
  return 5 + 9 * t; // 5px at the cutoff, 14px at p=1.0
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { PROB_MIN: PROB_MIN, COLOR_STOPS: COLOR_STOPS, colorForProb: colorForProb, radiusForProb: radiusForProb };
}
