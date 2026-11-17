/* ------------------------------------------------------------------
   Steel Ball Run pixel race — a small canvas diorama under the
   "Lesson 5" quote. Johnny & Gyro race across a dusk desert.
   No dependencies, no external assets.
   ------------------------------------------------------------------ */
(function () {
  'use strict';

  var canvas = document.getElementById('sbr-canvas');
  if (!canvas) return;

  var ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return;

  var W = canvas.width;
  var H = canvas.height;

  /* ---- palette ---------------------------------------------------- */
  var C = {
    ink:   '#241a1a',
    deep:  '#3a2a2a',
    plum:  '#5a3a3a',
    rust:  '#7a4a42',
    clay:  '#9a5f4e',
    ochre: '#c07a55',
    sand:  '#e0a36b',
    gold:  '#f0bd7a',
    cream: '#f7d9a0',
    paper: '#fbeccb',
    white: '#f4efe2',
    blue:  '#596a86',
    haze:  '#8fa0b5',
    teal:  '#3f5a63',
    olive: '#8a9a4a',
    brass: '#c9a24a'
  };

  var PALETTE = [];
  for (var key in C) {
    if (Object.prototype.hasOwnProperty.call(C, key)) PALETTE.push(C[key]);
  }

  function hexToRgb(hex) {
    return [
      parseInt(hex.slice(1, 3), 16),
      parseInt(hex.slice(3, 5), 16),
      parseInt(hex.slice(5, 7), 16)
    ];
  }

  var RGBA = PALETTE.map(hexToRgb);

  /* 5-bit RGB -> palette index. Built once; keeps every frame crisp. */
  var LUT = new Uint8Array(32 * 32 * 32);
  (function buildLut() {
    for (var i = 0; i < LUT.length; i++) {
      var r = ((i >> 10) & 31) * 255 / 31;
      var g = ((i >> 5) & 31) * 255 / 31;
      var b = (i & 31) * 255 / 31;
      var best = 0;
      var bestD = Infinity;
      for (var p = 0; p < RGBA.length; p++) {
        var dr = r - RGBA[p][0];
        var dg = g - RGBA[p][1];
        var db = b - RGBA[p][2];
        var d = dr * dr + dg * dg + db * db;
        if (d < bestD) { bestD = d; best = p; }
      }
      LUT[i] = best;
    }
  })();

  /* ---- tiny drawing helpers --------------------------------------- */
  function rect(x, y, w, h, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
  }

  function poly(points, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (var i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.fill();
  }

  function polyOutlined(points, fill, outline, width) {
    poly(points, fill);
    ctx.strokeStyle = outline;
    ctx.lineWidth = width || 1.2;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (var i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.stroke();
  }

  function circle(x, y, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  function limb(x1, y1, x2, y2, w, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = w;
    ctx.lineCap = 'square';
    ctx.lineJoin = 'miter';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  /* ---- sparse backdrops ------------------------------------------- */
  var THEMES = ['desert', 'grass', 'ice', 'storm'];
  var THEME_SECONDS = 7;
  var bgMatch = /[?&]bg=(desert|grass|ice|storm)/.exec(window.location.search);
  var forcedTheme = bgMatch ? bgMatch[1] : null;

  function drawDesert() {
    /* one tiny sun, a cactus, and dune tips at the edges */
    circle(252, 10, 5, C.gold);
    circle(252, 10, 3, C.cream);
    poly([[0, 65], [0, 61], [12, 60], [26, 62], [44, 65]], C.ochre);
    poly([[236, 65], [250, 62], [264, 61], [280, 62], [280, 65]], C.ochre);
    rect(20, 40, 5, 25, C.olive);
    rect(12, 47, 8, 4, C.olive);
    rect(12, 47, 4, 12, C.olive);
    rect(26, 45, 6, 4, C.olive);
    rect(29, 39, 3, 10, C.olive);
    rect(22, 40, 1, 18, C.teal);
    /* a second, smaller cactus on the right dune */
    rect(237, 49, 4, 16, C.olive);
    rect(231, 54, 6, 4, C.olive);
    rect(231, 54, 3, 8, C.olive);
    rect(242, 52, 6, 4, C.olive);
    rect(245, 48, 3, 8, C.olive);
    rect(239, 49, 1, 11, C.teal);
    rect(250, 62, 5, 2, C.clay);
  }

  function drawTuft(x, y, sway) {
    rect(x, y, 1, 65 - y, C.olive);
    rect(x - 2, y + 2 + sway, 1, 65 - y - 2, C.deep);
    rect(x - 1, y + 4 - sway, 1, 65 - y - 4, C.olive);
    rect(x + 1, y + 1 - sway, 1, 65 - y - 1, C.teal);
    rect(x + 3, y + 3, 1, 65 - y - 3, C.olive);
  }

  function drawGrass(t) {
    var sway = Math.floor(t * 2.5) % 2;
    poly([[0, 65], [0, 60], [16, 58], [34, 59], [52, 61], [70, 65]], C.teal);
    poly([[210, 65], [226, 61], [244, 59], [262, 58], [280, 60], [280, 65]], C.olive);
    /* a low grass bank under the riders */
    poly([[60, 65], [80, 62], [106, 61], [132, 62], [158, 61], [184, 62], [220, 65]], C.teal);
    rect(80, 62, 26, 1, C.olive);
    rect(106, 61, 26, 1, C.olive);
    rect(158, 61, 26, 1, C.olive);
    drawTuft(14, 52, sway);
    drawTuft(28, 55, 1 - sway);
    drawTuft(242, 53, sway);
    drawTuft(258, 56, 1 - sway);
    drawTuft(56, 54, sway);
    drawTuft(202, 54, 1 - sway);
    rect(34, 60, 1, 1, C.gold);
    rect(252, 59, 1, 1, C.paper);
  }

  function drawIce(t) {
    /* the frozen lake itself: a narrow parallelogram in perspective */
    poly([[70, 58], [250, 58], [230, 66], [50, 66]], C.haze);
    rect(70, 58, 180, 1, C.blue);
    rect(50, 66, 180, 1, C.blue);
    limb(70, 58, 50, 66, 1, C.blue);
    limb(250, 58, 230, 66, 1, C.blue);
    rect(84, 61, 14, 1, C.blue);
    rect(152, 63, 10, 1, C.blue);
    rect(206, 61, 12, 1, C.blue);
    limb(122, 60, 128, 65, 1, C.blue);
    /* perspective lines make the narrow parallelogram read as a plane */
    limb(120, 58, 80, 66, 1, C.blue);
    limb(190, 58, 150, 66, 1, C.blue);
    rect(94, 63, 20, 1, C.white);
    rect(190, 63, 10, 1, C.paper);
    rect(0, 62, 50, 2, C.haze);
    rect(230, 62, 50, 2, C.haze);
    rect(0, 64, 50, 1, C.blue);
    rect(230, 64, 50, 1, C.blue);
    rect(16, 60, 1, 6, C.blue);
    rect(17, 64, 5, 1, C.blue);
    rect(244, 62, 7, 1, C.blue);
    poly([[8, 62], [15, 50], [22, 62]], C.haze);
    poly([[11, 62], [15, 53], [19, 62]], C.paper);
    rect(250, 9, 7, 1, C.white);
    rect(253, 6, 1, 7, C.white);
    rect(251, 10, 3, 1, C.haze);
    if ((t % 2.5) < 0.2) rect(17, 57, 2, 1, C.white);
  }

  function drawStorm(t) {
    rect(238, 8, 20, 3, C.blue);
    rect(242, 6, 12, 2, C.haze);
    /* third cloud, top center */
    rect(150, 7, 18, 3, C.haze);
    rect(154, 5, 10, 2, C.blue);
    rect(155, 11, 1, 4, C.haze);
    rect(161, 12, 1, 3, C.blue);
    for (var i = 0; i < 3; i++) {
      var y = (t * 30 + i * 11) % 44;
      rect(8 + i * 7, 8 + y, 1, 5, C.haze);
      rect(9 + i * 7, 11 + y, 1, 3, C.blue);
      var y2 = (t * 30 + i * 13 + 17) % 44;
      rect(248 + i * 7, 8 + y2, 1, 5, C.haze);
      rect(249 + i * 7, 11 + y2, 1, 3, C.blue);
    }
    if ((t % 2.6) < 0.16) {
      rect(256, 10, 2, 7, C.brass);
      rect(253, 16, 2, 7, C.brass);
      rect(250, 22, 2, 6, C.brass);
      rect(250, 27, 3, 1, C.cream);
      rect(247, 7, 3, 1, C.cream);
    }
    rect(0, 64, 42, 1, C.blue);
    rect(238, 64, 42, 1, C.blue);
  }

  function drawTrack(name) {
    var horizon, ground, shadow;
    if (name === 'desert') { horizon = C.gold; ground = C.ochre; shadow = C.clay; }
    else if (name === 'grass') { horizon = C.teal; ground = C.olive; shadow = C.deep; }
    else if (name === 'ice') { horizon = C.haze; ground = C.blue; shadow = C.blue; }
    else { horizon = C.haze; ground = C.blue; shadow = C.blue; }

    /* faint horizon behind the riders */
    for (var x = 40; x < 240; x += 12) rect(x, 57, 7, 1, horizon);
    /* short shadows and a dashed strip under the hooves */
    rect(68, 61, 22, 1, shadow);
    rect(138, 61, 22, 1, shadow);
    if (name !== 'ice') {
      for (var x2 = 46; x2 < 236; x2 += 10) rect(x2, 62, 5, 1, ground);
    }
  }

  function drawCenter(name, t) {
    if (name === 'desert') {
      poly([[118, 58], [126, 52], [144, 52], [154, 58]], C.sand);
      rect(126, 52, 18, 1, C.cream);
      rect(86, 62, 3, 1, C.clay);
      rect(166, 62, 2, 1, C.ochre);
      rect(174, 62, 3, 1, C.clay);
      /* two tiny birds */
      rect(76, 18, 1, 1, C.deep); rect(77, 17, 1, 1, C.deep); rect(78, 18, 1, 1, C.deep);
      rect(84, 15, 1, 1, C.deep); rect(85, 14, 1, 1, C.deep); rect(86, 15, 1, 1, C.deep);
      /* second butte and dry scrub */
      poly([[198, 62], [206, 54], [220, 54], [228, 62]], C.sand);
      rect(206, 54, 14, 1, C.cream);
      rect(108, 63, 2, 1, C.clay);
      rect(118, 62, 1, 1, C.clay);
      rect(150, 63, 2, 1, C.ochre);
      rect(190, 60, 2, 2, C.olive);
      rect(204, 63, 2, 1, C.clay);
    } else if (name === 'grass') {
      rect(129, 50, 3, 12, C.deep);
      circle(131, 47, 7, C.olive);
      circle(128, 45, 4, C.teal);
      drawTuft(86, 52, 0);
      drawTuft(96, 54, 1);
      drawTuft(166, 52, 1);
      drawTuft(176, 54, 0);
      rect(90, 62, 1, 1, C.gold);
      /* second bush, more grass, one butterfly */
      rect(224, 51, 3, 11, C.deep);
      circle(226, 48, 6, C.olive);
      circle(224, 46, 3, C.teal);
      drawTuft(110, 53, 0);
      drawTuft(120, 55, 1);
      drawTuft(150, 52, 1);
      drawTuft(190, 54, 0);
      rect(180, 62, 1, 1, C.gold);
      rect(94, 30, 1, 1, C.gold); rect(96, 30, 1, 1, C.gold); rect(95, 29, 1, 1, C.paper);
    } else if (name === 'ice') {
      poly([[108, 58], [120, 46], [132, 54], [142, 44], [156, 58]], C.haze);
      poly([[116, 58], [124, 50], [132, 56], [136, 50], [144, 58]], C.paper);
      rect(86, 61, 6, 1, C.blue);
      rect(90, 62, 1, 3, C.blue);
      rect(164, 61, 5, 1, C.blue);
      rect(168, 62, 1, 3, C.blue);
      if ((t % 3.5) < 0.2) rect(120, 56, 1, 1, C.white);
      /* second peak, left snowflake, more cracks */
      poly([[196, 62], [206, 52], [216, 58], [224, 62]], C.haze);
      poly([[202, 62], [206, 56], [212, 59], [216, 62]], C.paper);
      rect(58, 11, 5, 1, C.white);
      rect(60, 8, 1, 7, C.white);
      rect(59, 12, 2, 1, C.haze);
      rect(108, 61, 4, 1, C.blue);
      rect(112, 62, 1, 2, C.blue);
      rect(118, 62, 3, 1, C.blue);
      rect(150, 61, 5, 1, C.blue);
      rect(154, 62, 1, 2, C.blue);
      rect(184, 62, 2, 1, C.white);
      rect(192, 63, 1, 1, C.haze);
    } else {
      rect(86, 7, 26, 3, C.haze);
      rect(92, 5, 14, 2, C.blue);
      for (var i = 0; i < 3; i++) {
        var y = (t * 24 + i * 11) % 30;
        rect(100 + i * 20, 18 + y, 1, 4, C.haze);
      }
      if ((t % 2.2) < 0.12) rect(104, 9, 2, 1, C.cream);
      rect(86, 62, 2, 1, C.paper);
      rect(92, 63, 1, 1, C.blue);
      rect(166, 62, 2, 1, C.paper);
      rect(172, 63, 1, 1, C.blue);
      /* extra rain behind the riders, a distant flash, puddle glints */
      for (var j = 0; j < 3; j++) {
        var yr = (t * 26 + j * 9 + 4) % 28;
        rect(78 + j * 24, 20 + yr, 1, 4, C.haze);
      }
      if ((t % 3.1) < 0.1) {
        rect(120, 52, 10, 1, C.cream);
        rect(150, 52, 6, 1, C.cream);
      }
      rect(88, 63, 3, 1, C.paper);
      rect(96, 63, 1, 1, C.blue);
      rect(160, 63, 2, 1, C.paper);
      rect(168, 63, 1, 1, C.blue);
    }
  }

  function drawBackdrop(t) {
    var name = forcedTheme || THEMES[Math.floor(t / THEME_SECONDS) % THEMES.length];
    if (name === 'desert') drawDesert();
    else if (name === 'grass') drawGrass(t);
    else if (name === 'ice') drawIce(t);
    else drawStorm(t);
    drawTrack(name);
    drawCenter(name, t);
  }

  /* ---- horses ----------------------------------------------------- */
  /* four leg poses: shoulder -> knee/hock -> fetlock -> hoof.
     The extra joint is what makes the gallop read at this size. */
  var LEG = {
    frontExt:   [44, -26, 52, -18, 60, -9, 63, 0],
    frontTuck:  [44, -26, 53, -17, 45, -10, 40, -5],
    frontUnder: [44, -26, 49, -14, 44, -8, 45, -3],
    frontTrail: [44, -26, 46, -14, 38, -7, 34, 0],
    hindExt:    [16, -24, 8, -17, 0, -9, -4, 0],
    hindUnder:  [16, -24, 10, -15, 14, -7, 18, 0],
    hindFwd:    [16, -24, 12, -14, 22, -8, 26, -3],
    hindTuck:   [16, -24, 8, -16, 19, -10, 23, -5]
  };

  var GALLOP = [
    { farHind: LEG.hindUnder, farFront: LEG.frontTuck,
      nearHind: LEG.hindExt, nearFront: LEG.frontExt },
    { farHind: LEG.hindFwd, farFront: LEG.frontUnder,
      nearHind: LEG.hindUnder, nearFront: LEG.frontTuck },
    { farHind: LEG.hindTuck, farFront: LEG.frontTrail,
      nearHind: LEG.hindFwd, nearFront: LEG.frontUnder },
    { farHind: LEG.hindExt, farFront: LEG.frontExt,
      nearHind: LEG.hindTuck, nearFront: LEG.frontTrail }
  ];

  function drawLeg(p, color, joint) {
    limb(p[0], p[1], p[2], p[3], 4.5, color);
    limb(p[2], p[3], p[4], p[5], 4.0, color);
    limb(p[4], p[5], p[6], p[7], 3.0, color);
    rect(p[2] - 1, p[3] - 1, 4, 4, joint);
    rect(p[4] - 1, p[5] - 1, 3, 3, joint);
    rect(p[6] - 2, p[7] - 1, 6, 3, C.ink);
  }

  /* one clean silhouette so the horse reads as a horse, not a stack
     of circles; the dark edge keeps it separate from the sand */
  function horsePath() {
    ctx.beginPath();
    ctx.moveTo(7, -30);
    ctx.quadraticCurveTo(4, -38, 16, -38);
    ctx.lineTo(32, -37);
    ctx.quadraticCurveTo(43, -37, 47, -32);
    ctx.lineTo(57, -43);
    ctx.lineTo(61, -42);
    ctx.lineTo(70, -38);
    ctx.lineTo(72, -35);
    ctx.lineTo(64, -33);
    ctx.lineTo(56, -37);
    ctx.lineTo(51, -26);
    ctx.quadraticCurveTo(48, -21, 41, -21);
    ctx.lineTo(20, -21);
    ctx.quadraticCurveTo(12, -21, 9, -25);
    ctx.closePath();
  }

  function drawHorse(x, base, opt, frame) {
    var pose = GALLOP[frame];

    ctx.save();
    ctx.translate(x, base);
    ctx.scale(0.8 * 0.78 * 0.6666667, 0.8 * 0.78 * 0.6666667);

    /* far legs */
    var jointFar = C.ink;
    var jointNear = opt.knots ? C.plum : C.haze;
    drawLeg(pose.farHind, opt.far, jointFar);
    drawLeg(pose.farFront, opt.far, jointFar);

    /* tail, a simple streaming flag */
    poly([[9, -36], [-6, -32], [-9, -24], [-4, -26], [-2, -30], [8, -32]], opt.mane);
    poly([[9, -34], [-3, -30], [-5, -26], [-2, -27], [0, -30], [8, -31]], opt.far);

    /* body + neck + head */
    horsePath();
    ctx.fillStyle = opt.coat;
    ctx.fill();
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1.1;
    ctx.stroke();

    /* clipped details */
    ctx.save();
    horsePath();
    ctx.clip();

    if (opt.shade) {
      ctx.fillStyle = opt.shade;
      ctx.beginPath();
      ctx.ellipse(30, -23, 17, 3, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    if (opt.spots) {
      rect(10, -34, 2, 2, C.plum);
      rect(14, -31, 3, 2, C.plum);
      rect(18, -36, 2, 1, C.plum);
      rect(22, -33, 2, 2, C.plum);
      rect(9, -28, 2, 2, C.plum);
      rect(17, -27, 2, 1, C.plum);
      rect(24, -35, 1, 1, C.plum);
      rect(12, -37, 1, 1, C.plum);
      rect(14, -25, 1, 1, C.clay);
      rect(22, -29, 1, 1, C.clay);
    }

    if (opt.rim) {
      rect(14, -38, 14, 1, C.clay);
      rect(28, -37, 9, 1, C.clay);
      rect(45, -33, 3, 1, C.clay);
    }

    ctx.restore();

    /* mane and forelock */
    poly([[46, -33], [56, -42], [60, -41], [50, -32]], opt.mane);
    rect(57, -43, 2, 2, opt.mane);

    /* ears */
    poly([[57, -43], [58, -47], [60, -43]], opt.coat);

    /* eye and nostril */
    rect(64, -38, 1, 1, C.ink);
    rect(70, -35, 1, 1, C.deep);

    /* mane knots for Valkyrie */
    if (opt.knots) {
      for (var k = 0; k < 4; k++) {
        rect(47 + k * 2.4, -34 - k * 2.3, 4, 3, C.plum);
        rect(47 + k * 2.4, -34 - k * 2.3, 4, 1, C.clay);
      }
    }

    /* near legs */
    drawLeg(pose.nearHind, opt.nearLeg || opt.coat, jointNear);
    drawLeg(pose.nearFront, opt.nearLeg || opt.coat, jointNear);

    /* saddle cloth */
    polyOutlined([[22, -38], [42, -40], [44, -29], [24, -28]], opt.cloth, C.ink, 0.8);

    /* saddle and bridle */
    rect(29, -41, 8, 3, C.brass);
    rect(29, -38, 8, 1, C.deep);
    rect(66, -39, 1, 5, C.deep);
    rect(66, -35, 6, 1, C.deep);

    /* Valkyrie's hood */
    if (opt.knots) {
      polyOutlined([[56, -42], [65, -45], [71, -40], [68, -34], [60, -34], [56, -37]], C.deep, C.ink, 0.9);
      rect(58, -44, 7, 1, C.clay);
      rect(66, -43, 4, 1, C.clay);
      rect(61, -40, 2, 1, C.paper);
      rect(67, -38, 1, 2, C.paper);
    }

    ctx.restore();

  }

  /* ---- riders ----------------------------------------------------- */
  function drawJohnny(x, base, frame) {
    ctx.save();
    ctx.translate(x, base);
    ctx.scale(0.95 * 0.6666667, 0.95 * 0.6666667);

    /* legs tucked against the saddle */
    rect(25, -37, 10, 3, C.blue);
    rect(23, -35, 3, 6, C.blue);
    rect(23, -30, 7, 3, C.blue);

    /* torso, leaning into the gallop */
    polyOutlined([[26, -38], [36, -51], [42, -49], [34, -37]], C.haze, C.ink, 1);
    rect(34, -47, 4, 2, C.paper);
    rect(33, -43, 4, 2, C.paper);
    rect(32, -39, 4, 2, C.paper);

    /* arm and rein */
    limb(39, -48, 47, -45, 2, C.haze);
    limb(47, -45, 53, -42, 1, C.deep);
    rect(51, -44, 2, 2, C.sand);

    /* bigger cartoon head */
    circle(40, -54, 5, C.sand);
    rect(34, -59, 2, 6, C.gold);
    rect(44, -59, 2, 6, C.gold);
    rect(35, -60, 10, 2, C.gold);
    rect(42, -53, 2, 2, C.white);
    rect(43, -53, 1, 1, C.ink);
    rect(45, -52, 1, 1, C.ochre);
    rect(43, -50, 1, 1, C.plum);

    /* star cap, fitted close to the head */
    ctx.fillStyle = C.white;
    ctx.beginPath();
    ctx.arc(40, -56, 5, Math.PI, 0);
    ctx.fill();
    ctx.strokeStyle = C.deep;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.arc(40, -56, 5, Math.PI, 0);
    ctx.stroke();
    rect(34, -56, 12, 2, C.white);
    rect(39, -59, 2, 2, C.blue);
    rect(38, -58, 4, 1, C.blue);
    rect(38, -57, 1, 1, C.blue);
    rect(41, -57, 1, 1, C.blue);
    rect(44, -54, 4, 1, C.deep);

    ctx.restore();
  }

  function drawGyro(x, base, frame) {
    var flutter = frame % 2;

    ctx.save();
    ctx.translate(x, base);
    ctx.scale(0.95 * 0.6666667, 0.95 * 0.6666667);

    /* olive cape streaming behind */
    polyOutlined([[37, -51], [20, -47], [10 - flutter, -38], [19, -37], [23, -42], [35, -43]], C.olive, C.ink, 0.9);
    poly([[36, -49], [25, -44], [27, -40], [35, -43]], C.deep);
    rect(17 - flutter, -41, 3, 1, C.brass);

    /* torso */
    polyOutlined([[26, -38], [36, -51], [43, -49], [35, -37]], C.deep, C.olive, 0.9);
    rect(36, -47, 1, 1, C.brass);
    rect(39, -45, 1, 1, C.brass);
    rect(35, -43, 1, 1, C.brass);
    rect(38, -41, 1, 1, C.brass);

    /* arm and steel ball */
    limb(40, -48, 48, -47, 2, C.deep);
    limb(48, -47, 54, -49, 1, C.deep);
    rect(53, -51, 2, 2, C.sand);
    circle(57, -50, 3, C.ink);
    circle(57, -50, 2, C.white);
    rect(56, -51, 1, 1, C.paper);

    /* bigger cartoon head */
    circle(41, -54, 5, C.sand);
    rect(43, -54, 2, 2, C.white);
    rect(44, -54, 1, 1, C.ink);
    rect(46, -52, 1, 1, C.ochre);
    rect(44, -50, 2, 1, C.brass);

    /* long blond hair */
    poly([[37, -59], [34, -51], [26 - flutter, -44], [31, -43], [38, -49], [45, -50], [46, -58]], C.gold);

    /* wide hat with slits and goggles */
    rect(31, -59, 19, 2, C.plum);
    rect(31, -58, 19, 1, C.ink);
    rect(35, -63, 11, 4, C.deep);
    rect(35, -60, 11, 1, C.brass);
    rect(35, -58, 11, 1, C.haze);
    rect(31, -58, 1, 1, C.ink);
    rect(33, -58, 1, 1, C.ink);
    rect(49, -58, 1, 1, C.ink);
    rect(37, -56, 2, 1, C.haze);
    rect(42, -56, 2, 1, C.haze);

    ctx.restore();
  }

  /* ---- one frame -------------------------------------------------- */
  function draw(t) {
    var frame = Math.floor(t * 9) % 4;           /* ~9 fps gallop */
    var bob = (frame === 1 || frame === 2) ? -3 : 0;

    ctx.clearRect(0, 0, W, H);

    /* logical space is 280 x 66, rendered into a 210 x 50 buffer.
       The riders are scaled down so the backdrop gets more room while
       the characters stay the same size on screen. */
    ctx.save();
    ctx.scale(0.75, 0.75);

    drawBackdrop(t);

    drawHorse(78, 60 + bob, {
      coat: C.white, far: C.plum, nearLeg: C.cream, mane: C.cream, cloth: C.blue,
      shade: C.cream, spots: true, knots: false, rim: false
    }, frame);

    drawJohnny(78 - 5.27, 60 + bob + 7.0, frame);

    drawHorse(146, 60 + bob, {
      coat: C.rust, far: C.deep, nearLeg: C.rust, mane: C.ink, cloth: C.ochre,
      shade: C.plum, spots: false, knots: true, rim: true
    }, frame);

    drawGyro(146 - 5.27, 60 + bob + 7.0, frame);

    ctx.restore();

    quantize();
  }

  /* ---- palette snap ----------------------------------------------- */
  function quantize() {
    var img = ctx.getImageData(0, 0, W, H);
    var data = img.data;
    for (var i = 0; i < data.length; i += 4) {
      if (data[i + 3] < 96) {
        data[i] = 0;
        data[i + 1] = 0;
        data[i + 2] = 0;
        data[i + 3] = 0;
        continue;
      }
      var idx = LUT[((data[i] >> 3) << 10) | ((data[i + 1] >> 3) << 5) | (data[i + 2] >> 3)];
      var c = RGBA[idx];
      data[i] = c[0];
      data[i + 1] = c[1];
      data[i + 2] = c[2];
      data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /* ---- play / pause ----------------------------------------------- */
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  var visible = true;
  var running = false;
  var rafId = null;
  var start = null;
  var lastDraw = 0;

  function loop(now) {
    rafId = null;
    if (!visible || (reduced && reduced.matches)) return;

    if (start === null) start = now;
    var t = (now - start) / 1000;

    if (now - lastDraw >= 33) {
      lastDraw = now;
      draw(t);
    }
    rafId = requestAnimationFrame(loop);
  }

  function startLoop() {
    if (running && rafId !== null) return;
    running = true;
    if (rafId === null) rafId = requestAnimationFrame(loop);
  }

  function stopLoop() {
    running = false;
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  }

  function onMotionChange() {
    if (reduced && reduced.matches) {
      stopLoop();
      draw(0);
    } else {
      startLoop();
    }
  }

  if (reduced) {
    if (reduced.addEventListener) reduced.addEventListener('change', onMotionChange);
    else if (reduced.addListener) reduced.addListener(onMotionChange);
  }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
      if (visible) startLoop();
      else stopLoop();
    }, { threshold: 0.01 }).observe(canvas);
  }

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) stopLoop();
    else if (visible) startLoop();
  });

  if (reduced && reduced.matches) {
    draw(0);
  } else {
    startLoop();
  }
})();
