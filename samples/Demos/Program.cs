using System.Numerics;
using FluentSvg;
using TinyLeastSquares;

// Generates the animated gallery for TinyLeastSquares. Every SVG bakes in a REAL solve:
// the Levenberg-Marquardt solver runs at generation time (per frame / per iteration) and
// the resulting motion is replayed as a self-contained, looping SMIL animation.
//
//   dotnet run --project samples/Demos -- <outputDir>     (defaults to ../../gallery)

var outDir = args.Length > 0
    ? args[0]
    : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "gallery"));
Directory.CreateDirectory(outDir);

const string Bg = "#0b1020";
const string Ink = "#e2e8f0";
const string Muted = "#94a3b8";
const string Grid = "#334155";
const string Good = "#34d399";
const string Warn = "#f472b6";
const string Gold = "#fbbf24";

Ik();
CurveFit();
CircleFit();
Registration();

Console.WriteLine($"Wrote demos to {outDir}");

// ---------- shared helpers ----------------------------------------------------

string FileP(string name) => Path.Combine(outDir, name);

void Background(Svg svg, float w, float h)
    => svg.AddRectangleFromTo(new Vector2(0, 0), new Vector2(w, h)).SetFill(Bg).ClearStroke();

Svg.Text Label(Svg svg, Vector2 at, string text, float size = 6f, string? color = null)
    => svg.AddText(at, text).Center().SetFill(color ?? Muted).SetFontSize(size).SetFontFamily("sans-serif");

// repeatCount='indefinite' makes each animation loop forever in a browser / README.
void Loop(params Svg.RenderItem[] anims)
{
    foreach (var a in anims) a.SetAttribute("repeatCount", "indefinite");
}

// Forward then back, so a converging demo loops smoothly instead of snapping to the start.
List<T> PingPong<T>(List<T> xs) => xs.Concat(Enumerable.Reverse(xs).Skip(1)).ToList();

string PolyD(IEnumerable<Vector2> pts)
{
    var list = pts.ToList();
    var sb = new System.Text.StringBuilder();
    sb.Append($"M{Svg.Tos(list[0].X)},{Svg.Tos(list[0].Y)}");
    for (var i = 1; i < list.Count; i++) sb.Append($" L{Svg.Tos(list[i].X)},{Svg.Tos(list[i].Y)}");
    return sb.ToString();
}

static Vector2 Rot(Vector2 v, double a)
    => new((float)(v.X * Math.Cos(a) - v.Y * Math.Sin(a)),
           (float)(v.X * Math.Sin(a) + v.Y * Math.Cos(a)));

// =============================================================================
// 1. INVERSE KINEMATICS — a 4-link arm chasing a moving target.
//    Params = 4 joint angles. Residual = end-effector - target (2 rows).
//    Analytic Jacobian; LM is re-solved every frame, warm-started from the last.
// =============================================================================
void Ik()
{
    const float w = 260, h = 200;
    var svg = new Svg(FileP("ik-target.svg"), title: "TinyLeastSquares - IK arm following a moving target");
    Background(svg, w, h);

    var basePos = new Vector2(52, 150);
    float[] len = { 52, 44, 34, 26 };
    var nLinks = len.Length;

    Vector2 Target(float t)
    {
        var a = MathF.Tau * t;
        return new Vector2(140f + 46f * MathF.Sin(a), 95f + 40f * MathF.Sin(a) * MathF.Cos(a));
    }

    Vector2[] Fk(double[] th)
    {
        var pts = new Vector2[nLinks + 1];
        pts[0] = basePos;
        double ang = 0;
        for (var i = 0; i < nLinks; i++)
        {
            ang += th[i];
            pts[i + 1] = pts[i] + new Vector2((float)(len[i] * Math.Cos(ang)), (float)(len[i] * Math.Sin(ang)));
        }
        return pts;
    }

    const int frames = 140;
    var theta = new double[nLinks];

    var joint = new List<Vector2>[nLinks + 1];
    for (var j = 0; j <= nLinks; j++) joint[j] = new List<Vector2>();
    var target = new List<Vector2>();

    // Two laps: discard the first (warm-up) so the recorded lap loops seamlessly.
    for (var lap = 0; lap < 2; lap++)
    {
        var record = lap == 1;
        for (var f = 0; f <= frames; f++)
        {
            var tgt = Target((float)f / frames);

            ResidualEvaluation Eval(double[] th)
            {
                var pts = Fk(th);
                var e = pts[nLinks];
                var res = new double[] { e.X - tgt.X, e.Y - tgt.Y };
                var jac = new double[2, nLinks];
                for (var k = 0; k < nLinks; k++)
                {
                    var d = e - pts[k];          // rotating joint k spins the tip about p_k
                    jac[0, k] = -d.Y;            // d(e.x)/d(theta_k)
                    jac[1, k] = d.X;             // d(e.y)/d(theta_k)
                }
                return new ResidualEvaluation(res, jac);
            }

            NonlinearLeastSquaresSolver.Solve(theta, Eval, new LeastSquaresOptions
            {
                MaxIterations = 40,
                InitialDamping = 1e-2,
                CostTolerance = 1e-10,
                ParamTolerance = 1e-10,
                GradientTolerance = 1e-10
            });

            if (!record) continue;
            var final = Fk(theta);
            for (var j = 0; j <= nLinks; j++) joint[j].Add(final[j]);
            target.Add(tgt);
        }
    }

    const string dur = "10s";

    // faint target path
    var trace = new List<Vector2>();
    for (var f = 0; f <= 200; f++) trace.Add(Target((float)f / 200));
    svg.AddPolyline(trace).SetFill("transparent").SetStroke(Grid, 1).SetStrokeDashArray("3 3");

    // bones
    string[] boneCols = { "#0ea5e9", "#22d3ee", "#38bdf8", "#7dd3fc" };
    for (var i = 0; i < nLinks; i++)
    {
        var line = svg.AddLine(joint[i][0], joint[i + 1][0])
                      .SetStroke(boneCols[i % boneCols.Length], 6f)
                      .SetAttribute("stroke-linecap", "round");
        Loop(svg.AddAnimateLine(line, dur, joint[i], joint[i + 1]));
    }

    // interior joints
    for (var j = 1; j < nLinks; j++)
    {
        var c = svg.AddCircle(joint[j][0], 3.5f).SetFill("#0f172a").SetStroke(Ink, 1.5f);
        var (cx, cy) = svg.AddAnimateXy(c, dur, joint[j]);
        Loop(cx, cy);
    }

    svg.AddCircle(basePos, 5f).SetFill("#1e293b").SetStroke(Ink, 2f);  // fixed base

    var ee = svg.AddCircle(joint[nLinks][0], 4.2f).SetFill(Good).ClearStroke();
    var (ex, ey) = svg.AddAnimateXy(ee, dur, joint[nLinks]);
    Loop(ex, ey);

    var tm = svg.AddCircle(target[0], 5.5f).SetFill("transparent").SetStroke(Warn, 2f);
    var (tx, ty) = svg.AddAnimateXy(tm, dur, target);
    Loop(tx, ty);

    Label(svg, new Vector2(w / 2, 16), "4-link IK - Levenberg-Marquardt reaches the moving target", 6.4f, Ink);
    Label(svg, new Vector2(w / 2, h - 8), "analytic Jacobian, re-solved & warm-started every frame", 5.2f);
    svg.SaveToFile();
}

// =============================================================================
// 2. CURVE FIT — y = a*e^(-b*x) + c fitted with a FINITE-DIFFERENCE Jacobian.
//    The model curve morphs from the initial guess to the fit over LM iterations.
// =============================================================================
void CurveFit()
{
    const float w = 280, h = 180;
    const float left = 34, right = 262, top = 30, bottom = 150;
    var svg = new Svg(FileP("curve-fit.svg"), title: "TinyLeastSquares - curve fitting (finite differences)");
    Background(svg, w, h);

    double[] truth = { 70, 0.35, 15 };
    double Model(double[] p, double x) => p[0] * Math.Exp(-p[1] * x) + p[2];

    var rng = new Random(1234);
    var dataX = new List<double>();
    var dataY = new List<double>();
    for (var i = 0; i < 14; i++)
    {
        var x = 0.3 + i * (9.4 / 13);
        dataX.Add(x);
        dataY.Add(Model(truth, x) + (rng.NextDouble() * 2 - 1) * 2.6);
    }

    double[] Residuals(double[] p)
    {
        var r = new double[dataX.Count];
        for (var i = 0; i < dataX.Count; i++) r[i] = Model(p, dataX[i]) - dataY[i];
        return r;
    }

    // Capture the parameter vector after each LM step (finite-difference Jacobian).
    var p = new double[] { 30, 1.0, 40 };
    var caps = new List<double[]> { (double[])p.Clone() };
    for (var it = 0; it < 30; it++)
    {
        var prev = (double[])p.Clone();
        NonlinearLeastSquaresSolver.Solve(p, Residuals, new LeastSquaresOptions
        {
            MaxIterations = 1,
            InitialDamping = 1e-1
        }, FiniteDifferenceScheme.Central);
        caps.Add((double[])p.Clone());
        var change = Math.Abs(p[0] - prev[0]) + Math.Abs(p[1] - prev[1]) + Math.Abs(p[2] - prev[2]);
        if (change < 1e-6) break;
    }

    // Auto-fit the value axis to everything we will draw.
    const int samples = 48;
    double vmin = double.MaxValue, vmax = double.MinValue;
    void Track(double v) { vmin = Math.Min(vmin, v); vmax = Math.Max(vmax, v); }
    foreach (var y in dataY) Track(y);
    foreach (var cap in caps)
        for (var s = 0; s <= samples; s++) Track(Model(cap, 10.0 * s / samples));
    var pad = (vmax - vmin) * 0.08;
    vmin -= pad; vmax += pad;

    float Sx(double x) => (float)(left + x / 10.0 * (right - left));
    float Sy(double v) => (float)(bottom - (v - vmin) / (vmax - vmin) * (bottom - top));

    List<Vector2> Curve(double[] cap)
    {
        var pts = new List<Vector2>();
        for (var s = 0; s <= samples; s++)
        {
            var x = 10.0 * s / samples;
            pts.Add(new Vector2(Sx(x), Sy(Model(cap, x))));
        }
        return pts;
    }

    // frame
    svg.AddRectangleFromTo(new Vector2(left, top), new Vector2(right, bottom))
       .SetFill("transparent").SetStroke(Grid, 1);

    // the morphing fit curve (animate the path 'd' across iterations)
    var seq = PingPong(caps);
    var fit = svg.AddPath(Curve(caps[0])).SetFill("transparent").SetStroke("#22d3ee", 2.5f);
    fit.SetAttribute("stroke-linecap", "round").SetAttribute("stroke-linejoin", "round");
    fit.SetId("fitcurve");
    Loop(new Svg.Animate(svg, fit, "d", "6s", seq.Select(Curve).Select(PolyD)));

    // data points (fixed)
    for (var i = 0; i < dataX.Count; i++)
        svg.AddCircle(new Vector2(Sx(dataX[i]), Sy(dataY[i])), 2.6f).SetFill(Gold).ClearStroke();

    Label(svg, new Vector2(w / 2, 16), "Curve fit:  y = a e^(-b x) + c", 6.6f, Ink);
    Label(svg, new Vector2(w / 2, h - 6), "finite-difference Jacobian - no derivatives supplied", 5.2f);
    svg.SaveToFile();
}

// =============================================================================
// 3. CIRCLE FIT — center + radius fitted to scattered points (analytic Jacobian).
//    residual_i = |p_i - center| - r.  The circle animates from guess to fit.
// =============================================================================
void CircleFit()
{
    const float w = 280, h = 220;
    var svg = new Svg(FileP("circle-fit.svg"), title: "TinyLeastSquares - circle fitting");
    Background(svg, w, h);

    var trueC = new Vector2(150, 112);
    const float trueR = 58;
    var rng = new Random(7);
    var pts = new List<Vector2>();
    for (var i = 0; i < 18; i++)
    {
        var a = -2.5f + 4.4f * i / 17f;                       // a partial arc, not a full circle
        var rr = trueR + (float)(rng.NextDouble() * 2 - 1) * 5f;
        pts.Add(trueC + new Vector2(rr * MathF.Cos(a), rr * MathF.Sin(a)));
    }

    ResidualEvaluation Eval(double[] q)
    {
        var cx = q[0]; var cy = q[1]; var r = q[2];
        var res = new double[pts.Count];
        var jac = new double[pts.Count, 3];
        for (var i = 0; i < pts.Count; i++)
        {
            double dx = pts[i].X - cx, dy = pts[i].Y - cy;
            var d = Math.Sqrt(dx * dx + dy * dy);
            if (d < 1e-9) d = 1e-9;
            res[i] = d - r;
            jac[i, 0] = -dx / d;
            jac[i, 1] = -dy / d;
            jac[i, 2] = -1;
        }
        return new ResidualEvaluation(res, jac);
    }

    var q = new double[] { 108, 78, 24 };                      // deliberately off
    var caps = new List<double[]> { (double[])q.Clone() };
    for (var it = 0; it < 30; it++)
    {
        var prev = (double[])q.Clone();
        NonlinearLeastSquaresSolver.Solve(q, Eval, new LeastSquaresOptions { MaxIterations = 1, InitialDamping = 1e-1 });
        caps.Add((double[])q.Clone());
        if (Math.Abs(q[0] - prev[0]) + Math.Abs(q[1] - prev[1]) + Math.Abs(q[2] - prev[2]) < 1e-6) break;
    }

    var seq = PingPong(caps);
    var centers = seq.Select(c => new Vector2((float)c[0], (float)c[1])).ToList();
    var radii = seq.Select(c => (float)c[2]).ToList();

    // fitted circle
    var circle = svg.AddCircle(centers[0], radii[0]).SetFill("transparent").SetStroke("#38bdf8", 2.5f);
    var (ccx, ccy) = svg.AddAnimateXy(circle, "5s", centers);
    var cr = svg.AddAnimate(circle, "r", "5s", radii);
    Loop(ccx, ccy, cr);

    // center marker
    var dot = svg.AddCircle(centers[0], 3f).SetFill(Warn).ClearStroke();
    var (dx2, dy2) = svg.AddAnimateXy(dot, "5s", centers);
    Loop(dx2, dy2);

    // data points (fixed)
    foreach (var pt in pts) svg.AddCircle(pt, 2.8f).SetFill(Gold).ClearStroke();

    Label(svg, new Vector2(w / 2, 16), "Circle fit:  minimize  S ( |p_i - c| - r )^2", 6.4f, Ink);
    Label(svg, new Vector2(w / 2, h - 8), "3 params (cx, cy, r), analytic Jacobian", 5.2f);
    svg.SaveToFile();
}

// =============================================================================
// 4. RIGID REGISTRATION — align a shape onto a noisy target by solving for a
//    rotation + translation (theta, tx, ty). Analytic Jacobian.
// =============================================================================
void Registration()
{
    const float w = 300, h = 190;
    var svg = new Svg(FileP("registration.svg"), title: "TinyLeastSquares - rigid registration");
    Background(svg, w, h);

    // template (a little house), centered on its own origin
    Vector2[] shape =
    {
        new(-24, 20), new(24, 20), new(24, -6), new(0, -26), new(-24, -6)
    };
    int[][] edges = { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 0 } };
    var v = shape.Length;

    // ground-truth target: rotate + translate the template, add a little noise
    const double trueAngle = 0.70;
    var trueT = new Vector2(206, 96);
    var rng = new Random(3);
    var qpts = new Vector2[v];
    for (var i = 0; i < v; i++)
        qpts[i] = Rot(shape[i], trueAngle) + trueT
                  + new Vector2((float)(rng.NextDouble() * 2 - 1) * 3f, (float)(rng.NextDouble() * 2 - 1) * 3f);

    ResidualEvaluation Eval(double[] pr)
    {
        double ang = pr[0]; float tx = (float)pr[1], ty = (float)pr[2];
        var res = new double[2 * v];
        var jac = new double[2 * v, 3];
        for (var i = 0; i < v; i++)
        {
            var rp = Rot(shape[i], ang);                       // rotated (pre-translation)
            var s = rp + new Vector2(tx, ty);
            res[2 * i] = s.X - qpts[i].X;
            res[2 * i + 1] = s.Y - qpts[i].Y;
            jac[2 * i, 0] = -rp.Y; jac[2 * i, 1] = 1; jac[2 * i, 2] = 0;      // d/dtheta, d/dtx, d/dty
            jac[2 * i + 1, 0] = rp.X; jac[2 * i + 1, 1] = 0; jac[2 * i + 1, 2] = 1;
        }
        return new ResidualEvaluation(res, jac);
    }

    var pr0 = new double[] { 0.0, 86, 104 };                    // start unrotated, over on the left
    var caps = new List<double[]> { (double[])pr0.Clone() };
    var pr = pr0;
    for (var it = 0; it < 40; it++)
    {
        var prev = (double[])pr.Clone();
        NonlinearLeastSquaresSolver.Solve(pr, Eval, new LeastSquaresOptions { MaxIterations = 1, InitialDamping = 1e-1 });
        caps.Add((double[])pr.Clone());
        if (Math.Abs(pr[0] - prev[0]) + Math.Abs(pr[1] - prev[1]) + Math.Abs(pr[2] - prev[2]) < 1e-7) break;
    }

    Vector2[] Transform(double[] c)
    {
        var s = new Vector2[v];
        for (var i = 0; i < v; i++) s[i] = Rot(shape[i], c[0]) + new Vector2((float)c[1], (float)c[2]);
        return s;
    }

    var seq = PingPong(caps);
    var frames = seq.Select(Transform).ToList();               // frames[f][vertex]

    // target shape (fixed): outline + dots
    for (var e = 0; e < edges.Length; e++)
        svg.AddLine(qpts[edges[e][0]], qpts[edges[e][1]]).SetStroke(Grid, 2f).SetStrokeDashArray("4 3");
    foreach (var pt in qpts) svg.AddCircle(pt, 2.6f).SetFill(Muted).ClearStroke();

    const string dur = "6s";

    // moving source shape: animated edges + vertices
    foreach (var e in edges)
    {
        var starts = frames.Select(fr => fr[e[0]]).ToList();
        var ends = frames.Select(fr => fr[e[1]]).ToList();
        var line = svg.AddLine(starts[0], ends[0]).SetStroke("#38bdf8", 3f).SetAttribute("stroke-linecap", "round");
        Loop(svg.AddAnimateLine(line, dur, starts, ends));
    }
    for (var i = 0; i < v; i++)
    {
        var path = frames.Select(fr => fr[i]).ToList();
        var c = svg.AddCircle(path[0], 3f).SetFill(Good).ClearStroke();
        var (ax, ay) = svg.AddAnimateXy(c, dur, path);
        Loop(ax, ay);
    }

    Label(svg, new Vector2(w / 2, 16), "Rigid registration - solve rotation + translation", 6.4f, Ink);
    Label(svg, new Vector2(w / 2, h - 8), "3 params (theta, tx, ty), aligning onto the dashed target", 5.2f);
    svg.SaveToFile();
}
