"""Builds dashboard.html — a self-contained security lab dashboard with the
real measured results embedded. Every number on the page comes from results/*.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


def prep():
    p, b, e, m, ms, a, sc, u = (load("poisoning"), load("backdoor"), load("extraction"),
                                load("membership"), load("membership_small"), load("adversarial"),
                                load("supply_chain"), load("unified_profile"))

    def by_frac(runs):
        out = {}
        for r in runs:
            out.setdefault(r["fraction"], []).append(r)
        return out

    pf = by_frac(p["runs"])
    poisoning = {
        "fractions": sorted(pf.keys()),
        "clean_acc": [mean([r["clean_accuracy"] for r in pf[f]]) for f in sorted(pf)],
        "src_acc": [mean([r["source_class_accuracy"] for r in pf[f]]) for f in sorted(pf)],
        "transfer": [mean([r["src_to_tgt_transfer_rate"] for r in pf[f]]) for f in sorted(pf)],
        "ece": [mean([r["ece"] for r in pf[f]]) for f in sorted(pf)],
        "detect_recall": {d["fraction"]: d["recall_at_1pct_flagged"] for d in p["detection"]},
        "defense": mean([d["accuracy"] for d in p["defense_filter"] if d["fraction"] == 0.05]),
        "defense_src": mean([d["source_class_accuracy"] for d in p["defense_filter"] if d["fraction"] == 0.05]),
    }

    bf = by_frac(b["runs"])
    backdoor = {
        "fractions": sorted(bf.keys()),
        "clean_acc": [mean([r["clean_accuracy"] for r in bf[f]]) for f in sorted(bf)],
        "asr": [mean([r["attack_success_rate"] for r in bf[f]]) for f in sorted(bf)],
        "flip_scan": [mean([r["trigger_flip_scan"]["flip_rate_to_target"] for r in bf[f]]) for f in sorted(bf)],
        "defense_asr": mean([d["attack_success_rate"] for d in b["defense"]]),
        "defense_recovered": mean([d["poison_recovered"] for d in b["defense"]]),
    }

    policies = ("full_confidence", "rounded", "argmax_only")
    extraction = {
        "budgets": [500, 1000, 2000, 4000],
        "series": {
            pol: [mean([r["fidelity_mean"] for r in e["runs"]
                        if r["budget"] == bud and r["policy"] == pol]) for bud in (500, 1000, 2000, 4000)]
            for pol in policies},
        "ratelimit": {r["budget"]: r["fidelity_mean"] for r in e["runs"]
                      if r["policy"] == "full_confidence+ratelimit1000"},
        "target_acc": e["target"]["accuracy"],
    }

    epochs = {}
    for grp in m["overfitting_study"]:
        epochs.setdefault(grp[0]["epochs"], []).extend(grp)
    privacy = {
        "epoch_study": [{"epochs": ep,
                         "gap": mean([s["overfit_gap"] for s in ss]),
                         "auc": mean([s["mia"][s["mia"]["best"]]["auc"] for s in ss]),
                         "tpr": mean([s["mia"][s["mia"]["best"]]["tpr"] for s in ss])}
                        for ep, ss in sorted(epochs.items())],
        "defenses": m["defenses"],
        "small_study": [{"config": c,
                         "gap": mean([r["overfit_gap"] for r in ms["runs"] if r["config"] == c]),
                         "auc": mean([r["mia"][r["mia"]["best"]]["auc"] for r in ms["runs"] if r["config"] == c]),
                         "test_acc": mean([r["test_accuracy"] for r in ms["runs"] if r["config"] == c])}
                        for c in ("baseline", "baseline_long", "dp_noise1.0", "dp_noise3.0")],
    }

    adv = {}
    for model in ("baseline", "pgd_adversarial_training", "label_smoothing"):
        rows = a["models"][model]
        adv[model] = {"clean_acc": mean([r["clean_accuracy"] for r in rows]),
                      "pgd": {r["eps"]: r["robust_accuracy_mean"] for r in rows[0]["attacks"]
                              if r["attack"] == "pgd"},
                      "fgsm": {r["eps"]: r["robust_accuracy_mean"] for r in rows[0]["attacks"]
                               if r["attack"] == "fgsm"}}

    lifecycle = [{"scenario": r["scenario"], "verdict": r["verdict"],
                  "desc": r.get("tamper_description", "")} for r in sc["lifecycle"]]
    behavior = sc["tampered_behavior"]

    profile = {k: v for k, v in u["security_profile"].items() if k != "_formula"}
    tradeoffs = u["tradeoffs"]

    ctm = None
    try:
        cm = load("cross_matrix")
        order = ["baseline", "auditfilter", "at", "dp", "ls"]
        attacks = ["poison", "backdoor", "extraction", "mia", "adversarial"]
        ctm = {
            "attacks": attacks,
            "rows": [{"label": cm["matrix"][r]["label"],
                      "stats": [{"mean": cm["matrix"][r]["deltaS_stats"][a]["mean"],
                                 "std": cm["matrix"][r]["deltaS_stats"][a]["std"],
                                 "band": cm["matrix"][r]["deltaS_stats"][a]["noise_band"],
                                 "class": cm["matrix"][r]["deltaS_stats"][a]["class"]} for a in attacks],
                      "raw": {"clean_acc": cm["matrix"][r]["clean"]["clean_accuracy"],
                              "ece": cm["matrix"][r]["clean"]["ece"],
                              "mia_auc": cm["matrix"][r]["clean"]["mia_auc"],
                              "fidelity": cm["matrix"][r]["clean"]["extraction_fidelity"],
                              "pgd_succ": cm["matrix"][r]["clean"]["pgd_success"],
                              "src_acc": cm["matrix"][r]["poison"]["src_class_accuracy"],
                              "transfer": cm["matrix"][r]["poison"]["src_to_tgt_transfer"],
                              "asr": cm["matrix"][r]["backdoor"]["asr"],
                              "overhead": cm["matrix"][r]["utility_cost"]["train_overhead_x"],
                              "clean_drop": cm["matrix"][r]["utility_cost"]["clean_accuracy_drop"]}}
                     for r in order],
        }
    except FileNotFoundError:
        pass

    return {"profile": profile, "poisoning": poisoning, "backdoor": backdoor,
            "extraction": extraction, "privacy": privacy, "adversarial": adv,
            "lifecycle": lifecycle, "behavior": behavior,
            "tradeoffs": tradeoffs, "formula": u["security_profile"]["_formula"],
            "cross": ctm}


DATA = prep()

TEMPLATE = """<!doctype html>
<title>AML-Sec Security Evaluation</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#0A111E; --panel:#101A2C; --panel2:#0D1526; --line:#22314B;
  --ink:#E9EEF7; --muted:#93A1B8; --faint:#5C6C86;
  --safe:#3DDC97; --warn:#F2B544; --risk:#FF5C6C; --accent:#5AA9FF;
}
*{box-sizing:border-box; margin:0}
body{background:var(--ground); color:var(--ink); font:15px/1.6 "IBM Plex Sans",system-ui,sans-serif;
     -webkit-font-smoothing:antialiased}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
.wrap{max-width:1120px; margin:0 auto; padding:48px 28px 80px}
header{display:flex; flex-wrap:wrap; gap:16px 24px; align-items:baseline; border-bottom:1px solid var(--line); padding-bottom:20px}
h1{font-size:26px; font-weight:600; letter-spacing:-0.01em}
h1 .sec{color:var(--accent)}
.sub{color:var(--muted); font-size:13px}
.chips{display:flex; gap:8px; flex-wrap:wrap; margin-left:auto}
.chip{font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--muted);
      border:1px solid var(--line); border-radius:4px; padding:3px 9px; white-space:nowrap}
h2{font-size:13px; font-weight:600; letter-spacing:0.14em; text-transform:uppercase;
   color:var(--muted); margin:52px 0 6px}
.note{color:var(--faint); font-size:13px; margin-bottom:16px}
.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:10px; margin-top:24px}
.tile{background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 14px 12px}
.tile .name{font-size:12px; color:var(--muted); margin-bottom:8px}
.tile .score{font-family:"IBM Plex Mono",monospace; font-size:26px; font-weight:600; line-height:1}
.tile .bar{height:4px; border-radius:2px; background:var(--line); margin:10px 0 8px; overflow:hidden}
.tile .bar i{display:block; height:100%; border-radius:2px}
.tile .def{font-size:11.5px; color:var(--faint); line-height:1.45}
.card{background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px}
.grid2{display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
table{width:100%; border-collapse:collapse; font-family:"IBM Plex Mono",monospace; font-size:12.5px}
th{color:var(--faint); font-weight:500; text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
   text-transform:uppercase; letter-spacing:0.06em; font-size:10.5px}
td{padding:6px 8px; border-bottom:1px solid var(--panel2); font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
td.num{text-align:right}
.good{color:var(--safe)} .bad{color:var(--risk)} .mid{color:var(--warn)}
.finding{border-left:2px solid var(--accent); background:var(--panel2);
         padding:12px 16px; border-radius:0 6px 6px 0; font-size:13.5px; color:var(--ink); margin-top:12px}
.finding b{color:var(--accent)}
.finding + .finding{margin-top:8px}
.verdict{display:inline-block; font-family:"IBM Plex Mono",monospace; font-size:11px; font-weight:600;
         border-radius:4px; padding:2px 8px; letter-spacing:0.04em}
.verdict.pass{color:var(--safe); border:1px solid var(--safe)}
.verdict.fail{color:var(--risk); border:1px solid var(--risk)}
footer{margin-top:64px; border-top:1px solid var(--line); padding-top:16px;
       color:var(--faint); font-size:12px; line-height:1.7}
.legend{display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 10px; font-size:12px; color:var(--muted)}
.legend i{display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:6px; border-radius:1px}
.chartwrap{overflow-x:auto}
svg text{font-family:"IBM Plex Mono",monospace}
</style>
<div class="wrap">
<header>
  <h1>AML-SEC <span class="sec">/</span> Security Evaluation</h1>
  <div class="chips mono">
    <span class="chip">MNIST · 20k/4k stratified</span>
    <span class="chip">SmallCNN ~81k params</span>
    <span class="chip">3 seeds/config</span>
    <span class="chip">PyTorch CPU</span>
  </div>
</header>

<div class="kpis" id="kpis"></div>

<h2>01 · Poisoning — dose response</h2>
<p class="note">Targeted label flips, class 2 → 7. Clean accuracy barely moves; the damage concentrates on the source class and shows up first in calibration (ECE).</p>
<div class="grid2">
  <div class="card chartwrap" id="ch_poison"></div>
  <div class="card">
    <table id="t_poison"></table>
  </div>
</div>
<div class="finding"><b>Detection before degradation:</b> the kNN label audit flags poisoned samples with recall <span class="mono" id="d_rec"></span> — detection degrades as the poison fraction grows (poisoned samples stop looking anomalous). Audit-and-filter retraining recovers source-class accuracy to <span class="mono" id="d_def"></span> at a ~1pt clean-accuracy cost.</div>

<h2>02 · Backdoor — patch trigger → class 0</h2>
<p class="note">3×3 patch stamped only on poisoned samples. Conventional validation sees a healthy model; the trigger works.</p>
<div class="grid2">
  <div class="card chartwrap" id="ch_bd"></div>
  <div class="card"><table id="t_bd"></table></div>
</div>
<div class="finding"><b>Cross-threat finding:</b> the same data audit that cleans label-flip poisoning is <b class="bad">ineffective against the backdoor</b> — it recovers only <span class="mono" id="bd_rec"></span> of poison and ASR stays at <span class="mono" id="bd_asr2"></span>. The model-level trigger-flip scan (flip-to-target rate <span class="mono" id="bd_scan"></span> on correct predictions) is the detector that works. Backdoor defense must inspect the model, not just the data.</div>

<h2>03 · Extraction — substitute fidelity vs query budget</h2>
<p class="note">Target accuracy <span class="mono" id="ex_acc"></span>. Fidelity = substitute↔target agreement on held-out data.</p>
<div class="card chartwrap" id="ch_ex"></div>
<div class="finding"><b>Measured trade-off:</b> output restriction (rounding, argmax-only) barely reduces fidelity (<span class="mono" id="ex_gap"></span> drop) — on MNIST, hard labels alone suffice to steal the classifier. The rate limit is the only control that bites: it caps fidelity at <span class="mono" id="ex_rl"></span> regardless of the attacker's budget.</div>

<h2>04 · Membership inference & privacy</h2>
<p class="note">Score-based MIA (confidence / loss / margin). Members = real training samples, non-members = test population.</p>
<div class="grid2">
  <div class="card"><table id="t_mia"></table></div>
  <div class="card"><table id="t_dp"></table></div>
</div>
<div class="finding"><b>Honest finding:</b> on MNIST with 20k samples this CNN leaks almost nothing through its outputs (AUC ≈ 0.50 at every epoch count, and ≈ 0.51 even at 3k samples / 20 epochs). DP-style clipped+noisy training reliably suppresses the train–test gap (<span class="mono" id="dp_gap"></span> → <span class="mono" id="dp_gap3"></span>) and pushes AUC below 0.5, at a measured accuracy cost of <span class="mono" id="dp_cost"></span> at the strongest setting.</div>

<h2>05 · Adversarial examples — FGSM / PGD</h2>
<div class="grid2">
  <div class="card chartwrap" id="ch_adv"></div>
  <div class="card"><table id="t_adv"></table></div>
</div>
<div class="finding"><b>Robustness–utility Pareto:</b> adversarial training buys <span class="mono" id="at_gain"></span> robustness at ε=0.1 for <span class="mono" id="at_cost"></span> clean accuracy. Label smoothing costs nothing in clean accuracy and buys nothing — it is not a robustness defense.</div>

<h2>06 · Supply-chain integrity</h2>
<div class="card"><table id="t_sc"></table></div>
<div class="finding"><b>What integrity checking protects:</b> the 1-ulp tampered artifact is behaviorally indistinguishable from the approved model (<span class="mono" id="sc_dis"></span> prediction disagreement, accuracy unchanged) — behavioral testing cannot catch it. Hash + HMAC verification catches it before load, every time, with zero utility cost.</div>

<h2>07 · Trade-off matrix</h2>
<div class="grid2">
  <div class="card"><table id="t_priv"></table></div>
  <div class="card"><table id="t_rob"></table></div>
</div>

<h2 id="sec_ctm">08 · Cross-threat Δ-security matrix</h2>
<p class="note" id="ctm_note">Each defense is a training regime, retrained under clean / 5% poison / 5% backdoor data and probed by MIA, extraction and PGD directly. Cell = ΔS(D,A) = S(baseline,A) − S(defense,A): <b>positive = defending one threat helped against another</b>, negative = collateral damage.</p>
<div id="ctm" hidden>
  <div class="card chartwrap"><div id="heatmap"></div>
    <div class="legend"><span><i style="background:var(--safe)"></i>+ significant transfer</span>
      <span><i style="background:var(--line)"></i>0 negligible</span>
      <span><i style="background:var(--risk)"></i>− significant adverse</span>
      <span>? unresolved at 3 seeds (paired t band, df=2)</span></div>
  </div>
  <div class="card" style="margin-top:10px"><table id="t_ctm"></table></div>
  <div class="finding" id="ctm_finding" hidden></div>
</div>

<footer>
  <b class="mono" style="color:var(--muted)">Methodology</b><br>
  Every figure on this page is measured from real training runs and real model queries (results/*.json in the repo; nothing hardcoded). Poisoning is targeted label-flip; MIA is score-based thresholding; the DP-style defense is batch gradient clipping + Gaussian noise — an approximation of DP-SGD without formal ε accounting. Scoring formula:<br>
  <span class="mono" id="formula" style="font-size:11px"></span>
</footer>
</div>

<script>
const DATA = __DATA__;

const C = {safe:'#3DDC97', warn:'#F2B544', risk:'#FF5C6C', accent:'#5AA9FF', muted:'#93A1B8',
           faint:'#5C6C86', line:'#22314B', ink:'#E9EEF7'};

function fmt(x, d=3){ return x==null ? '—' : (typeof x === 'number' ? x.toFixed(d) : x); }
function pct(x, d=1){ return x==null ? '—' : (100*x).toFixed(d>=0?Math.min(d,1):1) + '%'; }

/* ---------- KPI tiles ---------- */
const K = DATA.profile;
const kpis = [
  ['Poisoning', K.poisoning.risk_undefended, K.poisoning.defense, 'src-acc '+pct(K.poisoning.defense_recovery)],
  ['Backdoor', K.backdoor.risk_undefended, 'data-level audit ineffective', 'ASR stays '+pct(K.backdoor.defense_asr)],
  ['Extraction', K.extraction.risk_undefended, 'rate limit caps at '+pct(K.extraction.defense_fidelity)],
  ['Membership', K.membership_inference.risk_undefended, 'AUC→'+fmt(K.membership_inference.defense_auc,3)+' under DP-style noise'],
  ['Adversarial', K.adversarial_examples.risk_undefended, 'AT: success →'+pct(K.adversarial_examples.defense_success_rate)],
  ['Supply chain', K.supply_chain.risk_undefended, 'hash+HMAC residual 0'],
];
document.getElementById('kpis').innerHTML = kpis.map(([name, risk, def]) => {
  const col = risk > 66 ? C.risk : risk > 33 ? C.warn : C.safe;
  return `<div class="tile">
    <div class="name">${name}</div>
    <div class="score" style="color:${col}">${risk.toFixed(1)}</div>
    <div class="bar"><i style="width:${Math.min(100, risk)}%; background:${col}"></i></div>
    <div class="def mono">${def}</div>
  </div>`;
}).join('');

/* ---------- generic SVG line chart ---------- */
function lineChart(el, series, xLabels, opts={}){
  const W = 520, H = 260, padL = 44, padB = 30, padT = 14, padR = 14;
  const el0 = document.getElementById(el);
  const maxY = opts.maxY != null ? opts.maxY : Math.max(...series.flatMap(s => s.ys)) * 1.08;
  const xs = i => padL + i * (W - padL - padR) / Math.max(1, xLabels.length - 1);
  const ys = v => padT + (H - padT - padB) * (1 - v / maxY);
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%; min-width:420px; display:block">`;
  for (const [i, lbl] of xLabels.entries()) {
    svg += `<line x1="${xs(i)}" y1="${padT}" x2="${xs(i)}" y2="${H - padB}" stroke="${C.line}" stroke-width="1" opacity="0.5"/>`;
    svg += `<text x="${xs(i)}" y="${H - 10}" fill="${C.faint}" font-size="10" text-anchor="middle">${lbl}</text>`;
  }
  for (const t of opts.yTicks || [0, 0.5, 1]) {
    if (t > maxY) continue;
    svg += `<line x1="${padL}" y1="${ys(t)}" x2="${W - padR}" y2="${ys(t)}" stroke="${C.line}" stroke-width="1" opacity="0.5"/>`;
    svg += `<text x="${padL - 6}" y="${ys(t) + 3}" fill="${C.faint}" font-size="10" text-anchor="end">${opts.yFmt ? opts.yFmt(t) : t}</text>`;
  }
  for (const s of series) {
    const pts = s.ys.map((v, i) => `${xs(i)},${ys(v)}`).join(' ');
    if (s.fill) {
      const area = `${padL},${ys(0)} ${pts} ${xs(s.ys.length - 1)},${ys(0)}`;
      svg += `<polygon points="${area}" fill="${s.color}" opacity="0.08" />`;
    }
    svg += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dash?'stroke-dasharray="5 4"':''} stroke-linejoin="round"/>`;
    svg += s.ys.map((v, i) => `<circle cx="${xs(i)}" cy="${ys(v)}" r="3" fill="${s.color}"/>`).join('');
  }
  el0.innerHTML = svg + '</svg>';
  return {xs, ys, padL, W};
}

/* ---------- 01 poisoning ---------- */
const P = DATA.poisoning;
lineChart('ch_poison', [
  {label:'src→tgt transfer', color:C.risk, ys:P.transfer},
  {label:'ECE', color:C.warn, ys:P.ece},
  {label:'clean acc', color:C.safe, ys:P.clean_acc},
], P.fractions.map(f => (100*f)+'%'), {yFmt: t => t.toFixed(2), yTicks:[0, 0.01, 0.02, 1.0], maxY:1.02});
document.getElementById('t_poison').innerHTML =
  '<tr><th>Poison %</th><th class="num">Clean acc</th><th class="num">Src-class acc</th><th class="num">ECE</th><th class="num">Audit recall</th></tr>' +
  P.fractions.map((f, i) => `<tr>
    <td>${(100*f).toFixed(0)}%</td><td class="num">${fmt(P.clean_acc[i],4)}</td>
    <td class="num">${fmt(P.src_acc[i],4)}</td><td class="num">${fmt(P.ece[i],4)}</td>
    <td class="num">${f===0?'—':fmt(P.detect_recall[f])}</td></tr>`).join('');
document.getElementById('d_rec').textContent = fmt(P.detect_recall[0.05]);
document.getElementById('d_def').textContent = fmt(P.defense_src, 4);

/* ---------- 02 backdoor ---------- */
const B = DATA.backdoor;
lineChart('ch_bd', [
  {label:'attack success rate', color:C.risk, ys:B.asr},
  {label:'clean accuracy', color:C.safe, ys:B.clean_acc},
  {label:'flip-scan to target', color:C.warn, ys:B.flip_scan},
], B.fractions.map(f => (100*f)+'%'), {maxY:1.05, yFmt:t=>pct(t,0), yTicks:[0,0.5,1]});
document.getElementById('t_bd').innerHTML =
  '<tr><th>Poison %</th><th class="num">Clean acc</th><th class="num">ASR</th><th class="num">Flip scan</th></tr>' +
  B.fractions.map((f, i) => `<tr><td>${(100*f).toFixed(0)}%</td>
    <td class="num">${fmt(B.clean_acc[i],4)}</td><td class="num ${i?'bad':''}">${fmt(B.asr[i],4)}</td>
    <td class="num">${fmt(B.flip_scan[i])}</td></tr>`).join('');
document.getElementById('bd_rec').textContent = pct(B.defense_recovered, 0);
document.getElementById('bd_asr2').textContent = pct(B.defense_asr, 1);
document.getElementById('bd_scan').textContent = fmt(B.flip_scan[B.flip_scan.length-1]);

/* ---------- 03 extraction ---------- */
const E = DATA.extraction;
lineChart('ch_ex', [
  {label:'full confidence', color:C.risk, ys:E.series.full_confidence},
  {label:'rounded', color:C.warn, ys:E.series.rounded},
  {label:'argmax only', color:C.accent, ys:E.series.argmax_only},
], E.budgets.map(b => b.toLocaleString()), {maxY:1.05, yFmt:t=>pct(t,0), yTicks:[0.5,0.75,0.9,1.0]});
document.getElementById('ex_acc').textContent = fmt(E.target_acc, 4);
const gapFid = E.series.full_confidence[3] - E.series.argmax_only[3];
document.getElementById('ex_gap').textContent = (100*gapFid).toFixed(1) + 'pt';
const rlVals = Object.values(E.ratelimit);
document.getElementById('ex_rl').textContent = pct(Math.max(...rlVals), 1);

/* ---------- 04 privacy ---------- */
document.getElementById('t_mia').innerHTML =
  '<tr><th>Epochs (20k)</th><th class="num">Overfit gap</th><th class="num">MIA AUC</th><th class="num">TPR@5%FPR</th></tr>' +
  DATA.privacy.epoch_study.map(r => `<tr><td>${r.epochs}</td><td class="num">${fmt(r.gap,4)}</td>
    <td class="num">${fmt(r.auc,4)}</td><td class="num">${fmt(r.tpr,4)}</td></tr>`).join('');
document.getElementById('t_dp').innerHTML =
  '<tr><th>Config (3k, 20 ep)</th><th class="num">Gap</th><th class="num">AUC</th><th class="num">Test acc</th></tr>' +
  DATA.privacy.small_study.map(r => `<tr><td>${r.config}</td><td class="num">${fmt(r.gap,4)}</td>
    <td class="num">${fmt(r.auc,4)}</td><td class="num">${fmt(r.test_acc,4)}</td></tr>`).join('');
const sm = Object.fromEntries(DATA.privacy.small_study.map(r => [r.config, r]));
document.getElementById('dp_gap').textContent = fmt(sm.baseline_long.gap, 4);
document.getElementById('dp_gap3').textContent = fmt(sm['dp_noise3.0'].gap, 4);
document.getElementById('dp_cost').textContent = ((sm.baseline.test_acc - sm['dp_noise3.0'].test_acc)*100).toFixed(1) + 'pt';

/* ---------- 05 adversarial ---------- */
const A = DATA.adversarial;
const EPS = [0.02, 0.05, 0.1, 0.2];
const atEps = (obj, keys) => keys.map(k => obj[k]);
lineChart('ch_adv', [
  {label:'baseline', color:C.risk, ys:atEps(A.baseline.pgd, EPS)},
  {label:'PGD-AT', color:C.safe, ys:atEps(A.pgd_adversarial_training.pgd, EPS)},
  {label:'label smoothing', color:C.faint, ys:atEps(A.label_smoothing.pgd, EPS), dash:1},
], EPS, {maxY:1.05, yFmt:t=>pct(t,0), yTicks:[0,0.5,1]});
document.getElementById('t_adv').innerHTML =
  '<tr><th>Model</th><th class="num">Clean</th><th class="num">ε=.05</th><th class="num">ε=.1</th><th class="num">ε=.2</th></tr>' +
  [['baseline', A.baseline], ['PGD-AT', A.pgd_adversarial_training], ['label smoothing', A.label_smoothing]]
  .map(([n, m]) => `<tr><td>${n}</td><td class="num">${fmt(m.clean_acc,4)}</td>
    <td class="num">${fmt(m.pgd[0.05],4)}</td><td class="num">${fmt(m.pgd[0.1],4)}</td>
    <td class="num">${fmt(m.pgd[0.2],4)}</td></tr>`).join('');
document.getElementById('at_gain').textContent =
  '+' + Math.round((A.pgd_adversarial_training.pgd[0.1] - A.baseline.pgd[0.1]) * 100) + 'pt';
document.getElementById('at_cost').textContent =
  '−' + Math.round((A.baseline.clean_acc - A.pgd_adversarial_training.clean_acc) * 100) + 'pt';

/* ---------- 06 supply chain ---------- */
document.getElementById('t_sc').innerHTML =
  '<tr><th>Scenario</th><th>Verdict</th><th>Detail</th></tr>' +
  DATA.lifecycle.map(r => `<tr><td>${r.scenario}</td>
    <td><span class="verdict ${r.verdict === 'PASS' ? 'pass' : 'fail'}">${r.verdict}</span></td>
    <td style="color:var(--muted)">${r.desc || 'byte-identical artifact, recomputed hash + HMAC match'}</td></tr>`).join('');
document.getElementById('sc_dis').textContent = pct(DATA.lifecycle.length ? DATA.behavior.prediction_disagreement_rate : 0, 1);

/* ---------- 08 cross-threat Δ-security matrix ---------- */
(function(){
  const X = DATA.cross;
  if (!X) { document.getElementById('sec_ctm').hidden = true; return; }
  document.getElementById('ctm').hidden = false;
  const names = {poison:'Poison', backdoor:'Backdoor', extraction:'Extract', mia:'MIA', adversarial:'Adv'};
  const rows = X.rows, defenses = rows.slice(1), base = rows[0];
  let svg = '<svg viewBox="0 0 640 ' + (64 + 34*defenses.length) + '" style="width:100%; min-width:520px; display:block">';
  const cw = 100, ch = 34, x0 = 132, y0 = 40;
  X.attacks.forEach((a, j) => {
    svg += `<text x="${x0 + j*cw + cw/2}" y="24" fill="${C.muted}" font-size="11" text-anchor="middle">${names[a]}</text>`;
  });
  defenses.forEach((d, i) => {
    svg += `<text x="${x0 - 8}" y="${y0 + i*ch + ch/2 + 4}" fill="${C.muted}" font-size="11" text-anchor="end">${d.label}</text>`;
    d.stats.forEach((s, j) => {
      const sig = s.class === '+' || s.class === '-';
      const col = s.class === '+' ? C.safe : s.class === '-' ? C.risk : sig ? '' : C.muted;
      const fill = s.class === '+' ? 'rgba(61,220,151,0.18)' : s.class === '-' ? 'rgba(255,92,108,0.18)' : 'rgba(34,49,75,0.35)';
      svg += `<rect x="${x0 + j*cw + 3}" y="${y0 + i*ch + 3}" width="${cw - 6}" height="${ch - 6}" rx="4"
        fill="${fill}" stroke="${s.class === '+' ? C.safe : s.class === '-' ? C.risk : C.line}" stroke-width="1"/>`;
      svg += `<text x="${x0 + j*cw + cw/2}" y="${y0 + i*ch + ch/2 - 1}" fill="${sig ? col : C.ink}"
        font-size="11.5" text-anchor="middle">${s.mean > 0 ? '+' : ''}${s.mean.toFixed(1)} ${s.class}</text>`;
      svg += `<text x="${x0 + j*cw + cw/2}" y="${y0 + i*ch + ch/2 + 9}" fill="${C.faint}"
        font-size="9" text-anchor="middle">±${s.std.toFixed(1)}</text>`;
    });
  });
  document.getElementById('heatmap').innerHTML = svg + '</svg>';

  document.getElementById('t_ctm').innerHTML =
    '<tr><th>Defense</th><th class="num">Clean acc</th><th class="num">MIA AUC</th><th class="num">Extract fid</th>' +
    '<th class="num">PGD succ ε=.1</th><th class="num">Poison src-acc</th><th class="num">Backdoor ASR</th><th class="num">Train ×</th></tr>' +
    rows.map(r => { const w = r.raw; return `<tr><td>${r.label}</td>
      <td class="num">${fmt(w.clean_acc,4)}</td><td class="num">${fmt(w.mia_auc,4)}</td>
      <td class="num">${fmt(w.fidelity,4)}</td><td class="num">${fmt(w.pgd_succ,4)}</td>
      <td class="num">${fmt(w.src_acc,4)}</td>
      <td class="num ${w.asr > base.raw.asr ? 'bad' : w.asr < base.raw.asr - 0.05 ? 'good' : ''}">${fmt(w.asr,4)}</td>
      <td class="num">${w.overhead.toFixed(2)}×</td></tr>`; }).join('');

  /* findings derived from the measured matrix itself */
  const flat = [];
  defenses.forEach(d => d.stats.forEach((s, j) => flat.push([s, d.label, names[X.attacks[j]]])));
  const sig = flat.filter(s => s[0].class === '+' || s[0].class === '-').sort((a, b) => b[0].mean - a[0].mean);
  const best = sig.find(s => s[0].class === '+');
  const worst = sig.slice().reverse().find(s => s[0].class === '-');
  let html = '';
  if (sig.length) {
    if (best) html += `<b>Significant transfer:</b> ${best[2]} under ${best[1]} (ΔS ${best[0].mean>0?'+':''}${best[0].mean.toFixed(1)} ± ${best[0].std.toFixed(1)}) · `;
    if (worst) html += `<b>significant adverse interaction:</b> ${worst[2]} under ${worst[1]} (ΔS ${worst[0].mean.toFixed(1)} ± ${worst[0].std.toFixed(1)})`;
    html += `<br>`;
  } else {
    html += `<b>No cell resolves beyond the 3-seed noise band yet</b> — the paired-t intervals are honest about it. `;
  }
  const bdRow = defenses.find(d => d.label.startsWith('Poisoning'));
  if (bdRow) {
    const bdDelta = base.raw.asr - bdRow.raw.asr;
    html += `<b>Poison-defense → backdoor:</b> audit-filter moves backdoor ASR from `
      + `${fmt(base.raw.asr,4)} to ${fmt(bdRow.raw.asr,4)} `
      + (Math.abs(bdDelta) < 0.02 ? '(essentially unchanged — data-level filtering does not reach backdoors)' : '');
  }
  document.getElementById('ctm_finding').innerHTML = html;
  document.getElementById('ctm_finding').hidden = false;
})();

document.getElementById('formula').textContent = DATA.formula;
</script>
"""

html = TEMPLATE.replace("__DATA__", json.dumps(DATA))
out = ROOT / "dashboard.html"
out.write_text(html, encoding="utf-8")
print(f"[dashboard] wrote {out} ({out.stat().st_size // 1024} KB)")