from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

KEYS = ("branch", "interaction", "disorder_full_width", "target_filling", "temperature")
FIELDS = ("sigma", "kappa_e")

def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def _write(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return path

def combine_summaries(coarse, refined, output):
    unique = {}
    for row in _read(coarse) + _read(refined):
        key = tuple(row[k] if k == "branch" else round(float(row[k]), 12) for k in KEYS)
        unique[key] = row
    rows = list(unique.values())
    return rows, _write(output, rows, sorted({k for row in rows for k in row}))

def _relative_rows(rows, temperature, filling, bandwidth):
    selected = [r for r in rows if np.isclose(float(r["temperature"]), temperature)
                and np.isclose(float(r["target_filling"]), filling)]
    lookup = {(r["branch"], round(float(r["interaction"]), 12),
               round(float(r["disorder_full_width"]), 12)): r for r in selected}
    result = []
    for u, disorder in sorted({(key[1], key[2]) for key in lookup}):
        arith, typ = lookup.get(("arith", u, disorder)), lookup.get(("typ", u, disorder))
        if arith is None or typ is None: continue
        item = {"interaction_over_W": u/bandwidth, "disorder_over_W": disorder/bandwidth}
        for field in FIELDS:
            a, t = float(arith[field]), float(typ[field])
            if a <= 0 or t <= 0 or not np.isfinite(a+t): break
            item[f"R_{field}"] = t/a
        else: result.append(item)
    return result

def _crossings(relative, ratio, threshold):
    result = []
    for u in sorted({r["interaction_over_W"] for r in relative}):
        group = sorted((r for r in relative if np.isclose(r["interaction_over_W"], u)),
                       key=lambda r: r["disorder_over_W"])
        found = []
        for left, right in zip(group, group[1:]):
            x0, x1, y0, y1 = left["disorder_over_W"], right["disorder_over_W"], left[ratio], right[ratio]
            if (y0-threshold)*(y1-threshold) > 0 or y0 == y1: continue
            q = (np.log10(threshold)-np.log10(y0))/(np.log10(y1)-np.log10(y0))
            found.append(x0+q*(x1-x0))
        result.extend((u, d, branch) for branch, d in enumerate(sorted(found)))
    return result

def _gradients(relative, ratio):
    result = []
    for u in sorted({r["interaction_over_W"] for r in relative}):
        group = sorted((r for r in relative if np.isclose(r["interaction_over_W"], u)), key=lambda r:r["disorder_over_W"])
        if len(group) < 3: continue
        x = np.array([r["disorder_over_W"] for r in group]); y = np.log10(np.maximum([r[ratio] for r in group], 1e-300))
        g = np.abs(np.gradient(y, x)); peaks = [i for i in range(1,len(x)-1) if g[i]>=g[i-1] and g[i]>=g[i+1]]
        result.extend((u, float(x[i]), branch, float(g[i])) for branch, i in enumerate(peaks))
    return result

def analyze_boundaries(coarse, refined, output_directory, thresholds=(1e-2,1e-4,1e-6), bandwidth=1.0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if bandwidth <= 0 or any(t <= 0 or t >= 1 for t in thresholds):
        raise ValueError("bandwidth must be positive and relative thresholds must lie in (0,1)")
    out=Path(output_directory); out.mkdir(parents=True, exist_ok=True)
    # Remove obsolete absolute-threshold products, which are not comparable
    # between electrical and thermal conductivity.
    for obsolete in [out/"boundary_crossings.csv", *out.glob("boundary_crossings_*.png"), *out.glob("boundary_crossings_*.pdf")]:
        if obsolete.is_file(): obsolete.unlink()
    rows,combined=combine_summaries(coarse,refined,out/"combined_summary.csv")
    outputs=[combined]; crossings=[]; differences=[]; gradients=[]
    colors=plt.get_cmap("viridis")(np.linspace(.12,.9,len(thresholds)))
    for filling in sorted({float(r["target_filling"]) for r in rows}):
      for temperature in sorted({float(r["temperature"]) for r in rows}):
        rel=_relative_rows(rows,temperature,filling,bandwidth); found={}
        for field in FIELDS:
          for threshold in thresholds:
            pts=_crossings(rel,"R_"+field,threshold); found[field,threshold]=pts
            crossings += [{"temperature":temperature,"target_filling":filling,"field":field,"relative_threshold":threshold,"branch_index":b,"interaction_over_W":u,"disorder_over_W":d} for u,d,b in pts]
          gradients += [{"temperature":temperature,"target_filling":filling,"field":field,"branch_index":b,"interaction_over_W":u,"disorder_over_W":d,"abs_dlog10R_dDelta":score} for u,d,b,score in _gradients(rel,"R_"+field)]
        fig,axes=plt.subplots(2,2,figsize=(8.4,6.4)); names={"sigma":r"$R_\sigma$","kappa_e":r"$R_\kappa$"}
        for column,field in enumerate(FIELDS):
          for color,threshold in zip(colors,thresholds):
            pts=found[field,threshold]
            for b in sorted({p[2] for p in pts}):
              part=sorted(p for p in pts if p[2]==b); axes[0,column].plot([p[1] for p in part],[p[0] for p in part],".-",color=color,ms=3,lw=.7,label=rf"$r={threshold:.0e}$" if b==0 else None)
          axes[0,column].set_title(names[field]); axes[0,column].legend(frameon=False)
        for color,threshold in zip(colors,thresholds):
          for field,style in (("sigma","-"),("kappa_e","--")):
            pts=found[field,threshold]
            for b in sorted({p[2] for p in pts}):
              part=sorted(p for p in pts if p[2]==b); axes[1,0].plot([p[1] for p in part],[p[0] for p in part],style,color=color,lw=.9)
          sig={(u,b):d for u,d,b in found["sigma",threshold]}; kap={(u,b):d for u,d,b in found["kappa_e",threshold]}
          delta=[(u,kap[u,b]-sig[u,b],b) for u,b in sorted(set(sig)&set(kap))]
          differences += [{"temperature":temperature,"target_filling":filling,"relative_threshold":threshold,"branch_index":b,"interaction_over_W":u,"delta_disorder_over_W":v} for u,v,b in delta]
          for b in sorted({p[2] for p in delta}):
            part=[p for p in delta if p[2]==b]; axes[1,1].plot([p[0] for p in part],[p[1] for p in part],".-",color=color,ms=3,lw=.7,label=rf"$r={threshold:.0e}$" if b==0 else None)
        axes[1,0].set_title(r"solid $R_\sigma$, dashed $R_\kappa$"); axes[1,1].axhline(0,color=".4",lw=.7); axes[1,1].set_title(r"$\Delta_c^\kappa-\Delta_c^\sigma$"); axes[1,1].legend(frameon=False)
        for i,axis in enumerate(axes.flat): axis.set_xlabel(r"$U/W$" if i==3 else r"$\Delta/W$"); axis.set_ylabel(r"$\delta\Delta_c/W$" if i==3 else r"$U/W$")
        fig.suptitle(rf"$n_c={filling:g}$, $T/W={temperature/bandwidth:g}$"); fig.tight_layout()
        tag=f"n_{filling:.6g}_T_{temperature/bandwidth:.6g}".replace(".","p"); path=out/f"relative_boundary_crossings_{tag}.png"
        fig.savefig(path,dpi=300); fig.savefig(path.with_suffix(".pdf")); plt.close(fig); outputs.append(path)
    schemas={"relative_boundary_crossings.csv":(crossings,["temperature","target_filling","field","relative_threshold","branch_index","interaction_over_W","disorder_over_W"]),"relative_boundary_differences.csv":(differences,["temperature","target_filling","relative_threshold","branch_index","interaction_over_W","delta_disorder_over_W"]),"gradient_boundaries.csv":(gradients,["temperature","target_filling","field","branch_index","interaction_over_W","disorder_over_W","abs_dlog10R_dDelta"])}
    for name,(data,fields) in schemas.items(): outputs.append(_write(out/name,data,fields))
    return outputs
