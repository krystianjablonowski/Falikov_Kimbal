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

def _contour_paths(axis, relative, ratio, thresholds, colors):
    import matplotlib.tri as mtri
    x=np.array([r["disorder_over_W"] for r in relative]); y=np.array([r["interaction_over_W"] for r in relative])
    z=np.log10(np.maximum([r[ratio] for r in relative],1e-300)); tri=mtri.Triangulation(x,y)
    ordered=sorted(enumerate(thresholds),key=lambda item:np.log10(item[1]))
    levels=[np.log10(item[1]) for item in ordered]; contour=axis.tricontour(tri,z,levels=levels,colors=[colors[i] for i,_ in ordered],linewidths=.9)
    paths={threshold:[] for threshold in thresholds}; status={}
    for level_index,(_,threshold) in enumerate(ordered):
        paths[threshold]=[np.asarray(segment) for segment in contour.allsegs[level_index] if len(segment)>=2]
        status[threshold]={"minimum":float(10**np.nanmin(z)),"maximum":float(10**np.nanmax(z)),"found":bool(paths[threshold])}
    return paths,status

def _horizontal_intersections(paths, u):
    values=[]
    for path in paths:
        for first,second in zip(path,path[1:]):
            y0,y1=first[1],second[1]
            if y0==y1 or not (min(y0,y1)<=u<max(y0,y1)): continue
            values.append(float(first[0]+(u-y0)*(second[0]-first[0])/(y1-y0)))
    return sorted(values)

def analyze_boundaries(coarse, refined, output_directory, thresholds=(1e-2,1e-4,1e-6), bandwidth=1.0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if bandwidth <= 0 or any(t <= 0 or t >= 1 for t in thresholds):
        raise ValueError("bandwidth must be positive and relative thresholds must lie in (0,1)")
    out=Path(output_directory); out.mkdir(parents=True, exist_ok=True)
    # Remove obsolete absolute-threshold products, which are not comparable
    # between electrical and thermal conductivity.
    for obsolete in [out/"boundary_crossings.csv", *out.glob("boundary_crossings_*.png"), *out.glob("boundary_crossings_*.pdf")]:
        if obsolete.is_file(): obsolete.unlink()
    rows,combined=combine_summaries(coarse,refined,out/"combined_summary.csv")
    outputs=[combined]; crossings=[]; differences=[]; gradients=[]; level_status=[]
    colors=plt.get_cmap("viridis")(np.linspace(.12,.9,len(thresholds)))
    for filling in sorted({float(r["target_filling"]) for r in rows}):
      for temperature in sorted({float(r["temperature"]) for r in rows}):
        rel=_relative_rows(rows,temperature,filling,bandwidth); found={}; statuses={}
        fig,axes=plt.subplots(2,2,figsize=(8.4,6.4)); names={"sigma":r"$R_\sigma$","kappa_e":r"$R_\kappa$"}
        for field in FIELDS:
          column=FIELDS.index(field); paths,status=_contour_paths(axes[0,column],rel,"R_"+field,thresholds,colors)
          for threshold in thresholds:
            found[field,threshold]=paths[threshold]; statuses[field,threshold]=status[threshold]
            level_status.append({"temperature":temperature,"target_filling":filling,"field":field,"relative_threshold":threshold,"minimum_ratio":status[threshold]["minimum"],"maximum_ratio":status[threshold]["maximum"],"contour_found":status[threshold]["found"],"connected_components":len(paths[threshold])})
            for branch,path_vertices in enumerate(paths[threshold]):
              crossings += [{"temperature":temperature,"target_filling":filling,"field":field,"relative_threshold":threshold,"branch_index":branch,"interaction_over_W":float(vertex[1]),"disorder_over_W":float(vertex[0])} for vertex in path_vertices]
          gradients += [{"temperature":temperature,"target_filling":filling,"field":field,"branch_index":b,"interaction_over_W":u,"disorder_over_W":d,"abs_dlog10R_dDelta":score} for u,d,b,score in _gradients(rel,"R_"+field)]
        for column,field in enumerate(FIELDS):
          handles=[Line2D([0],[0],color=color,lw=1,label=rf"$r={threshold:.0e}$") for color,threshold in zip(colors,thresholds)]
          axes[0,column].set_title(names[field]); axes[0,column].legend(handles=handles,frameon=False)
        for color,threshold in zip(colors,thresholds):
          for field,style in (("sigma","-"),("kappa_e","--")):
            for path_vertices in found[field,threshold]: axes[1,0].plot(path_vertices[:,0],path_vertices[:,1],style,color=color,lw=.9)
          delta=[]
          for u in sorted({r["interaction_over_W"] for r in rel}):
            sig=_horizontal_intersections(found["sigma",threshold],u); kap=_horizontal_intersections(found["kappa_e",threshold],u)
            for branch,(ds,dk) in enumerate(zip(sig,kap)): delta.append((u,dk-ds,branch))
          differences += [{"temperature":temperature,"target_filling":filling,"relative_threshold":threshold,"branch_index":b,"interaction_over_W":u,"delta_disorder_over_W":v} for u,v,b in delta]
          axes[1,1].scatter([p[0] for p in delta],[p[1] for p in delta],s=7,color=color,label=rf"$r={threshold:.0e}$")
        axes[1,0].set_title(r"solid $R_\sigma$, dashed $R_\kappa$"); axes[1,1].axhline(0,color=".4",lw=.7); axes[1,1].set_title(r"$\Delta_c^\kappa-\Delta_c^\sigma$"); axes[1,1].legend(frameon=False)
        for i,axis in enumerate(axes.flat): axis.set_xlabel(r"$U/W$" if i==3 else r"$\Delta/W$"); axis.set_ylabel(r"$\delta\Delta_c/W$" if i==3 else r"$U/W$")
        fig.suptitle(rf"$n_c={filling:g}$, $T/W={temperature/bandwidth:g}$"); fig.tight_layout()
        tag=f"n_{filling:.6g}_T_{temperature/bandwidth:.6g}".replace(".","p"); path=out/f"relative_boundary_crossings_{tag}.png"
        fig.savefig(path,dpi=300); fig.savefig(path.with_suffix(".pdf")); plt.close(fig); outputs.append(path)
    schemas={"relative_boundary_crossings.csv":(crossings,["temperature","target_filling","field","relative_threshold","branch_index","interaction_over_W","disorder_over_W"]),"relative_boundary_differences.csv":(differences,["temperature","target_filling","relative_threshold","branch_index","interaction_over_W","delta_disorder_over_W"]),"gradient_boundaries.csv":(gradients,["temperature","target_filling","field","branch_index","interaction_over_W","disorder_over_W","abs_dlog10R_dDelta"]),"relative_boundary_level_status.csv":(level_status,["temperature","target_filling","field","relative_threshold","minimum_ratio","maximum_ratio","contour_found","connected_components"])}
    for name,(data,fields) in schemas.items(): outputs.append(_write(out/name,data,fields))
    return outputs
