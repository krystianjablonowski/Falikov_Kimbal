from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .boundary_analysis import _match_intersections

def _read(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))

def _write(path,rows,fields):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return path

def _ratios(rows,temperature,filling):
    chosen=[r for r in rows if np.isclose(float(r["temperature"]),temperature) and np.isclose(float(r["target_filling"]),filling)]
    lookup={(r["branch"],round(float(r["interaction"]),12),round(float(r["disorder_full_width"]),12)):r for r in chosen}
    result=[]
    for u,d in sorted({(k[1],k[2]) for k in lookup}):
        a,t=lookup.get(("arith",u,d)),lookup.get(("typ",u,d))
        if not a or not t:continue
        item={"interaction":u,"disorder":d}
        for field in ("sigma","kappa_e"):
            av,tv=float(a[field]),float(t[field])
            if av<=0 or tv<=0:break
            item[field]=tv/av
        else:result.append(item)
    return result

def _one_dimensional_crossings(points,field,threshold):
    found=[]
    for left,right in zip(points,points[1:]):
        x0,x1,y0,y1=left["disorder"],right["disorder"],left[field],right[field]
        if (y0-threshold)*(y1-threshold)>0 or y0==y1:continue
        q=(np.log10(threshold)-np.log10(y0))/(np.log10(y1)-np.log10(y0));found.append(x0+q*(x1-x0))
    return sorted(found)

def analyze_convergence(summaries,etas,output_directory,threshold=1e-2,maximum_separation=0.1):
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(summaries)!=len(etas) or len(summaries)<2:raise ValueError("provide matching lists of at least two summaries and etas")
    datasets=[_read(path) for path in summaries];out=Path(output_directory);out.mkdir(parents=True,exist_ok=True)
    crossings=[];statistics=[]
    temperatures=sorted({float(r["temperature"]) for rows in datasets for r in rows});fillings=sorted({float(r["target_filling"]) for rows in datasets for r in rows})
    for eta,rows in zip(etas,datasets):
      for filling in fillings:
       for temperature in temperatures:
        relative=_ratios(rows,temperature,filling);deltas=[];rejected=ambiguous=0
        for u in sorted({p["interaction"] for p in relative}):
            points=sorted((p for p in relative if np.isclose(p["interaction"],u)),key=lambda p:p["disorder"])
            sigma=_one_dimensional_crossings(points,"sigma",threshold);kappa=_one_dimensional_crossings(points,"kappa_e",threshold)
            pairs,rej,amb=_match_intersections(sigma,kappa,maximum_separation,0.025);rejected+=rej;ambiguous+=amb
            for branch,(ds,dk) in enumerate(pairs):
                delta=dk-ds;deltas.append(delta);crossings.append({"eta":eta,"temperature":temperature,"target_filling":filling,"interaction":u,"branch_index":branch,"sigma_crossing":ds,"kappa_crossing":dk,"delta":delta})
        absolute=np.abs(deltas)
        statistics.append({"eta":eta,"temperature":temperature,"target_filling":filling,"threshold":threshold,"accepted_pairs":len(deltas),"rejected":rejected,"ambiguous":ambiguous,"median_abs_delta":float(np.median(absolute)) if len(absolute) else np.nan,"p90_abs_delta":float(np.percentile(absolute,90)) if len(absolute) else np.nan})
    cross_path=_write(out/"eta_convergence_crossings.csv",crossings,["eta","temperature","target_filling","interaction","branch_index","sigma_crossing","kappa_crossing","delta"])
    stat_path=_write(out/"eta_convergence_summary.csv",statistics,["eta","temperature","target_filling","threshold","accepted_pairs","rejected","ambiguous","median_abs_delta","p90_abs_delta"])
    outputs=[cross_path,stat_path]
    for filling in fillings:
      for temperature in temperatures:
        group=sorted((r for r in statistics if np.isclose(r["temperature"],temperature) and np.isclose(r["target_filling"],filling)),key=lambda r:r["eta"])
        fig,axis=plt.subplots(figsize=(4.8,3.5));x=np.array([r["eta"] for r in group])
        axis.plot(x,[r["median_abs_delta"] for r in group],"o-",label="median")
        axis.plot(x,[r["p90_abs_delta"] for r in group],"s--",label="90th percentile")
        axis.set_xscale("log");axis.invert_xaxis();axis.set_xlabel(r"broadening $\eta/W$");axis.set_ylabel(r"$|\Delta_c^\kappa-\Delta_c^\sigma|/W$");axis.set_title(rf"$n_c={filling:g}$, $T/W={temperature:g}$");axis.legend(frameon=False);fig.tight_layout()
        tag=f"n_{filling:.6g}_T_{temperature:.6g}".replace(".","p");path=out/f"eta_convergence_{tag}.pdf";fig.savefig(path);plt.close(fig);outputs.append(path)
    return outputs
