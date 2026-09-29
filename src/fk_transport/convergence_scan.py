from __future__ import annotations
import copy, csv, json, math
from pathlib import Path
from .config import canonical_config, validate_config
from .sweep import build_tasks

def _rows(path):
    with Path(path).open(newline="",encoding="utf-8") as handle: return list(csv.DictReader(handle))

def prepare_convergence_configs(base_config, combined_summary, boundary_differences,
                                output_prefix, maximum_points=36, neighbors=3,
                                temperatures=(0.005,0.01,0.03), threshold=1e-2):
    differences=[r for r in _rows(boundary_differences)
                 if any(math.isclose(float(r["temperature"]),t,abs_tol=1e-12) for t in temperatures)
                 and math.isclose(float(r["relative_threshold"]),threshold,abs_tol=1e-15)]
    if not differences: raise ValueError("no boundary differences match requested temperatures and threshold")
    if "midpoint_disorder_over_W" not in differences[0]:
        raise ValueError("rerun analyze-boundaries first; differences lack contour midpoints")
    bandwidth=2*float(base_config["model"]["half_bandwidth"])
    summary=_rows(combined_summary)
    available={}
    for row in summary:
        key=round(float(row["interaction"])/bandwidth,10)
        available.setdefault(key,set()).add(round(float(row["disorder_full_width"])/bandwidth,10))
    ordered=sorted(differences,key=lambda r:abs(float(r["delta_disorder_over_W"])))
    candidates=[]
    for low,high in zip(ordered, reversed(ordered)):
        candidates.extend((high,low))
    selected=[]; seen=set()
    for candidate in candidates:
        u=round(float(candidate["interaction_over_W"]),10)
        if u not in available: continue
        target=float(candidate["midpoint_disorder_over_W"])
        disorders=sorted(available[u],key=lambda d:abs(d-target))[:neighbors]
        for disorder in disorders:
            point=(u,disorder)
            if point not in seen:
                seen.add(point); selected.append(point)
                if len(selected)>=maximum_points: break
        if len(selected)>=maximum_points: break
    if not selected: raise ValueError("no existing grid points could be selected around boundaries")
    settings=((5e-4,20001,"eta_5e-4"),(2.5e-4,40001,"eta_2p5e-4"),(1.25e-4,80001,"eta_1p25e-4"))
    prefix=Path(output_prefix); prefix.parent.mkdir(parents=True,exist_ok=True); reports=[]
    for eta,n_omega,label in settings:
        cfg=copy.deepcopy(canonical_config(base_config)); cfg["numerics"]["broadening"]=eta; cfg["grid"]["n_omega"]=n_omega
        cfg["sweep"]["parameter_points"]=[{"interaction":u*bandwidth,"disorder_full_width":d*bandwidth} for u,d in selected]
        cfg["output"]["directory"]=f"results/stage3_convergence/{label}"
        cfg["convergence_scan"]={"source_summary":str(combined_summary),"source_boundaries":str(boundary_differences),"relative_threshold":threshold,"selection_temperatures":list(temperatures),"eta":eta,"n_omega":n_omega}
        validate_config(cfg); path=prefix.with_name(prefix.name+f"_{label}.json"); path.write_text(json.dumps(cfg,indent=2)+"\n",encoding="utf-8")
        reports.append({"config":str(path),"eta":eta,"n_omega":n_omega,"parameter_points":len(selected),"spectral_tasks":len(build_tasks(cfg))})
    return {"selected_parameter_points":len(selected),"configs":reports}
