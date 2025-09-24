from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re
import itertools
import yaml
from types import SimpleNamespace
import numpy as np
import pandas as pd
import xarray as xr

from imagematerials import iamc_config as CFG

# ---------------------------------------------------------------------
# Helpers for time, labels, units
# ---------------------------------------------------------------------

# find time coordinate/dimension key
def _find_time_key(da: xr.DataArray) -> str:
    for k in ("time", "Time"):
        if k in da.coords or k in da.dims:
            return k
    raise KeyError("No time coordinate/dimension found (time/Time/year/Year).")

# define years to report (2005 to last in 5-year steps)
def _years(da: xr.DataArray) -> List[int]:
    """Pick 2005..last in 5-year steps, always include last."""
    tk = _find_time_key(da)
    t = da.coords[tk] if tk in da.coords else da[tk]
    try:
        ys = t.dt.year.values.astype(int)
    except Exception:
        ys = np.asarray(getattr(t, "values", t)).astype(int)

    ys = np.unique(ys)
    if ys.size == 0:
        return []
    last = int(ys[-1])
    if last <= 2005:
        return [last]

    wanted = list(range(2005, last + 1, 5))
    if last not in wanted:
        wanted.append(last)
    avail = set(map(int, ys.tolist()))
    return [y for y in wanted if y in avail]

# list of labels for a given dimension to iterate over (e.g. Regions, Types)
def _labels(da: xr.DataArray, dim: str) -> List[str]:
    if dim not in da.coords:
        raise KeyError(f"Coordinate {dim!r} not found.")
    vals = da.coords[dim].values
    return [str(v) for v in np.asarray(vals).tolist()]

# unit for a given family/template (preferably from variable YAML))
def _unit(da: xr.DataArray, family: str, template: str) -> str:
    yaml_unit = CFG.IAMC_VAR_SPECS.get(family, {}).get("units_per_template", {}).get(template, "")
    if yaml_unit:
        return yaml_unit
    for k in ("unit", "units", "Units"):
        if k in da.attrs and da.attrs[k]:
            return str(da.attrs[k])
    return ""

# load unit conversion factors from YAML
def _load_unit_conv(path: Path | None = None) -> Dict:
    if path is None:
        path = CFG.YAML_DIR.parent / "unit_conv.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("UNIT_CONVERSIONS", data)

UNIT_CONV = _load_unit_conv()

# conversion factor for a given family/template
def _conv_for(family: str, template: str) -> float:
    fam = UNIT_CONV[family]
    rec = fam[template]
    return float(rec.get("factor", 1.0))

# ---------------------------------------------------------------------
# Knowledge graph helpers
# ---------------------------------------------------------------------

# find family of variables from template (e.g., "Final Material Demand|Transportation|{Vehicles}" -> "final_material_demand")
def _family_from_template(template: str) -> str:
    for fam, spec in CFG.IAMC_VAR_SPECS.items():
        if template in (spec.get("templates") or []):
            return fam
    raise ValueError(f"Template not found in IAMC_VAR_SPECS: {template!r}")

# assign model variable name for a given family (e.g., "inflow_materials" -> "final_material_demand")
def _model_var_for_family(family: str) -> str:
    mv = CFG.IAMC_VAR_SPECS.get(family, {}).get("model_var")
    if not mv:
        raise KeyError(f"Missing model_var for family {family!r}")
    return mv

# get canonical path for a label from knowledge graphs
def _kg_path(kg, label: str, prefer_depth: int | None = None) -> str:
    """
    Resolve a canonical path-like string from the KG.
    If prefer_depth is given (e.g., 2), truncate deeper paths to that depth:
        Road|Cars|BEV  --(prefer_depth=2)-->  Road|Cars
    """
    s = str(label).strip()
    if "|" in s:
        path = s
    else:
        try:
            node = kg[s]
        except Exception:
            raise KeyError(f"[KG] Unknown label {s!r}")

        syns = getattr(node, "synonyms", None) or []
        for syn in syns:
            if isinstance(syn, str) and "|" in syn:
                path = syn
                break
        else:
            def _coerce_parent_to_path(p):
                if isinstance(p, str) and "|" in p:
                    return p
                name = getattr(p, "name", None) or getattr(p, "label", None)
                if isinstance(name, str) and "|" in name:
                    return name
                ip = getattr(p, "inherits_from", None)
                if isinstance(ip, list):
                    for q in ip:
                        out = _coerce_parent_to_path(q)
                        if out:
                            return out
                return None

            parent = getattr(node, "inherits_from", None)
            path = None
            if parent is not None and not isinstance(parent, list):
                path = _coerce_parent_to_path(parent)
            if path is None and isinstance(parent, list):
                for p in parent:
                    path = _coerce_parent_to_path(p)
                    if path:
                        break

            if not path:
                name = getattr(node, "name", None) or getattr(node, "label", None)
                if isinstance(name, str) and "|" in name:
                    path = name

            if not path:
                raise KeyError(f"[KG] Node {s!r} lacks a canonical path.")

    if prefer_depth is not None and isinstance(prefer_depth, int) and prefer_depth > 0:
        parts = path.split("|")
        if len(parts) >= prefer_depth:
            return "|".join(parts[:prefer_depth])
    return path

# get region code (e.g., "Middle East" -> "ME")
def _region_code(label: str) -> str:
    region_label = str(label)
    try:
        node = CFG.kgraph_region[region_label]
    except Exception:
        try:
            node = CFG.kgraph_region[str(region_label)]
        except Exception:
            return region_label
    syns = getattr(node, "synonyms", None) or []
    codes = [x for x in syns if isinstance(x, str) and x.isupper() and 2 <= len(x) <= 5]
    if codes:
        codes.sort(key=len)
        return codes[0]
    parent = getattr(node, "inherits_from", None)
    if isinstance(parent, str) and parent.isupper():
        return parent
    return region_label

# deduplicate lists in a dict and sort them (used for rollups) - e.g., {"a": ["x", "y", "x"], "b": ["z"]} -> {"a": ["x", "y"], "b": ["z"]}
def _dedup_lists(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {k: sorted(set(v)) for k, v in d.items()}

# add rollup paths to a mapping (e.g., "Transportation|Road|Cars" -> "Transportation|Road", "Transportation")
def _with_rollups(mapping: Dict[str, List[str]], allowed_parents: set[str] | None = None) -> Dict[str, List[str]]:
    out = {k: list(v) for k, v in mapping.items()}
    for k, labels in mapping.items():
        parts = k.split("|")
        for i in range(1, len(parts)):
            parent = "|".join(parts[:i])
            if allowed_parents and parent not in allowed_parents:
                continue
            out.setdefault(parent, []).extend(labels)
    return _dedup_lists(out)

# ---------------------------------------------------------------------
# Placeholder mappers
# ---------------------------------------------------------------------
def _prefixes_match(tag: str, full: str) -> bool:
    """True if 'tag' is a component-wise prefix of 'full' (IAMC paths)."""
    t = tag.split("|")
    f = full.split("|")
    return len(t) <= len(f) and t == f[:len(t)]

# map a placeholder to a dimension and a mapping of IAMC labels to source model labels 
# (e.g., "Vehicles" -> ("Type", {"Transportation|Road|Cars": ["Cars - BEV", "'Cars - FCV', ...], ...}))"
def _map_placeholder(placeholder: str, da: xr.DataArray, sector_name: str | None = None) -> Tuple[str, Dict[str, List[str]]]:
    ph = placeholder.strip()

    # ----- Vehicles -----
    if ph == "Vehicles":
        dim = "Type"
        labels = _labels(da, dim)

        # Deep path for every model Type (from your KG)
        deep_paths = {t: _kg_path(CFG.kgraph_v, t) for t in labels}

        # Optional whitelist of allowed IAMC tags from YAML (exposed in CFG)
        allowed = set(getattr(CFG, "TAG_WHITELISTS", {}).get("Vehicles", []))

        mapping: Dict[str, List[str]] = {}
        if allowed:
            # Assign each Type to every allowed tag that's a prefix of its deep path
            for t, full in deep_paths.items():
                for tag in allowed:
                    if _prefixes_match(tag, full):
                        mapping.setdefault(tag, []).append(t)
        else:
            # Fallback: include leaf + parent@depth=2
            for t, full in deep_paths.items():
                mapping.setdefault(full, []).append(t)
                p2 = _kg_path(CFG.kgraph_v, t, prefer_depth=2)
                mapping.setdefault(p2, []).append(t)
                # If you also want depth=1 rollups (e.g., "Road"), uncomment:
                # p1 = _kg_path(CFG.kgraph_v, t, prefer_depth=1)
                # mapping.setdefault(p1, []).append(t)

        return dim, {k: sorted(set(v)) for k, v in mapping.items()}

    # ----- Buildings -----
    if ph == "Building Types":
        dim = "Type"
        labels = _labels(da, dim)

        deep_paths = {t: _kg_path(CFG.kgraph_b, t) for t in labels}
        allowed = set(getattr(CFG, "TAG_WHITELISTS", {}).get("Building Types", []))

        mapping: Dict[str, List[str]] = {}
        if allowed:
            for t, full in deep_paths.items():
                for tag in allowed:
                    if _prefixes_match(tag, full):
                        mapping.setdefault(tag, []).append(t)
        else:
            for t, full in deep_paths.items():
                mapping.setdefault(full, []).append(t)
                p2 = _kg_path(CFG.kgraph_b, t, prefer_depth=2)
                mapping.setdefault(p2, []).append(t)
                # optional depth=1 same as above

        return dim, {k: sorted(set(v)) for k, v in mapping.items()}

    # ----- Materials -----
    if ph == "Engineered Material":
        dim = "material"
        labels = _labels(da, dim)
        return dim, { CFG.MATERIAL_NAME_MAP.get(str(t), str(t)): [t] for t in labels }

    # ----- Demand Sector  -----
    if ph == "Demand Sector":
        dim = "Type"
        labels = _labels(da, dim)
        mapping: Dict[str, List[str]] = {}

        if sector_name == "vehicles":
            for t in labels:
                mapping.setdefault("Transportation|" + _kg_path(CFG.kgraph_v, t), []).append(t)
            mapping = _with_rollups(mapping, {"Transportation", "Transportation|Road", "Transportation|Rail"})
            return dim, mapping

        if sector_name == "buildings":
            for t in labels:
                path = _kg_path(CFG.kgraph_b, t)
                path = path.replace("Residential|Residential High-Rise", "Residential|Residential Towers")
                mapping.setdefault("Buildings|" + path, []).append(t)
            mapping = _with_rollups(mapping, {"Buildings", "Buildings|Residential", "Buildings|Commercial"})
            return dim, mapping

    raise KeyError(f"Unsupported placeholder {placeholder!r}")


# ---------------------------------------------------------------------
# End-of-life helpers
# ---------------------------------------------------------------------

def _eol_source_key_for_template(template: str) -> str:
    tl = template.lower()
    if "final material demand" in tl and "|reused" in tl:
        return "reusable_materials"
    raise KeyError(f"Don’t know which EoL key to use for template: {template!r}")

# ---------------------------------------------------------------------
# Core IAMC reporting
# ---------------------------------------------------------------------

# Create IAMC reporting DataFrame from all_output, templates, sector, and model name
def create_iamc_reporting(
    models: Dict[str, object],   # 
    templates: List[str],
    sector: str,                
    model_name: str,
    outfile: str | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    all_rows: List[Dict[str, object]] = []

    # Iterate over scenarios and templates to extract data
    for scen_label, src_model in models.items():
        # for each template, find family and model var
        for tpl in templates:
            try:
                family = _family_from_template(tpl)
            except ValueError:
                continue
            model_var = _model_var_for_family(family)

            # get data array for the sector and model_var
            sector_obj = getattr(src_model, sector, None)
            if sector_obj is None or model_var not in sector_obj:
                continue
            
            # extract data array 
            da_raw = sector_obj[model_var]
            da = da_raw if isinstance(da_raw, xr.DataArray) else da_raw.to_array()
            
            # find time key, region key, years, unit, conversion factor
            tkey = _find_time_key(da)
            rkey = "Region" if "Region" in da.dims or "Region" in da.coords else "region"
            years = _years(da)
            unit = _unit(da, family, tpl)
            try:
                factor = _conv_for(family, tpl)
            except Exception:
                factor = 1.0

            # get regions and their codes
            regions = _labels(da, rkey)
            region_codes = {r: _region_code(r) for r in regions}
            
            # find placeholders in template
            ph_list = re.findall(r"\{([^}]+)\}", tpl)
            if not ph_list:
                continue
            # build placeholder specs (list of dicts with ph, dim, map)
            ph_specs = [ {"ph": ph, "dim": _map_placeholder(ph, da, sector)[0], "map": _map_placeholder(ph, da, sector)[1]} for ph in ph_list ]
            combos = list(itertools.product(*[list(ps["map"].keys()) for ps in ph_specs]))

            # debugging output
            if debug:
                chk = float(np.asarray(da.sum().values))
                print(f"[dbg] scen={scen_label} sec={sector} fam={family} sum={chk:.3e}")

            # iterate over regions and placeholder combinations to build rows
            for reg in regions:
                for combo in combos:
                    # select and sum data array over placeholder dims
                    sel, sum_dims = {}, []
                    for ps, label in zip(ph_specs, combo):
                        sel[ps["dim"]] = ps["map"][label]
                        if ps["dim"] not in sum_dims:
                            sum_dims.append(ps["dim"])
                    # select, sum, and get values for years and region
                    part = da.sel(sel).sum(sum_dims, keep_attrs=True).sel({tkey: years})
                    vec = np.asarray(part.sel({rkey: reg}).values, dtype=float).reshape(-1) * factor

                    # build variable name by replacing placeholders in template
                    var = tpl
                    for ph, label in zip(ph_list, combo):
                        var = var.replace(f"{{{ph}}}", label)
                    # build row dict and append to all_rows list 
                    row = {
                        "model": model_name,
                        "scenario": scen_label,
                        "region": region_codes[reg],
                        "variable": var,
                        "unit": unit,
                    }
                    row.update({int(y): float(v) for y, v in zip(years, vec)})
                    all_rows.append(row)

    # check if any rows were produced
    if not all_rows:
        raise RuntimeError("No rows produced.")
    # create DataFrame and order columns
    df = pd.DataFrame(all_rows)
    fixed = ["model", "scenario", "region", "variable", "unit"]
    years_cols = sorted([c for c in df.columns if isinstance(c, int)])
    df = df[fixed + years_cols]

    # write to file if outfile is given
    if outfile:
        p = Path(outfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in (".xlsx", ".xls"):
            with pd.ExcelWriter(p, engine="xlsxwriter") as xlw:
                df.to_excel(xlw, sheet_name="data", index=False)
        else:
            df.to_csv(p, index=False)
    return df

# ---------------------------------------------------------------------
# End-of-life reporting
# ---------------------------------------------------------------------

def create_iamc_eol(
    model,
    templates: List[str],
    model_name: str,
    scenarios: List[str],
    outfile: str | None = None,
) -> pd.DataFrame:
    eol = getattr(model, "eol")
    all_rows: List[Dict[str, object]] = []

    for tpl in templates:
        try:
            family = _family_from_template(tpl)
        except ValueError:
            family = None

        src_key = _eol_source_key_for_template(tpl)
        if src_key not in eol:
            continue

        da_raw = eol[src_key]
        da = da_raw if isinstance(da_raw, xr.DataArray) else da_raw.to_array()
        tkey = _find_time_key(da)
        rkey = "Region" if "Region" in da.dims or "Region" in da.coords else "region"
        ysel = _years(da)
        regions = _labels(da, rkey)
        region_codes = {r: _region_code(r) for r in regions}

        unit = _unit(da, family or "", tpl) if family else (da.attrs.get("unit") or "")
        try:
            factor = _conv_for(family, tpl) if family else 1.0
        except Exception:
            factor = 1.0

        ph_list = re.findall(r"\{([^}]+)\}", tpl)
        ph_specs = []
        if "Engineered Material" in ph_list:
            dim = "material"
            mat_map = { CFG.MATERIAL_NAME_MAP.get(str(m), str(m)): [m] for m in _labels(da, dim) }
            ph_specs.append({"ph": "Engineered Material", "dim": dim, "map": mat_map})
        if "Demand Sector" in ph_list:
            dim = "Type"
            types_avail = set(_labels(da, dim))
            ds_map = {}
            for iamc_label, source_types in CFG.EOL_DEMAND_SECTOR_GROUPS.items():
                kept = [t for t in source_types if t in types_avail]
                if kept:
                    ds_map[iamc_label] = kept
            ph_specs.append({"ph": "Demand Sector", "dim": dim, "map": ds_map})

        combos = list(itertools.product(*[list(ps["map"].keys()) for ps in ph_specs])) or [()]

        for scen in scenarios:
            for reg in regions:
                for combo in combos:
                    sel, sum_dims = {}, []
                    for ps, label in zip(ph_specs, combo):
                        sel[ps["dim"]] = ps["map"][label]
                        if ps["dim"] not in sum_dims:
                            sum_dims.append(ps["dim"])
                    part = da.sel(sel).sum(sum_dims, keep_attrs=True).sel({tkey: ysel})
                    vec = np.asarray(part.sel({rkey: reg}).values, dtype=float).reshape(-1) * factor

                    var = tpl
                    for ps, label in zip(ph_specs, combo):
                        var = var.replace(f"{{{ps['ph']}}}", label)

                    row = {
                        "model": model_name,
                        "scenario": scen,
                        "region": region_codes[reg],
                        "variable": var,
                        "unit": unit,
                    }
                    row.update({int(y): float(v) for y, v in zip(ysel, vec)})
                    all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No EoL rows produced.")

    df = pd.DataFrame(all_rows)
    fixed = ["model", "scenario", "region", "variable", "unit"]
    year_cols = sorted([c for c in df.columns if isinstance(c, int)])
    df = df[fixed + year_cols]

    if outfile:
        p = Path(outfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        (df.to_excel(p, index=False) if p.suffix.lower() in (".xlsx", ".xls") else df.to_csv(p, index=False))
    return df
