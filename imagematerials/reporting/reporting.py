from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re
import itertools
import yaml
import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.reporting import iamc_config as CFG

# ---------------------------------------------------------------------
# Helpers for time, labels, units
# ---------------------------------------------------------------------

# find time coordinate/dimension key 
def _find_time_key(da: xr.DataArray) -> str:
    for k in ("time", "Time"):
        if k in da.coords or k in da.dims:
            return k
    raise KeyError("No time coordinate/dimension found (time/Time).")

# define years to report (2005 to last in 5-year steps)
def _years(da: xr.DataArray) -> List[int]:
    """
    Allow only:
      - 2005, 2010, ..., 2050 (5-year steps)
      - 2060, 2070, ..., 2100 (10-year steps)
    and return only those that are available in the DataArray.
    """
    tkey = _find_time_key(da)
    years = np.asarray(da.coords[tkey].values, dtype=int)
    years = np.unique(years)

    if years.size == 0:
        return []

    allowed = (
        list(range(2005, 2051, 5)) +
        list(range(2060, 2101, 10))
    )

    available = set(years.tolist())
    return [y for y in allowed if y in available]


# list of labels for a given dimension to iterate over (e.g. Regions, Types)
def _labels(da: xr.DataArray, dim: str) -> List[str]:
    # allow dimension without an explicit coordinate
    if (dim not in da.coords) and (dim not in da.dims):
        raise KeyError(f"Dimension/coordinate {dim!r} not found. dims={da.dims}, coords={list(da.coords)}")

    # If it's a coordinate, use it
    if dim in da.coords:
        vals = da.coords[dim].values
        return [str(v) for v in np.asarray(vals).tolist()]

    # Otherwise fall back to the index of the dimension
    idx = da.get_index(dim)
    return [str(v) for v in idx.tolist()]

# unit for a given family/template YAML file
def _unit(da: xr.DataArray, family: str, template: str) -> str:
    yaml_unit = CFG.IAMC_VAR_SPECS.get(family, {}).get("units_per_template", {}).get(template, "")
    if yaml_unit:
        return yaml_unit
    for k in ("unit", "units", "Units"):
        if k in da.attrs and da.attrs[k]:
            return str(da.attrs[k])
    return ""

def _restrict_to_whitelist(mapping: Dict[str, List[str]], whitelist: List[str]) -> Dict[str, List[str]]:
    """Return mapping containing ONLY whitelist keys (if present) and only non-empty lists."""
    out = {}
    for k in whitelist:
        vals = mapping.get(k, [])
        if vals:
            out[k] = sorted(set(vals))
    return out

# load unit conversion factors from YAML file
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
    """
    Return conversion factor for (family, template).

    Matching order:
      1) exact key match
      2) wildcard prefix match: keys ending with '*' match template.startswith(key_without_star)
      3) default 1.0
    """
    fam = UNIT_CONV.get(family, {})
    if not fam:
        return 1.0

    # 1) exact match
    rec = fam.get(template)
    if rec is not None:
        return float(rec.get("factor", 1.0))

    # 2) wildcard/prefix match (keys ending in '*')
    # Prefer the longest prefix (most specific)
    best_factor = None
    best_len = -1
    for key, rec in fam.items():
        if not isinstance(key, str) or not key.endswith("*"):
            continue
        prefix = key[:-1]
        if template.startswith(prefix) and len(prefix) > best_len:
            best_len = len(prefix)
            best_factor = float(rec.get("factor", 1.0))

    if best_factor is not None:
        return best_factor

    # 3) fallback
    return 1.0


# ---------------------------------------------------------------------
# Knowledge graph helpers
# ---------------------------------------------------------------------

# find family of variables from template (e.g., "Final Material Demand|Transportation|{Vehicles}" -> "final_material_demand.yaml")
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

# deduplicate lists in a dict and sort them (used for rollups) - e.g., {"a": ["x", "y", "x"], "b": ["z"]} -> {"a": ["x", "y"], "b": ["z"]}
def _dedup_lists(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {k: sorted(set(v)) for k, v in d.items()}

# add rollup paths to a mapping (e.g., "Transportation|Road|Cars" -> "Transportation|Road", "Transportation")
def _with_rollups(mapping: Dict[str, List[str]], allowed_parents: set[str] | None = None) -> Dict[str, List[str]]:
    out = {k: list(v) for k, v in mapping.items()}
    # for each key, split into parts and create parent keys
    for k, labels in mapping.items():
        parts = k.split("|")
        # iterate over parent levels
        for i in range(1, len(parts)):
            parent = "|".join(parts[:i])
            # skip if not in allowed parents
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
    if ph == "Vehicle Types":
        dim = "Type"
        labels = _labels(da, dim)

        # Deep path for every model Type (from your KG)
        deep_paths = {t: _kg_path(CFG.kgraph_v, t) for t in labels}

        # Optional whitelist of allowed IAMC tags from YAML (exposed in CFG)
        allowed = set(getattr(CFG, "TAG_WHITELISTS", {}).get("Vehicle Types", []))

        mapping: Dict[str, List[str]] = {}
        if allowed:
            # Assign each Type to every allowed tag that's a prefix of its deep path
            mapping = {tag: [] for tag in allowed}
            for t, full in deep_paths.items():
                for tag in allowed:
                    if _prefixes_match(tag, full):
                        mapping[tag].append(t)
            mapping = _restrict_to_whitelist(mapping, list(allowed))
            return dim, mapping
        
        else:
            # Fallback: include leaf + parent@depth=2
            for t, full in deep_paths.items():
                mapping.setdefault(full, []).append(t)
                p2 = _kg_path(CFG.kgraph_v, t, prefer_depth=2)
                mapping.setdefault(p2, []).append(t)

        return dim, {k: sorted(set(v)) for k, v in mapping.items()}

# ----- Buildings -----
    if ph == "Building Types":
        dim = "Type"
        labels = _labels(da, dim)

        deep_paths = {t: _kg_path(CFG.kgraph_b, t) for t in labels}
        allowed = list(getattr(CFG, "TAG_WHITELISTS", {}).get("Building Types", []))
        allowed_set = set(allowed)

        mapping: Dict[str, List[str]] = {}

        if allowed:
            # Only whitelist tags
            mapping = {tag: [] for tag in allowed}

            for t, full in deep_paths.items():
                # normalize: if someone accidentally returns Buildings|..., strip Buildings|

                if full == "Buildings":
                    full_norm = "Buildings"
                elif full.startswith("Buildings|"):
                    full_norm = full[len("Buildings|"):]
                else:
                    full_norm = full

                for tag in allowed:
                    if _prefixes_match(tag, full_norm):
                        mapping[tag].append(t)

            # keep only whitelist keys that have members
            mapping = {k: sorted(set(v)) for k, v in mapping.items() if v}

        else:
            # Fallback: include leaf + parent@depth=2, but never create Buildings|Buildings
            for t, full in deep_paths.items():
                mapping.setdefault(full, []).append(t)

                p2 = _kg_path(CFG.kgraph_b, t, prefer_depth=2)
                mapping.setdefault(p2, []).append(t)

            # Residential/Commercial rollups (based on deep path)
            res_types = [t for t, full in deep_paths.items() if full.startswith("Residential|")]
            com_types = [t for t, full in deep_paths.items() if full.startswith("Commercial|")]

            if res_types:
                mapping.setdefault("Residential", []).extend(res_types)
            if com_types:
                mapping.setdefault("Commercial", []).extend(com_types)

            # roll up to total Buildings (only from res+com members)
            if res_types or com_types:
                mapping.setdefault("Buildings", []).extend(res_types + com_types)

            mapping = {k: sorted(set(v)) for k, v in mapping.items() if v}

        return dim, mapping


    # ----- Electricity -----
    if ph == "Electricity Types":
        allowed_list = list(getattr(CFG, "TAG_WHITELISTS", {}).get("Electricity Types", []))
        allowed_set = set(allowed_list)

        # choose dim
        dim = "Type" if "Type" in da.dims else ("SuperType" if "SuperType" in da.dims else None)
        if dim is None:
            raise KeyError(f"Neither 'Type' nor 'SuperType' found for Electricity Types. dims={da.dims}")

        labels = _labels(da, dim)

        # ---- Case A: SuperType already aggregated (Generation / Transmission and Distribution / Storage) ----
        if dim == "SuperType":
            # only keep exact matches with whitelist
            mapping = {}
            for tag in allowed_list:
                kept = [t for t in labels if str(t) == tag]
                if kept:
                    mapping[tag] = kept  # selecting SuperType is selecting itself
            return dim, mapping

        # ---- Case B: Type is tech-level: use KG to map each tech to a whitelisted aggregate ----
        deep_paths = {t: _kg_path(CFG.kgraph_e, t) for t in labels}

        # Build ONLY whitelist keys, each collecting matching types
        mapping = {tag: [] for tag in allowed_list}

        for t, full in deep_paths.items():
            for tag in allowed_list:
                if _prefixes_match(tag, full):
                    mapping[tag].append(t)

        # ONLY whitelist keys survive
        mapping = _restrict_to_whitelist(mapping, allowed_list)
        return dim, mapping


    # ----- Demand Sector  -----
    if ph == "Demand Sector Disaggregated":
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

                # 🔒 guard: never create Buildings|Buildings
                if path == "Buildings":
                    mapping.setdefault("Buildings", []).append(t)
                else:
                    mapping.setdefault(f"Buildings|{path}", []).append(t)

            # rollups (only meaningful parents)
            mapping = _with_rollups(
                mapping,
                {
                    "Buildings",
                    "Buildings|Residential",
                    "Buildings|Commercial",
                },
            )

            # final cleanup
            mapping = {k: sorted(set(v)) for k, v in mapping.items()}
            return dim, mapping


        if sector_name == "electricity":
            for t in labels:
                path = _kg_path(CFG.kgraph_e, t)
                mapping.setdefault("Electricity|" + path, []).append(t)
            mapping = _with_rollups(mapping, {"Electricity", "Electricity|Generation", "Electricity|Transmission and Distribution", "Electricity|Storage"})
            return dim, mapping

    # ----- Materials -----
    
    if ph == "Engineered Material":
        dim = "material"
        labels = _labels(da, dim)
        labels = [m for m in labels if m in CFG.RAW_MATERIALS_KEEP]
        return dim, {CFG.MATERIAL_NAME_MAP.get(m, m): [m] for m in labels}

    raise KeyError(f"Unsupported placeholder {placeholder!r}")

# ---------------------------------------------------------------------
# Core IAMC reporting
# ---------------------------------------------------------------------
EXCLUDED_VARIABLES = {
    "Product Stock|Buildings",
}
# Create IAMC reporting DataFrame from all_output, templates, sector, and model name
def create_iamc_reporting(
    models: Dict[str, object],   # 
    templates: List[str],
    sector: str,                
    model_name: str,
    outdir: str | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    

    outdir = Path(outdir) if outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    dfs = []
    # Iterate over scenarios and templates to extract data
    for scen_label, src_model in models.items():
        all_rows: List[Dict[str, object]] = []
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
            if hasattr(da, "pint"):
                da = da.pint.dequantify()
            
            # find time key, region key, years, unit, conversion factor
            tkey = _find_time_key(da)
            rkey = "Region"
            years = _years(da)
            unit_tpl = _unit(da, family, tpl)
            try:
                factor = _conv_for(family, tpl)
            except Exception:
                factor = 1.0

            # get regions and their codes
            regions = _labels(da, rkey)

            # find placeholders in template 
            ph_list = re.findall(r"\{([^}]+)\}", tpl)   # regex pattern to capture what is inside the tags 
                                                        # (e.g., Vehicle Types in {Vehicle Types}) 
                                                        # this tells which dimensions to expand over (tags)
            # in case tags DO NOT EXIST:
            if not ph_list:
                raise ValueError(
                    f"Template has no placeholders, but this reporting pipeline assumes every template "
                    f"includes at least one placeholder so rollups/aggregation are applied.\n"
                    f"Template: {tpl!r}"
                )

            # in case tags DO EXIST:
            # build placeholder specs (list of dicts with ph, dim, map)
            ph_specs = []
            for ph in ph_list:
                dim, mp = _map_placeholder(ph, da, sector)
                ph_specs.append({"ph": ph, "dim": dim, "map": mp})
            # if any placeholder map is empty, skip this template for this DA
            if any(len(ps["map"]) == 0 for ps in ph_specs):
                if debug:
                    empty = [ps["ph"] for ps in ph_specs if len(ps["map"]) == 0]
                    print(f"[dbg] skipping tpl={tpl} because empty maps for {empty}")
                continue

            combos = list(itertools.product(*[list(ps["map"].keys()) for ps in ph_specs])) 

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
                    unit = unit_tpl
                    for ph, label in zip(ph_list, combo):
                        var = var.replace(f"{{{ph}}}", label)
                        if isinstance(unit, str) and "{" in unit:
                            unit = unit.replace(f"{{{ph}}}", label)
                    if var in EXCLUDED_VARIABLES:
                        continue
                    # build row dict and append to all_rows list 
                    row = {
                        "model": model_name,
                        "scenario": scen_label,
                        "region": reg,
                        "variable": var,
                        "unit": unit,
                    }
                    # fill year columns with corresponding values
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
        

        if outdir:
            scen_dir = outdir / scen_label
            scen_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(scen_dir / f"{sector}.csv", index=False)

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No rows produced for any scenario.")

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------
# End-of-life reporting
# ---------------------------------------------------------------------
def _eol_source_key_for_template(template: str) -> str:
    tl = template.lower()
    if "final material demand" in tl:
        if "|reused" in tl:
            return "reusable_materials"
        if "|recycled" in tl:
            return "recyclable_materials" # assuming that all recyclable will be recycled but this info should come from TIMER
        if "|virgin input" in tl:
            return "virgin_materials"
    if "material losses" in tl:
        return "losses_materials"
    if "scrap" in tl:
        return "sum_outflow"
    raise KeyError(f"Don’t know which EoL key to use for template: {template!r}")

def create_iamc_eol(
    models: Dict[str, object],
    templates: List[str],
    sector: str,
    model_name: str,
    outdir: str | None = None,
) -> pd.DataFrame:

    outdir_path = Path(outdir) if outdir else None
    if outdir_path:
        outdir_path.mkdir(parents=True, exist_ok=True)

    dfs = []

    for scen_label, src_model in models.items():
        all_rows: List[Dict[str, object]] = []   # ✅ reset per scenario

        eol = getattr(src_model, sector, None)
        if eol is None:
            continue

        for tpl in templates:
            # --- family (optional) ---
            try:
                family = _family_from_template(tpl)
            except ValueError:
                family = None

            # --- which source key in `eol` dict ---
            try:
                src_key = _eol_source_key_for_template(tpl)
            except KeyError:
                continue

            if src_key not in eol:
                continue

            da_raw = eol[src_key]
            da = da_raw if isinstance(da_raw, xr.DataArray) else da_raw.to_array()
            if hasattr(da, "pint"):
                da = da.pint.dequantify()

            # ✅ normalize time coord to int so _years() works
            tkey = _find_time_key(da)
            try:
                da = da.assign_coords({tkey: da.coords[tkey].astype(int)})
            except Exception:
                pass

            ysel = _years(da)
            if not ysel:
                # makes the failure explicit instead of silent empty columns
                raise RuntimeError(f"EoL: no report years found for tpl={tpl!r}. time coord={da.coords[tkey].values[:5]}...")

            regions = _labels(da, "Region")

            # --- unit + factor ---
            unit_tpl = _unit(da, family or "", tpl) if family else (da.attrs.get("unit") or "")
            try:
                factor = _conv_for(family or "", tpl)  # ✅ attempt even if family missing (falls back to 1.0)
            except Exception:
                factor = 1.0

            # --- placeholders (same as you had) ---
            ph_list = re.findall(r"\{([^}]+)\}", tpl)

            ph_specs = []
            if "Engineered Material" in ph_list:
                dim = "material"
                mats = _labels(da, dim)
                mat_map = {CFG.MATERIAL_NAME_MAP.get(str(m), str(m)): [m] for m in mats}
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

            for reg in regions:
                for combo in combos:
                    sel, sum_dims = {}, []
                    for ps, label in zip(ph_specs, combo):
                        sel[ps["dim"]] = ps["map"][label]
                        if ps["dim"] not in sum_dims:
                            sum_dims.append(ps["dim"])

                    part = da.sel(sel).sum(sum_dims, keep_attrs=True).sel({tkey: ysel})
                    vec = np.asarray(part.sel({"Region": reg}).values, dtype=float).reshape(-1) * factor

                    var = tpl
                    unit = unit_tpl

                    # ✅ replace placeholders in variable + unit (same logic as create_iamc_reporting)
                    for ps, label in zip(ph_specs, combo):
                        var = var.replace(f"{{{ps['ph']}}}", label)
                        if isinstance(unit, str) and "{" in unit:
                            unit = unit.replace(f"{{{ps['ph']}}}", label)

                    row = {
                        "model": model_name,
                        "scenario": scen_label,
                        "region": reg,
                        "variable": var,
                        "unit": unit,
                    }
                    row.update({int(y): float(v) for y, v in zip(ysel, vec)})
                    all_rows.append(row)

        if not all_rows:
            print(f"No EoL rows produced for {scen_label}.")
            continue

        df = pd.DataFrame(all_rows)
        fixed = ["model", "scenario", "region", "variable", "unit"]
        year_cols = sorted([c for c in df.columns if isinstance(c, int)])
        df = df[fixed + year_cols]

        # ✅ write csv
        if outdir_path:
            scen_dir = outdir_path / scen_label
            scen_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(scen_dir / f"{scen_label}_eol.csv", index=False)

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No EoL rows produced for any scenario.")

    return pd.concat(dfs, ignore_index=True)


def _safe_fname(s: str) -> str:
    # Replace Windows-illegal filename chars with underscore
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    # Also avoid trailing dots/spaces (Windows doesn't like those either)
    s = s.rstrip(" .")
    return s


# -------------------------------------------------------------
#                   Single variable reporting
#--------------------------------------------------------------

def export_iamc_csv(
    da: xr.DataArray,
    *,
    model: str,
    scenario: str,
    variable: str,
    unit: str,
    factor: float = 1.0,
    years: np.ndarray | None = None,
    fill: str = "interp_ffill_bfill",  # "none" | "zero" | "interp_ffill_bfill",
    outdir: str | None = None,
):
    """
    Export a DataArray to a iamc format:
    model, scenario, region, variable, unit, 2005, 2010, ..., 2100

    Assumes da has a region dim and a time dim only! da must be previously reduced by sum.
    Useful for adding extra variables without having the run the code, or to add variables resulting from post-processing calculations.
    """
    outdir = Path(outdir) if outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    # --- detect time dim ---
    time_dim = next((d for d in ("Time", "time") if d in da.dims), None)
    if time_dim is None:
        raise ValueError(f"Could not find a time dimension in {da.dims}.")

    # --- set default years (2005..2100 step 5) ---
    if years is None:
        years = np.arange(2005, 2101, 5)

    # --- reduce any extra dims (so output is [time, region]) ---
    extra_dims = [d for d in da.dims if d not in (time_dim, "Region")]
    if extra_dims:
        da = da.sum(dim=extra_dims, skipna=True)

    # --- coerce time coord to integer years if needed ---
    t = da[time_dim].values
    if np.issubdtype(np.asarray(t).dtype, np.datetime64):
        da = da.assign_coords({time_dim: pd.to_datetime(t).year.astype(int)})
    else:
        da = da.assign_coords({time_dim: da[time_dim].astype(int)})

    # --- reindex to full year grid ---
    da = da.reindex({time_dim: years})

    # --- fill missing values if requested ---
    if fill == "zero":
        da = da.fillna(0.0)
    elif fill == "interp_ffill_bfill":
        # interpolate along time, then forward/back fill any remaining gaps
        da = da.interpolate_na(dim=time_dim, method="linear", fill_value="extrapolate")
        da = da.ffill(time_dim).bfill(time_dim)
    elif fill == "none":
        pass
    else:
        raise ValueError("fill must be one of: 'none', 'zero', 'interp_ffill_bfill'")

    # --- apply conversion factor ---
    if factor is None:
        factor = 1.0
    da = da * float(factor)

    # --- to dataframe and pivot wide ---
    df = da.to_dataframe(name="value").reset_index()

    wide = (
        df.pivot_table(
            index="Region",
            columns=time_dim,
            values="value",
            aggfunc="sum",
        )
        .reset_index()
        .rename(columns={"Region": "region"})
    )

    # ensure column order: model, scenario, region, variable, unit, 2005..2100
    wide.insert(0, "unit", unit)
    wide.insert(0, "variable", variable)
    wide.insert(0, "scenario", scenario)
    wide.insert(0, "model", model)

    # make sure year columns are present and ordered
    year_cols = [y for y in years]
    for y in year_cols:
        if y not in wide.columns:
            wide[y] = np.nan
    wide = wide[["model", "scenario", "region", "variable", "unit", *year_cols]]
    if outdir:
        fname = _safe_fname(f"{variable}_{scenario}.csv")
        wide.to_csv(outdir / fname, index=False)
    return wide
