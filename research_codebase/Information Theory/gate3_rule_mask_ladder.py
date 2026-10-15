#!/usr/bin/env python3
"""
Generate paired realized rule-mask and six-family mask ladders on the same
simulated nuisance channel.

This is the comparison surface needed to answer the operational question:
does rule_mask_v1 retain a materially higher Bayes ceiling than the old
six-prototype mask after the same measurement degradation used in Gates 2/3?
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bayes_ceiling import bayes_topk_exact, fit_channel
from discrete_channel import encode_mask_vector
from gate2_minimal_channel import (
    CLEAN_NUISANCE,
    FAMILY_SPECS,
    family_mask_states,
    get_num_processes,
    get_valid_structure,
    load_eg_to_sgs,
    observed_pattern,
    reference_family_peaks,
    write_json,
    write_progress_json,
)
from gate3_physical_ladder import (
    apply_background_noise,
    apply_family_attenuation,
    apply_gaussian_broadening,
    family_mask_states_soft,
    ideal_mask_states,
    local_window_stats,
    structure_record,
)
from rule_mask_v2 import RULE_BASE_V2, RULE_REFLS_V2


RUNG_NAMES = [
    "ideal_mask",
    "structure_factor_like",
    "broadening_overlap_like",
    "rruff_like",
    "background_noise_like",
]


def _perm_closure(base):
    s = set()
    for hkl in base:
        for p in itertools.permutations(range(3)):
            s.add((hkl[p[0]], hkl[p[1]], hkl[p[2]]))
    return sorted(s)


RULE_BASE = [
    (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0),
    (1, 1, 0), (2, 1, 0), (2, 2, 0), (3, 1, 0), (4, 2, 0), (3, 3, 0), (4, 4, 0),
    (1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 1),
]
RULE_REFLS = _perm_closure(RULE_BASE)
STATE_TO_CODE = {"absent": 0, "present": 1, "unresolved": 2}


WORKER_CFG = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paired realized rule-mask ceilings under the Gate-3 nuisance ladder.")
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--samples-per-eg", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-attempts", type=int, default=150)
    parser.add_argument("--wavelength", type=float, default=1.54030)
    parser.add_argument("--tt-min", type=float, default=10.0)
    parser.add_argument("--tt-max", type=float, default=110.0)
    parser.add_argument("--tt-step", type=float, default=0.1)
    parser.add_argument("--present-threshold-frac", type=float, default=0.03)
    parser.add_argument("--overlap-window-deg-clean", type=float, default=0.12)
    parser.add_argument("--overlap-window-deg-degraded", type=float, default=0.32)
    parser.add_argument("--background-frac", type=float, default=0.03)
    parser.add_argument("--noise-frac", type=float, default=0.01)
    parser.add_argument("--orientation-sigma", type=float, default=0.35)
    parser.add_argument("--num-processes", type=int, default=0)
    parser.add_argument("--eg-index-start", type=int, default=0)
    parser.add_argument("--eg-index-stop", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--progress-json", default="")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--checkpoint-npz", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--save-structures", action="store_true")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-npz", default="")
    return parser.parse_args()


def build_tasks(eg_to_sgs, samples_per_eg: int, seed: int):
    task_id = 0
    for eg, sgs in eg_to_sgs:
        for rep in range(samples_per_eg):
            seq = np.random.SeedSequence([seed, int(eg), int(rep)])
            rng = np.random.default_rng(seq)
            sg = int(rng.choice(np.asarray(sgs, dtype=np.int64)))
            wseed = int(rng.integers(0, 2**63 - 1))
            yield (task_id, int(eg), sg, int(rep), wseed)
            task_id += 1


def init_worker(cfg):
    global WORKER_CFG
    WORKER_CFG = dict(cfg)


def generalized_reference_peaks(structure, wavelength: float, tt_min: float, tt_max: float, hkls: list[tuple[int, int, int]]) -> dict[tuple[int, int, int], list[float]]:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    pmg = AseAtomsAdaptor.get_structure(structure.to_ase())
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(pmg, two_theta_range=(tt_min, tt_max))
    wanted = {tuple(int(v) for v in hkl) for hkl in hkls}
    peaks: dict[tuple[int, int, int], list[float]] = {hkl: [] for hkl in hkls}

    def normalize(entry) -> tuple[int, int, int] | None:
        if isinstance(entry, dict):
            return normalize(entry.get("hkl"))
        if isinstance(entry, np.ndarray):
            entry = entry.tolist()
        if isinstance(entry, (list, tuple)):
            if len(entry) == 3 and all(np.isscalar(v) for v in entry):
                return tuple(abs(int(v)) for v in entry)
            if len(entry) > 0:
                return normalize(entry[0])
        return None

    for tt, hkls_at_tt in zip(pattern.x, pattern.hkls):
        seen_here = set()
        for entry in hkls_at_tt:
            hkl = normalize(entry)
            if hkl is None or hkl not in wanted or hkl in seen_here:
                continue
            peaks[hkl].append(float(tt))
            seen_here.add(hkl)
    return peaks


def reflection_mask_states(
    reference_peaks: dict[tuple[int, int, int], list[float]],
    x: np.ndarray,
    y: np.ndarray,
    *,
    overlap_window_deg: float,
    present_threshold_frac: float,
    soft: bool,
) -> dict[tuple[int, int, int], str]:
    ymax = float(np.max(y)) if y.size else 0.0
    global_threshold = present_threshold_frac * ymax

    centers: dict[tuple[int, int, int], float | None] = {}
    for hkl, refl_centers in reference_peaks.items():
        centers[hkl] = min(refl_centers) if refl_centers else None

    if y.size:
        diffs = np.diff(np.asarray(y, dtype=np.float64))
        global_noise = float(np.std(diffs) / np.sqrt(2.0)) if diffs.size else 0.0
    else:
        global_noise = 0.0

    states: dict[tuple[int, int, int], str] = {}
    for hkl in reference_peaks:
        center = centers.get(hkl)
        if center is None:
            states[hkl] = "absent"
            continue
        overlap = any(
            other != hkl and other_center is not None and abs(other_center - center) <= overlap_window_deg
            for other, other_center in centers.items()
        )
        peak, baseline = local_window_stats(x, y, center, overlap_window_deg)
        prominence = max(peak - baseline, 0.0)
        if soft:
            local_noise = max(global_noise, 0.05 * max(peak, baseline, 1e-8))
            snr = prominence / max(local_noise, 1e-8)
            decisive = (prominence >= global_threshold) and (snr >= 3.0)
            states[hkl] = "present" if decisive and not overlap else "unresolved"
        else:
            states[hkl] = "unresolved" if overlap else ("present" if peak >= global_threshold else "unresolved")
    return states


def canonicalize_rule_states(states: dict[tuple[int, int, int], str], refls: list[tuple[int, int, int]]) -> tuple[str, ...]:
    best = None
    for perm in itertools.permutations(range(3)):
        sig = tuple(
            states[(hkl[perm[0]], hkl[perm[1]], hkl[perm[2]])]
            for hkl in refls
        )
        codes = tuple(STATE_TO_CODE[s] for s in sig)
        if best is None or codes < best[0]:
            best = (codes, sig)
    assert best is not None
    return best[1]


def rule_state_codes(states: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(STATE_TO_CODE[state] for state in states)


def summarize_channel(labels: np.ndarray, symbols: np.ndarray, *, n_classes: int, alpha: float) -> dict:
    n_symbols = int(symbols.max()) + 1 if len(symbols) else 1
    W, prior = fit_channel(labels, symbols, n_classes=n_classes, n_symbols=n_symbols, alpha=alpha)
    return {
        "n_symbols_realized": int(n_symbols),
        "bayes_ceiling": bayes_topk_exact(W, prior),
    }


def checkpoint_payload(
    *,
    labels,
    accepted_task_ids,
    six_symbols_by_rung,
    rule_state_codes_by_rung,
    rule_state_codes_v2_by_rung,
    structure_records,
    save_structures: bool,
):
    payload = {
        "labels": np.asarray(labels, dtype=np.int64),
        "accepted_task_ids": np.asarray(accepted_task_ids, dtype=np.int64),
    }
    for name in RUNG_NAMES:
        payload[f"six_symbols_{name}"] = np.asarray(six_symbols_by_rung[name], dtype=np.int64)
        payload[f"rule_state_codes_{name}"] = np.asarray(rule_state_codes_by_rung[name], dtype=np.int16)
        payload[f"rule_state_codes_v2_{name}"] = np.asarray(rule_state_codes_v2_by_rung[name], dtype=np.int16)
    if save_structures:
        payload["structure_records"] = np.asarray(structure_records, dtype=object)
    return payload


def write_checkpoint(path: str, **kwargs):
    np.savez_compressed(path, **checkpoint_payload(**kwargs))


def load_checkpoint(path: str, save_structures: bool):
    d = np.load(path, allow_pickle=True)
    labels = np.asarray(d["labels"], dtype=np.int64).tolist()
    accepted_task_ids = np.asarray(d["accepted_task_ids"], dtype=np.int64).tolist()
    six_symbols_by_rung = {name: np.asarray(d[f"six_symbols_{name}"], dtype=np.int64).tolist() for name in RUNG_NAMES}
    rule_state_codes_by_rung = {
        name: [tuple(int(v) for v in row) for row in np.asarray(d[f"rule_state_codes_{name}"], dtype=np.int16)]
        for name in RUNG_NAMES
    }
    rule_state_codes_v2_by_rung = {
        name: [tuple(int(v) for v in row) for row in np.asarray(d[f"rule_state_codes_v2_{name}"], dtype=np.int16)]
        for name in RUNG_NAMES
    }
    structure_records = []
    if save_structures and "structure_records" in d:
        structure_records = np.asarray(d["structure_records"], dtype=object).tolist()
    return {
        "labels": labels,
        "accepted_task_ids": accepted_task_ids,
        "six_symbols_by_rung": six_symbols_by_rung,
        "rule_state_codes_by_rung": rule_state_codes_by_rung,
        "rule_state_codes_v2_by_rung": rule_state_codes_v2_by_rung,
        "structure_records": structure_records,
    }


def gate3_rule_worker(task):
    task_id, eg, sg, rep, seed = task
    rng = np.random.default_rng(seed)
    try:
        structure = get_valid_structure(sg, max_attempts=int(WORKER_CFG["max_attempts"]))
        ref_six = reference_family_peaks(
            structure,
            wavelength=float(WORKER_CFG["wavelength"]),
            tt_min=float(WORKER_CFG["tt_min"]),
            tt_max=float(WORKER_CFG["tt_max"]),
        )
        ref_rule = generalized_reference_peaks(
            structure,
            wavelength=float(WORKER_CFG["wavelength"]),
            tt_min=float(WORKER_CFG["tt_min"]),
            tt_max=float(WORKER_CFG["tt_max"]),
            hkls=RULE_REFLS,
        )
        ref_rule_v2 = generalized_reference_peaks(
            structure,
            wavelength=float(WORKER_CFG["wavelength"]),
            tt_min=float(WORKER_CFG["tt_min"]),
            tt_max=float(WORKER_CFG["tt_max"]),
            hkls=RULE_REFLS_V2,
        )

        x_clean, y_clean = observed_pattern(
            structure,
            wavelength=float(WORKER_CFG["wavelength"]),
            tt_min=float(WORKER_CFG["tt_min"]),
            tt_max=float(WORKER_CFG["tt_max"]),
            tt_step=float(WORKER_CFG["tt_step"]),
            nuisance=CLEAN_NUISANCE,
        )
        broadening_sigma = max(
            0.0,
            float(WORKER_CFG["overlap_window_deg_degraded"]) - float(WORKER_CFG["overlap_window_deg_clean"]),
        )
        y_broad = apply_gaussian_broadening(x_clean, y_clean, sigma_deg=broadening_sigma)
        y_rruff = apply_family_attenuation(
            x_clean,
            y_broad,
            reference_peaks=ref_rule,
            rng=rng,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            orientation_sigma=float(WORKER_CFG["orientation_sigma"]),
        )
        y_bg = apply_background_noise(
            y_rruff,
            rng=rng,
            background_frac=float(WORKER_CFG["background_frac"]),
            noise_frac=float(WORKER_CFG["noise_frac"]),
        )

        rung_patterns = {
            "ideal_mask": (x_clean, None),
            "structure_factor_like": (x_clean, y_clean),
            "broadening_overlap_like": (x_clean, y_broad),
            "rruff_like": (x_clean, y_rruff),
            "background_noise_like": (x_clean, y_bg),
        }

        six_states = {}
        rule_states = {}
        rule_states_v2 = {}
        for rung_name, (x_rung, y_rung) in rung_patterns.items():
            if rung_name == "ideal_mask":
                six_states[rung_name] = tuple(ideal_mask_states(ref_six))
                exact_rule = {hkl: ("absent" if not ref_rule.get(hkl) else "present") for hkl in RULE_REFLS}
                exact_rule_v2 = {hkl: ("absent" if not ref_rule_v2.get(hkl) else "present") for hkl in RULE_REFLS_V2}
                rule_states[rung_name] = canonicalize_rule_states(exact_rule, RULE_REFLS)
                rule_states_v2[rung_name] = canonicalize_rule_states(exact_rule_v2, RULE_REFLS_V2)
            elif rung_name in {"rruff_like", "background_noise_like"}:
                six_states[rung_name] = tuple(
                    family_mask_states_soft(
                        ref_six,
                        x_rung,
                        y_rung,
                        overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
                        present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
                    )
                )
                rule_states[rung_name] = canonicalize_rule_states(
                    reflection_mask_states(
                        ref_rule,
                        x_rung,
                        y_rung,
                        overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
                        present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
                        soft=True,
                    ),
                    RULE_REFLS,
                )
                rule_states_v2[rung_name] = canonicalize_rule_states(
                    reflection_mask_states(
                        ref_rule_v2,
                        x_rung,
                        y_rung,
                        overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
                        present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
                        soft=True,
                    ),
                    RULE_REFLS_V2,
                )
            else:
                overlap = (
                    float(WORKER_CFG["overlap_window_deg_clean"])
                    if rung_name == "structure_factor_like"
                    else float(WORKER_CFG["overlap_window_deg_degraded"])
                )
                six_states[rung_name] = tuple(
                    family_mask_states(
                        ref_six,
                        x_rung,
                        y_rung,
                        overlap_window_deg=overlap,
                        present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
                    )
                )
                rule_states[rung_name] = canonicalize_rule_states(
                    reflection_mask_states(
                        ref_rule,
                        x_rung,
                        y_rung,
                        overlap_window_deg=overlap,
                        present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
                        soft=False,
                    ),
                    RULE_REFLS,
                )
                rule_states_v2[rung_name] = canonicalize_rule_states(
                    reflection_mask_states(
                        ref_rule_v2,
                        x_rung,
                        y_rung,
                        overlap_window_deg=overlap,
                        present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
                        soft=False,
                    ),
                    RULE_REFLS_V2,
                )

        result = {
            "ok": True,
            "task_id": int(task_id),
            "eg": int(eg),
            "sg": int(sg),
            "rep": int(rep),
            "six_symbols": {name: int(encode_mask_vector(list(states))) for name, states in six_states.items()},
            "rule_state_codes": {name: rule_state_codes(states) for name, states in rule_states.items()},
            "rule_state_codes_v2": {name: rule_state_codes(states) for name, states in rule_states_v2.items()},
        }
        if WORKER_CFG.get("save_structures", False):
            result["structure_record"] = structure_record(structure)
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "task_id": int(task_id),
            "eg": int(eg),
            "sg": int(sg),
            "rep_target": int(rep),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    eg_to_sgs_all = load_eg_to_sgs(args.mapping_csv)
    eg_start = max(0, int(args.eg_index_start))
    eg_stop = int(args.eg_index_stop) if int(args.eg_index_stop) > 0 else len(eg_to_sgs_all)
    eg_to_sgs = eg_to_sgs_all[eg_start:eg_stop]
    n_classes = len(eg_to_sgs_all)
    num_processes = get_num_processes(args.num_processes)
    out_npz = args.out_npz.strip() or str(Path(args.out_json).with_suffix(".npz"))
    progress_every = max(1, int(args.progress_every))
    progress_json = args.progress_json.strip() or str(Path(args.out_json).with_suffix(".progress.json"))
    checkpoint_every = max(1, int(args.checkpoint_every))
    checkpoint_npz = args.checkpoint_npz.strip() or f"{out_npz}.checkpoint.npz"
    resume_from = args.resume_from.strip() or checkpoint_npz

    output = {
        "config": {
            "samples_per_eg": args.samples_per_eg,
            "seed": args.seed,
            "num_processes": num_processes,
            "eg_index_start": eg_start,
            "eg_index_stop": eg_stop,
            "n_classes_global": n_classes,
            "n_egs_in_shard": len(eg_to_sgs),
            "rung_names": list(RUNG_NAMES),
            "six_families": [name for name, _ in FAMILY_SPECS],
            "rule_reflections": [list(hkl) for hkl in RULE_REFLS],
            "rule_base_reflections": [list(hkl) for hkl in RULE_BASE],
            "rule_reflections_v2": [list(hkl) for hkl in RULE_REFLS_V2],
            "rule_base_reflections_v2": [list(hkl) for hkl in RULE_BASE_V2],
            "dirichlet_alpha": args.dirichlet_alpha,
            "checkpoint_npz": checkpoint_npz,
            "resume_from": resume_from,
        }
    }

    worker_cfg = {
        "max_attempts": args.max_attempts,
        "wavelength": args.wavelength,
        "tt_min": args.tt_min,
        "tt_max": args.tt_max,
        "tt_step": args.tt_step,
        "present_threshold_frac": args.present_threshold_frac,
        "overlap_window_deg_clean": args.overlap_window_deg_clean,
        "overlap_window_deg_degraded": args.overlap_window_deg_degraded,
        "background_frac": args.background_frac,
        "noise_frac": args.noise_frac,
        "orientation_sigma": args.orientation_sigma,
        "save_structures": bool(args.save_structures),
    }
    tasks = list(build_tasks(eg_to_sgs, samples_per_eg=args.samples_per_eg, seed=args.seed))
    labels = []
    accepted_task_ids = []
    six_symbols_by_rung = {name: [] for name in RUNG_NAMES}
    rule_state_codes_by_rung = {name: [] for name in RUNG_NAMES}
    rule_state_codes_v2_by_rung = {name: [] for name in RUNG_NAMES}
    dropped_examples = []
    structure_records = []

    if Path(resume_from).exists():
        restored = load_checkpoint(resume_from, save_structures=bool(args.save_structures))
        labels = restored["labels"]
        accepted_task_ids = restored["accepted_task_ids"]
        six_symbols_by_rung = restored["six_symbols_by_rung"]
        rule_state_codes_by_rung = restored["rule_state_codes_by_rung"]
        rule_state_codes_v2_by_rung = restored["rule_state_codes_v2_by_rung"]
        structure_records = restored["structure_records"]

    completed_task_ids = set(int(t) for t in accepted_task_ids)
    total_tasks = len(tasks)
    tasks = [task for task in tasks if int(task[0]) not in completed_task_ids]

    chunksize = 1
    if num_processes <= 1:
        init_worker(worker_cfg)
        iterator = (gate3_rule_worker(task) for task in tasks)
    else:
        pool = Pool(processes=num_processes, initializer=init_worker, initargs=(worker_cfg,))
        iterator = pool.imap_unordered(gate3_rule_worker, tasks, chunksize=chunksize)

    try:
        completed_tasks = len(accepted_task_ids)
        current_task_id = None
        for res in iterator:
            completed_tasks += 1
            current_task_id = int(res["task_id"])
            if res["ok"]:
                accepted_task_ids.append(int(res["task_id"]))
                labels.append(int(res["eg"]))
                for name in RUNG_NAMES:
                    six_symbols_by_rung[name].append(int(res["six_symbols"][name]))
                    rule_state_codes_by_rung[name].append(tuple(int(v) for v in res["rule_state_codes"][name]))
                    rule_state_codes_v2_by_rung[name].append(tuple(int(v) for v in res["rule_state_codes_v2"][name]))
                if args.save_structures:
                    structure_records.append(res["structure_record"])
            else:
                if len(dropped_examples) < 16:
                    dropped_examples.append({
                        "eg": int(res["eg"]),
                        "sg": int(res["sg"]),
                        "rep_target": int(res["rep_target"]),
                        "error_type": res["error_type"],
                        "error": res["error"],
                    })
            if (completed_tasks % checkpoint_every) == 0 or completed_tasks == total_tasks:
                write_checkpoint(
                    checkpoint_npz,
                    labels=labels,
                    accepted_task_ids=accepted_task_ids,
                    six_symbols_by_rung=six_symbols_by_rung,
                    rule_state_codes_by_rung=rule_state_codes_by_rung,
                    rule_state_codes_v2_by_rung=rule_state_codes_v2_by_rung,
                    structure_records=structure_records,
                    save_structures=bool(args.save_structures),
                )
            if (completed_tasks % progress_every) == 0 or completed_tasks == total_tasks:
                write_progress_json(
                    progress_json,
                    completed_tasks=completed_tasks,
                    total_tasks=total_tasks,
                    accepted_examples=len(labels),
                    dropped_attempts=completed_tasks - len(labels),
                    current_task_id=current_task_id,
                    eg_index_start=eg_start,
                    eg_index_stop=eg_stop,
                )
    finally:
        if num_processes > 1:
            pool.close()
            pool.join()

    labels = np.asarray(labels, dtype=np.int64)
    alpha = float(args.dirichlet_alpha) if args.dirichlet_alpha > 0 else 0.0
    npz_payload = {"labels": labels}

    output["six_prototype"] = {}
    output["rule_mask_v1"] = {}
    output["rule_mask_v2"] = {}
    for name in RUNG_NAMES:
        six_sym = np.asarray(six_symbols_by_rung[name], dtype=np.int64)
        rule_ids: list[int] = []
        rule_vocab: dict[tuple[int, ...], int] = {}
        for key in rule_state_codes_by_rung[name]:
            if key not in rule_vocab:
                rule_vocab[key] = len(rule_vocab)
            rule_ids.append(rule_vocab[key])
        rule_sym = np.asarray(rule_ids, dtype=np.int64)
        rule_ids_v2: list[int] = []
        rule_vocab_v2: dict[tuple[int, ...], int] = {}
        for key in rule_state_codes_v2_by_rung[name]:
            if key not in rule_vocab_v2:
                rule_vocab_v2[key] = len(rule_vocab_v2)
            rule_ids_v2.append(rule_vocab_v2[key])
        rule_sym_v2 = np.asarray(rule_ids_v2, dtype=np.int64)
        npz_payload[f"six_symbols_{name}"] = six_sym
        npz_payload[f"rule_symbols_{name}"] = rule_sym
        npz_payload[f"rule_vocab_{name}"] = np.asarray(list(rule_vocab.keys()), dtype=object)
        npz_payload[f"rule_symbols_v2_{name}"] = rule_sym_v2
        npz_payload[f"rule_vocab_v2_{name}"] = np.asarray(list(rule_vocab_v2.keys()), dtype=object)
        output["six_prototype"][name] = summarize_channel(labels, six_sym, n_classes=n_classes, alpha=alpha)
        output["rule_mask_v1"][name] = summarize_channel(labels, rule_sym, n_classes=n_classes, alpha=alpha)
        output["rule_mask_v2"][name] = summarize_channel(labels, rule_sym_v2, n_classes=n_classes, alpha=alpha)

    if args.save_structures:
        npz_payload["structure_space_group"] = np.asarray([r["space_group"] for r in structure_records], dtype=np.int16)
        npz_payload["structure_lattice"] = np.asarray([r["lattice"] for r in structure_records], dtype=np.float32)
        npz_payload["structure_symbols"] = np.asarray([np.asarray(r["symbols"], dtype=object) for r in structure_records], dtype=object)
        npz_payload["structure_scaled_positions"] = np.asarray([np.asarray(r["scaled_positions"], dtype=np.float32) for r in structure_records], dtype=object)

    output["sample_counts"] = {
        "n_examples": int(len(labels)),
        "n_dropped_attempts": int(total_tasks - len(labels)),
        "dropped_preview": dropped_examples,
    }
    output["artifacts"] = {
        "out_json": str(args.out_json),
        "out_npz": str(out_npz),
        "progress_json": str(progress_json),
        "checkpoint_npz": str(checkpoint_npz),
    }
    np.savez_compressed(out_npz, **npz_payload)
    write_json(args.out_json, output)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
