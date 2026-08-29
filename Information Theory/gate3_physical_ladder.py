#!/usr/bin/env python3
"""
Generate a paired multi-rung physical ladder for Gate 3.

This is an information-theory variant only. It leaves the production generator
untouched and reuses the existing Gate-2 helper machinery:

- same accepted structure shared across all rungs
- same task/EG indexing across all rungs
- rung symbols saved side-by-side for paired DPI-style analysis

The current default ladder is intentionally minimal but cumulative on a single
realized intensity trace:

1. ideal_mask
2. structure_factor_like
3. broadening_overlap_like
4. rruff_like
5. background_noise_like

The later rungs are approximate pattern-space degradations, not final paper
physics. Their purpose is to establish a literal downstream chain
`G -> X0 -> X1 -> ...` and produce reusable rung arrays for
`ladder_waterfall.py`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from discrete_channel import encode_mask_vector, estimate_channel, full_summary
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

from multiprocessing import Pool


RUNG_NAMES = [
    "ideal_mask",
    "structure_factor_like",
    "broadening_overlap_like",
    "rruff_like",
    "background_noise_like",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paired Gate-3 ladder masks from shared structures.")
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--samples-per-eg", type=int, default=10)
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


def ideal_mask_states(reference_peaks: dict[str, list[float]]) -> list[str]:
    states = []
    for fam, _proto in FAMILY_SPECS:
        states.append("present" if reference_peaks.get(fam) else "absent")
    return states


def apply_background_noise(y: np.ndarray, rng: np.random.Generator, background_frac: float, noise_frac: float) -> np.ndarray:
    ymax = float(np.max(y)) if y.size else 0.0
    if ymax <= 0:
        return y.copy()
    grid = np.linspace(0.0, 1.0, len(y), dtype=np.float64)
    baseline = ymax * background_frac * (0.25 + 0.55 * grid + 0.20 * grid * grid)
    noise = rng.normal(0.0, noise_frac * ymax, size=len(y))
    return np.clip(y + baseline + noise, 0.0, None)


def apply_gaussian_broadening(x: np.ndarray, y: np.ndarray, sigma_deg: float) -> np.ndarray:
    if sigma_deg <= 0 or y.size == 0:
        return y.copy()
    if x.size < 2:
        return y.copy()
    step = float(np.median(np.diff(x)))
    if not np.isfinite(step) or step <= 0:
        return y.copy()
    sigma_pts = float(sigma_deg / step)
    if sigma_pts <= 0.25:
        return y.copy()
    radius = max(1, int(np.ceil(4.0 * sigma_pts)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma_pts) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def local_window_stats(x: np.ndarray, y: np.ndarray, center: float, half_window: float) -> tuple[float, float]:
    mask = (x >= center - half_window) & (x <= center + half_window)
    if not np.any(mask):
        return 0.0, 0.0
    vals = np.asarray(y[mask], dtype=np.float64)
    peak = float(np.max(vals))
    baseline = float(np.median(vals))
    return peak, baseline


def family_mask_states_soft(
    reference_peaks: dict[str, list[float]],
    x: np.ndarray,
    y: np.ndarray,
    *,
    overlap_window_deg: float,
    present_threshold_frac: float,
) -> list[str]:
    states: list[str] = []
    ymax = float(np.max(y)) if y.size else 0.0
    global_threshold = present_threshold_frac * ymax

    family_centers: dict[str, float | None] = {}
    for fam, centers in reference_peaks.items():
        family_centers[fam] = min(centers) if centers else None

    if y.size:
        diffs = np.diff(np.asarray(y, dtype=np.float64))
        global_noise = float(np.std(diffs) / np.sqrt(2.0)) if diffs.size else 0.0
    else:
        global_noise = 0.0

    for fam, _proto in FAMILY_SPECS:
        center = family_centers[fam]
        if center is None:
            states.append("absent")
            continue

        overlap = any(
            other != fam and other_center is not None and abs(other_center - center) <= overlap_window_deg
            for other, other_center in family_centers.items()
        )
        peak, baseline = local_window_stats(x, y, center, overlap_window_deg)
        prominence = max(peak - baseline, 0.0)
        # Use prominence and a local noise proxy so additive background/noise can
        # only flatten the decision rather than sharpen it via a hard absolute threshold.
        local_noise = max(global_noise, 0.05 * max(peak, baseline, 1e-8))
        snr = prominence / max(local_noise, 1e-8)
        decisive = (prominence >= global_threshold) and (snr >= 3.0)
        states.append("present" if decisive and not overlap else "unresolved")
    return states


def apply_family_attenuation(
    x: np.ndarray,
    y: np.ndarray,
    reference_peaks: dict[str, list[float]],
    rng: np.random.Generator,
    overlap_window_deg: float,
    orientation_sigma: float,
) -> np.ndarray:
    out = y.copy()
    for fam, centers in reference_peaks.items():
        if not centers:
            continue
        center = min(centers)
        window = (x >= center - overlap_window_deg) & (x <= center + overlap_window_deg)
        if not np.any(window):
            continue
        # Keep this as a pure information-loss transform. Allow attenuation but
        # never amplification, so the rung stays downstream of its parent.
        weight = float(np.clip(np.exp(-abs(rng.normal(0.0, orientation_sigma))), 0.2, 1.0))
        out[window] *= weight
    return np.clip(out, 0.0, None)


def structure_record(structure) -> dict:
    atoms = structure.to_ase()
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    return {
        "space_group": int(structure.group.number),
        "lattice": [float(a), float(b), float(c), float(alpha), float(beta), float(gamma)],
        "symbols": list(atoms.get_chemical_symbols()),
        "scaled_positions": np.asarray(atoms.get_scaled_positions(), dtype=np.float32),
    }


WORKER_CFG = {}


def init_worker(cfg):
    global WORKER_CFG
    WORKER_CFG = dict(cfg)


def gate3_worker(task):
    task_id, eg, sg, rep, seed = task
    rng = np.random.default_rng(seed)
    try:
        structure = get_valid_structure(sg, max_attempts=int(WORKER_CFG["max_attempts"]))
        ref = reference_family_peaks(
            structure,
            wavelength=float(WORKER_CFG["wavelength"]),
            tt_min=float(WORKER_CFG["tt_min"]),
            tt_max=float(WORKER_CFG["tt_max"]),
        )
        rung_states = {}
        rung_states["ideal_mask"] = ideal_mask_states(ref)

        x_clean, y_clean = observed_pattern(
            structure,
            wavelength=float(WORKER_CFG["wavelength"]),
            tt_min=float(WORKER_CFG["tt_min"]),
            tt_max=float(WORKER_CFG["tt_max"]),
            tt_step=float(WORKER_CFG["tt_step"]),
            nuisance=CLEAN_NUISANCE,
        )
        rung_states["structure_factor_like"] = family_mask_states(
            ref,
            x_clean,
            y_clean,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_clean"]),
            present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
        )
        broadening_sigma = max(
            0.0,
            float(WORKER_CFG["overlap_window_deg_degraded"]) - float(WORKER_CFG["overlap_window_deg_clean"]),
        )
        y_broad = apply_gaussian_broadening(x_clean, y_clean, sigma_deg=broadening_sigma)
        rung_states["broadening_overlap_like"] = family_mask_states(
            ref,
            x_clean,
            y_broad,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
        )

        y_rruff = apply_family_attenuation(
            x_clean,
            y_broad,
            reference_peaks=ref,
            rng=rng,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            orientation_sigma=float(WORKER_CFG["orientation_sigma"]),
        )
        rung_states["rruff_like"] = family_mask_states_soft(
            ref,
            x_clean,
            y_rruff,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
        )

        y_bg = apply_background_noise(
            y_rruff,
            rng=rng,
            background_frac=float(WORKER_CFG["background_frac"]),
            noise_frac=float(WORKER_CFG["noise_frac"]),
        )
        rung_states["background_noise_like"] = family_mask_states_soft(
            ref,
            x_clean,
            y_bg,
            overlap_window_deg=float(WORKER_CFG["overlap_window_deg_degraded"]),
            present_threshold_frac=float(WORKER_CFG["present_threshold_frac"]),
        )

        result = {
            "ok": True,
            "task_id": int(task_id),
            "eg": int(eg),
            "sg": int(sg),
            "rep": int(rep),
            "rung_states": rung_states,
            "rung_symbols": {name: int(encode_mask_vector(states)) for name, states in rung_states.items()},
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
    n_symbols = 3 ** len(FAMILY_SPECS)
    num_processes = get_num_processes(args.num_processes)
    out_npz = args.out_npz.strip() or str(Path(args.out_json).with_suffix(".npz"))
    progress_every = max(1, int(args.progress_every))
    progress_json = args.progress_json.strip() or str(Path(args.out_json).with_suffix(".progress.json"))

    output = {
        "config": {
            "samples_per_eg": args.samples_per_eg,
            "seed": args.seed,
            "num_processes": num_processes,
            "eg_index_start": eg_start,
            "eg_index_stop": eg_stop,
            "n_classes_global": n_classes,
            "n_egs_in_shard": len(eg_to_sgs),
            "families": [name for name, _ in FAMILY_SPECS],
            "rung_names": list(RUNG_NAMES),
            "n_symbols": n_symbols,
            "dirichlet_alpha": args.dirichlet_alpha,
            "present_threshold_frac": args.present_threshold_frac,
            "overlap_window_deg_clean": args.overlap_window_deg_clean,
            "overlap_window_deg_degraded": args.overlap_window_deg_degraded,
            "background_frac": args.background_frac,
            "noise_frac": args.noise_frac,
            "orientation_sigma": args.orientation_sigma,
            "wavelength": args.wavelength,
            "tt_min": args.tt_min,
            "tt_max": args.tt_max,
            "tt_step": args.tt_step,
            "save_structures": bool(args.save_structures),
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
    symbols_by_rung = {name: [] for name in RUNG_NAMES}
    rung_state_counts = {name: {"present": 0, "absent": 0, "unresolved": 0} for name in RUNG_NAMES}
    dropped_examples = []
    example_rows = []
    structure_records = []

    chunksize = 8
    if num_processes <= 1:
        init_worker(worker_cfg)
        iterator = (gate3_worker(task) for task in tasks)
    else:
        pool = Pool(processes=num_processes, initializer=init_worker, initargs=(worker_cfg,))
        iterator = pool.imap_unordered(gate3_worker, tasks, chunksize=chunksize)
    try:
        completed_tasks = 0
        current_task_id = None
        for res in iterator:
            completed_tasks += 1
            current_task_id = int(res["task_id"])
            if res["ok"]:
                labels.append(int(res["eg"]))
                for name in RUNG_NAMES:
                    symbols_by_rung[name].append(int(res["rung_symbols"][name]))
                    for state in res["rung_states"][name]:
                        rung_state_counts[name][state] += 1
                if len(example_rows) < 8:
                    example_rows.append({
                        "eg": int(res["eg"]),
                        "sg": int(res["sg"]),
                        "rep": int(res["rep"]),
                        "rung_states": res["rung_states"],
                    })
                if args.save_structures:
                    structure_records.append(res["structure_record"])
            else:
                if len(dropped_examples) < 8:
                    dropped_examples.append({
                        "eg": int(res["eg"]),
                        "sg": int(res["sg"]),
                        "rep_target": int(res["rep_target"]),
                        "error_type": res["error_type"],
                        "error": res["error"],
                    })
            if (completed_tasks % progress_every) == 0 or completed_tasks == len(tasks):
                write_progress_json(
                    progress_json,
                    completed_tasks=completed_tasks,
                    total_tasks=len(tasks),
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
    npz_payload = {"labels": labels}
    for name in RUNG_NAMES:
        sym = np.asarray(symbols_by_rung[name], dtype=np.int64)
        npz_payload[f"symbols_{name}"] = sym
        rung_W, rung_warn = estimate_channel(
            labels,
            sym,
            n_classes=n_classes,
            n_symbols=n_symbols,
            dirichlet_alpha=args.dirichlet_alpha,
        )
        output[name] = {
            "mask_state_counts": rung_state_counts[name],
            "n_examples": int(len(labels)),
            "n_dropped_attempts": int(len(tasks) - len(labels)),
            "warning": rung_warn,
            "summary": full_summary(rung_W, np.full(n_classes, 1.0 / n_classes, dtype=np.float64), run_capacity=True),
            "dropped_preview": dropped_examples,
            "example_rows_preview": example_rows,
        }
    if args.save_structures:
        npz_payload["structure_space_group"] = np.asarray([r["space_group"] for r in structure_records], dtype=np.int16)
        npz_payload["structure_lattice"] = np.asarray([r["lattice"] for r in structure_records], dtype=np.float32)
        npz_payload["structure_symbols"] = np.asarray([np.asarray(r["symbols"], dtype=object) for r in structure_records], dtype=object)
        npz_payload["structure_scaled_positions"] = np.asarray([np.asarray(r["scaled_positions"], dtype=np.float32) for r in structure_records], dtype=object)

    np.savez_compressed(out_npz, **npz_payload)
    output["artifacts"] = {
        "out_json": str(args.out_json),
        "out_npz": str(out_npz),
        "progress_json": str(progress_json),
    }
    write_json(args.out_json, output)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
