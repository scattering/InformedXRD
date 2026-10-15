#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
GEN_DIR = THIS_DIR.parent
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

import gen_rruff_conditioned as grc


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def profile_sample(task_id: int, base_seed: int) -> tuple[bool, dict[str, float], dict[str, int], str | None]:
    timings: dict[str, float] = {}
    counts: dict[str, int] = defaultdict(int)
    last_error = None

    t0 = time.perf_counter()
    task = grc.build_task(task_id, base_seed)
    timings["build_task"] = time.perf_counter() - t0
    _, eg, sg_main, sg_imp, worker_seed = task

    rng = np.random.default_rng(worker_seed)

    try:
        t = time.perf_counter()
        st_main = grc.get_valid_structure(sg_main)
        timings["get_valid_structure_main"] = time.perf_counter() - t
        counts["sg_main"] = int(sg_main)
        counts["eg_main"] = int(eg)
        if st_main is None:
            return False, timings, counts, "main_structure_failed"

        t = time.perf_counter()
        st_imp = grc.get_valid_structure(sg_imp)
        timings["get_valid_structure_imp"] = time.perf_counter() - t
        counts["sg_imp"] = int(sg_imp)
        if st_imp is None:
            return False, timings, counts, "imp_structure_failed"

        t = time.perf_counter()
        nuisance = grc.sample_empirical_nuisance(sg_main, rng)
        timings["sample_empirical_nuisance"] = time.perf_counter() - t

        t = time.perf_counter()
        sg_out, main_01 = grc.generate_pure_pattern_01(st_main, rng, nuisance, apply_distortion=True)
        timings["generate_main_pattern"] = time.perf_counter() - t
        counts["sg_out"] = int(sg_out)

        t = time.perf_counter()
        _, imp_01 = grc.generate_pure_pattern_01(st_imp, rng, nuisance, apply_distortion=False)
        timings["generate_imp_pattern"] = time.perf_counter() - t

        t = time.perf_counter()
        mixed_01 = grc.mix_impurity(main_01, imp_01, rng)
        intensity_counts = (mixed_01 * grc.MAX_INTENSITY).astype(np.float32)
        intensity_out = grc.add_rruff_background_noise(intensity_counts, grc.TWO_THETA, rng, nuisance, sg_out)
        timings["mix_and_noise"] = time.perf_counter() - t

        if not np.all(np.isfinite(intensity_out)):
            return False, timings, counts, "nonfinite_output"

        timings["total_sample"] = sum(timings.values())
        return True, timings, counts, None
    except Exception as exc:  # pragma: no cover - profiling path
        last_error = f"{type(exc).__name__}: {exc}"
        return False, timings, counts, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile gen_rruff_conditioned sample-generation throughput.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to attempt")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed for task stream")
    args = parser.parse_args()

    metric_values: dict[str, list[float]] = defaultdict(list)
    failure_reasons = Counter()
    eg_counts = Counter()
    sg_counts = Counter()
    success_count = 0

    print(f"[PROFILE] samples={args.samples} seed={args.seed}")
    print(f"[PROFILE] eg_mode={grc.EG_SAMPLING_MODE} prior_mode={grc.PRIOR_MODE}")
    print(f"[PROFILE] chemistry_csv={grc.ICSD_CHEMISTRY_CSV}")
    print(f"[PROFILE] rruff_prior_csv={grc.RRUFF_PRIOR_CSV}")

    overall_start = time.perf_counter()
    for task_id in range(args.samples):
        ok, timings, counts, error = profile_sample(task_id, args.seed)
        for key, value in timings.items():
            metric_values[key].append(value)
        if ok:
            success_count += 1
            if "eg_main" in counts:
                eg_counts[counts["eg_main"]] += 1
            if "sg_out" in counts:
                sg_counts[counts["sg_out"]] += 1
        else:
            failure_reasons[error or "unknown_failure"] += 1

    wall = time.perf_counter() - overall_start
    print(f"[PROFILE] successes={success_count}/{args.samples}")
    print(f"[PROFILE] wall_seconds={wall:.3f}")
    if success_count:
        print(f"[PROFILE] samples_per_hour={success_count / wall * 3600.0:.2f}")

    for metric in [
        "build_task",
        "get_valid_structure_main",
        "get_valid_structure_imp",
        "sample_empirical_nuisance",
        "generate_main_pattern",
        "generate_imp_pattern",
        "mix_and_noise",
        "total_sample",
    ]:
        stats = summarize(metric_values.get(metric, []))
        print(
            f"[PROFILE] {metric}: count={int(stats['count'])} "
            f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s max={stats['max']:.3f}s"
        )

    if failure_reasons:
        print("[PROFILE] failure_reasons:")
        for reason, count in failure_reasons.most_common():
            print(f"  - {reason}: {count}")

    if eg_counts:
        print("[PROFILE] sampled_egs:")
        for eg, count in eg_counts.most_common(10):
            print(f"  - EG {eg}: {count}")

    if sg_counts:
        print("[PROFILE] sampled_output_sgs:")
        for sg, count in sg_counts.most_common(10):
            print(f"  - SG {sg}: {count}")


if __name__ == "__main__":
    main()
