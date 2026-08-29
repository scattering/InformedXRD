## Information Theory Compute Tracking

Scope:
- only jobs launched for the `information-theory` branch work
- do not mix with the referee-response compute ledger

### Jobs

| Date | System | Job / purpose | Partition | Runtime | GPU job-hours | CPU job-hours | Notes |
|---|---|---|---|---:|---:|---:|---|
| 2026-05-30 | Vista | `730100` Gate 1 posterior dump for `RRUFF-325` / `9rwv1qly_best` | `gg` | `00:00:06` | 0.00 | 0.00 | failed immediately on `libpython3.11.so.1.0` shared-library environment issue |
| 2026-05-30 | Stampede3 | `3169101` Gate 1 posterior dump for `RRUFF-325` / `9rwv1qly_best` | `skx-dev` | `00:00:16` | 0.00 | 0.00 | successful fallback run; wrote full `325 x 99` posterior dump |
| 2026-05-31 | Stampede3 | `3169232` Gate 2 minimal channel first run | `spr` | `00:01:45` | 0.00 | 0.23 | failed on HKL payload normalization (`too many values to unpack`) |
| 2026-05-31 | Stampede3 | `3169240` Gate 2 minimal channel rerun | `spr` | `00:09:29` | 0.00 | 1.24 | got through generation but failed on final JSON serialization of NumPy arrays |
| 2026-05-31 | Stampede3 | `3169283` Gate 2 minimal channel rerun | `spr` | `00:09:29` | 0.00 | 1.24 | reached final write path after SG retry/skip fix but still failed before durable result |
| 2026-05-31 | Stampede3 | `3169406` Gate 2 minimal channel successful run | `spr` | `00:06:53` | 0.00 | 0.92 | successful end-to-end run; later diagnosis showed near-zero capacity was estimation-starved, not a weak-mask conclusion |
| 2026-05-31 | Stampede3 | `3169914` Gate 2 rerun with saved arrays for diagnosis | `spr` | `00:15:12` | 0.00 | 2.03 | wrote `.npz` sidecar for `(labels, symbols)` and enabled starvation-vs-mask diagnosis |
| 2026-05-31 | Stampede3 | `3169944` Gate 2 `10/class`, `alpha=1/729` | `spr` | `00:47:34` | 0.00 | 6.34 | first sane-smoothing scaled run; moved the channel out of the near-zero starvation regime |
| 2026-05-31 | Stampede3 | `3170098` Gate 2 `50/class`, production-backed parallel | `spr` | `00:11:59` | 0.00 | 6.39 | established the fast production-backed paired generation path |
| 2026-05-31 | Stampede3 | `3170154` Gate 2 shard `a`, `1000/class` | `spr` | `00:15:23` | 0.00 | 8.20 | completed shard `a` of the full merged `1000/class` campaign |
| 2026-05-31 | Stampede3 | `3170155` Gate 2 shard `b`, `1000/class` | `spr` | `00:18:07` | 0.00 | 9.66 | completed shard `b` of the full merged `1000/class` campaign |
| 2026-05-31 | Stampede3 | `3170156` Gate 2 shard `c`, `1000/class` | `spr` | `00:20:40` | 0.00 | 11.02 | completed shard `c` of the full merged `1000/class` campaign |
| 2026-05-31 | Stampede3 | `3170157` Gate 2 shard `d`, `1000/class` | `spr` | `00:56:14` | 0.00 | 29.99 | slow high-symmetry tail shard; completed and enabled the final merged Gate 2 result |
| 2026-05-31 | Stampede3 | `3170291` Gate 3 first cumulative ladder pilot | `spr` | `00:14:37` | 0.00 | 7.80 | validated ladder plumbing but remained non-monotone |
| 2026-05-31 | Stampede3 | `3170364` Gate 3 cumulative rerun | `spr` | `00:11:48` | 0.00 | 6.29 | fixed `ideal_mask` anchor but still had localized non-monotonicity |
| 2026-05-31 | Stampede3 | `3170431` Gate 3 soft-call cumulative rerun | `spr` | `00:11:34` | 0.00 | 6.15 | removed broadening→background inversion; left small final-step inversion |
| 2026-05-31 | Stampede3 | `3170490` Gate 3 reordered cumulative rerun | `spr` | `00:12:04` | 0.00 | 6.44 | moved attenuation-like rung upstream of background/noise and reduced final inversion to ~0.01 bits |
| 2026-05-31 | Stampede3 | `3170917` Rule-mask ladder pilot `s10` | `spr` | `00:02:15` | 0.00 | 4.20 | first realized rule-mask ladder artifact; later diagnosed as dominated by encoding-driven null |
| 2026-05-31 | Stampede3 | `3170933` Rule-mask ladder `s100` | `spr` | `00:10:55` | 0.00 | 20.38 | produced the `s100` rule-mask artifact used for the corrected `v3` physical-family analysis |
| 2026-06-01 | Stampede3 | `3171044` Rule-mask v2-aware ladder run | `spr` | `00:11:58` | 0.00 | 22.34 | first `v2`-aware generator run; cancelled after pathological tail in the last worker chunk |
| 2026-06-01 | Stampede3 | `3171055` Rule-mask v2-aware rerun with `chunksize=1` | `spr` | `00:09:12` | 0.00 | 17.17 | removed stranded-chunk failure mode but still slowed in the long retry tail |
| 2026-06-01 | Stampede3 | `3171067` Rule-mask v2-aware fast rerun with lower retry cap | `spr` | `00:11:45` | 0.00 | 21.93 | current production comparison surface for `rule_mask_v2`; run used reduced `MAX_ATTEMPTS` to avoid pathological tail blocking |
| 2026-06-01 | Stampede3 | `3173569` Gate 2 `RRUFF-473`-marginal continuous benchmark build | `spr` | `00:02:41` | 0.00 | 5.01 | successful CFML-backed `background_noise_like` benchmark generation for the continuous-to-mask control |
| 2026-06-01 | Vista | `732559` first `gh-dev` Gate 2 decoder control run | `gh-dev` | `00:00:16` | 0.00 | 0.00 | wrote decoder summary, failed after inference because `bayes_ceiling.py` was not staged on Vista |
| 2026-06-01 | Vista | `732560` `gh-dev` Gate 2 offset diagnostic | `gh-dev` | `00:00:15` | 0.00 | 0.00 | established that the raw one-based labels produce `10.78%` Top-1 while the old zero-based scorer produced the spurious `0%` result |
| 2026-06-01 | Vista | `732590` patched launcher Gate 2 rerun | `gh-dev` | `00:00:16` | 0.00 | 0.00 | still hit stale-analyzer staging; useful only as a failed validation step |
| 2026-06-01 | Vista | `732593` direct `gh-dev` Gate 2 corrected rerun | `gh-dev` | `00:00:17` | 0.00 | 0.00 | successful corrected same-surface comparison: discrete ceiling `67.38/85.71`, continuous `stage2c` `10.78/41.23` |
| 2026-06-01 | Stampede3 | `3172927` RRUFF-`209` validated-case Selling / MP overlap pass | `skx` | pending | 0.00 | 0.00 | queued search over the visually accepted `49` processed-overlap cases against the staged MP candidate JSONL; not yet countable until runtime lands |
| 2026-06-01 | Stampede3 | `3173105` RRUFF-`209` MP-overlap smoke after `s6dist` build fix | `skx-dev` | running | 0.00 | 0.00 | current smoke validates the rebuilt minimal `ncdist`-only `s6dist` extension and then exercises the `--max-cases 2` Python scan path |
| 2026-06-01 | Stampede3 | `3173117` RRUFF-`209` validated-case Selling / MP overlap production rerun | `spr` | pending (dependency) | 0.00 | 0.00 | production run chained `afterok:3173105`; will start automatically on `spr` if the smoke exits cleanly |

### Update rule

For each new information-theory run, append:
- job id
- exact purpose
- partition
- elapsed walltime
- whether it was GPU or CPU
- any notable failure mode or restart reason

### Current note

The information-theory branch is now using three distinct execution safeguards for
the rule-mask ladder surface:

- `chunksize=1` in the multiprocessing `imap_unordered` path
- tunable `MAX_ATTEMPTS` in the Stampede launcher
- checkpoint / resume support in `gate3_rule_mask_ladder.py`

So future long-tail rule-mask comparison runs should no longer require a full
restart after a late-stage stall.

Additional 2026-06-01 note:

- the decoder-floor suite (`stage2c` + `XRDBench`) was run as a direct shared-
  surface posterior analysis rather than as a scheduled cluster job, so it does
  not add a new scheduler-visible job-hours row here
- the new Selling / MP-overlap pass **has** been scheduled and is tracked above,
  but it should remain at zero job-hours until the queue run actually starts
