# Collapse, Convergence, and the Scale of the Prime-Gap Imprint on Polymer Dynamics

Manuscript, scripts and data for

> R. Chen, *Collapse, Convergence, and the Scale of the Prime-Gap Imprint on Polymer Dynamics* (2026).
> DOI: [10.5281/zenodo.23076433](https://doi.org/10.5281/zenodo.23076433).
> Repository: https://github.com/Ruqing1963/prime-gap-imprint-collapse-and-scale

**Two results.** (1) The ordering imprint of the prime-gap sequence on the collapsed
prime-driven topological polymer (PDTP) survives complete relaxation: Δρ∞ = −0.0269 ± 0.0033
at N = 4×10⁴ from two independent routes (8× slower annealing; 10⁶-step isothermal hold into a
melt-density globule, R_g = 16.81 ± 0.03), and the gap is present at every size 10⁴–8×10⁴ under
the converged protocol. (2) The size-proportional correlation length ξ ∝ n_gap of the S5
local-scramble surrogate is a property of the readout, not of the primes: the retention-scaling
theorem of PDTP v4.0 predicts it for any chain-scale readout, and surrogates with no
prime-specific arithmetic (S6′, S6″) reproduce the law and its coefficient in the exact
ideal-chain readout (master-curve ξ/n_gap = 0.30, 0.30, 0.33 for true, S6′, S6″). The earlier,
unpublished inference of "extensive arithmetic order" is withdrawn. The smooth PNT envelope
supplies f_S6′ = 0.25, 0.34, 0.57 of the effect at N = 2, 4, 8×10⁴; the residual is significant
at the sequence level (10 fixed S6′ sequences × 5 trajectories: Z_seq = −3.75, rank 0/10).

This paper merges and supersedes two unpublished drafts: the Letter *Scale-free arithmetic
order in the prime numbers revealed by polymer folding* and *The arithmetic imprint survives
collapse*. It builds on the PDTP paper, version 4.0
(DOI [10.5281/zenodo.22944122](https://doi.org/10.5281/zenodo.22944122);
[prime-driven-topological-polymer](https://github.com/Ruqing1963/prime-driven-topological-polymer)),
whose exact solver (`core.py`) and Langevin engine (`pdtp_langevin.py`) are included here,
unmodified, in `code/`.

## Layout

```
code/
  core.py                                exact ideal-chain solver and surrogate library (PDTP v4.0)
  pdtp_langevin.py                       PDTP force field and annealing (PDTP v4.0)
paper/
  merged_paper.tex, merged_paper.pdf     REVTeX 4-2 manuscript (9 pages)
  figures/                               fig1_protocol, fig2_metastability, fig3_globule
                                         (convergence), fig1_scale_free (production ξ),
                                         fig5_collapse (retention master curves)
scripts/
  ideal_chain_exact.py                   exact ideal-chain ξ for True / S6′ / S6″, bootstrap,
                                         master-curve collapse  (Table V, Fig. 5a,b)
  random_effects_s6prime.py              sequence-level (random-effects) S6′ test, Langevin,
                                         K fixed sequences × m trajectories  (Table VII)
  make_fig_collapse.py                   Fig. 5
  protocol_stability_test.py             original driver of Sec. III: Stage B anneal sweep and
                                         Stage C isothermal hold  (Tables I-II, Figs. 1-3)
  s5_converged_scaling.py                original driver of the converged S5 and S6' ladders
                                         (Tables III, IV, VI, VIII)
data/
  protocol_independence_results.csv      Stage B: N = 4e4, 1x-8x anneals, n = 40 per arm
  metastability_traces.csv               Stage C: 1e6-step isothermal traces, n = 5 per arm
  tau_calibration.csv, converged_protocol.json   relaxation-time calibration per size
  s5_scaling_results.csv, s5_scalefree_ext_results.csv   production-protocol S5 ladders
  pnt_decoupling_results.csv                             production S6′ decomposition
  s5_converged_results.csv                               32000-step True / S1 / S5 ladders
  s6prime_scale_control.csv                              32000-step S6′ / S6′-S5 ladders
  ideal_exact_summary.csv, ideal_exact_retention.csv, ideal_exact_collapse.csv
  random_effects_s6prime.csv, random_effects_summary.csv, random_effects_run.log
```

The ten relaxed endpoint conformations of the isothermal hold are in the PDTP v4.0 repository
(`data/v4/relaxed_endpoints/`).

## Reproduce

```bash
pip install -r requirements.txt
python scripts/ideal_chain_exact.py            # minutes; Table V, data/ideal_exact_*.csv
python scripts/make_fig_collapse.py            # Fig. 5
PDTP_ANALYZE_ONLY=1 python scripts/random_effects_s6prime.py   # Table VII from the CSV
PDTP_WORKERS=12 python scripts/random_effects_s6prime.py       # rerun: ~1.6 h on 12 cores
cd data && PDTP_ANALYZE_ONLY=1 python ../scripts/protocol_stability_test.py   # Sec. III from the CSVs
cd data && PDTP_ANALYZE_ONLY=1 python ../scripts/s5_converged_scaling.py      # converged ladders
```

The two original drivers read and write their CSVs in the working directory, so run them from
`data/`. The scripts written for this paper seed every random draw with `zlib.crc32` of a label
string and are reproducible bit for bit. `s5_converged_scaling.py` seeds with Python's `hash()` of
a string, which is randomized per process, so its trajectories cannot be regenerated bit for bit;
its released CSVs are the record.

## License

MIT.
