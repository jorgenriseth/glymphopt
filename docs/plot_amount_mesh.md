# Plot mesh solute amount

From the project directory, run a comparison for `a = 1, 2, 3, 5`:

```bash
bash scripts/run_amount_sweep.sh \
  mri_processed_data/sub-01/modeling/resolution32/data.hdf \
  results/amount_sweep
```

The script runs each simulation with `r=1e-5` and `dt=3600` seconds, then
creates one CSV per `a`, `measured_amount.csv`, and
`amount_comparison.png`. Supply different `a` values by appending them to the
command, for example `... results/amount_sweep 1 3 10`.

The input is glymphopt's processed `data.hdf`. Simulation CSVs contain `time`
(seconds) and `amount_mesh` (mmol) at every timestep. The figure converts time
to hours and shows measured tissue amounts as crosses. Amount is the mesh
concentration integral (`sum(M @ Y) * 1e-6`); mmol assumes mesh coordinates in
mm and concentration in mmol/L. It is total mesh amount, not average CSF
concentration.

To run one simulation and plot its CSV separately:

```bash
pixi run python scripts/singlecompartment_amount.py \
  -i mri_processed_data/sub-01/modeling/resolution32/data.hdf \
  -o amount.csv --data-output measured.csv --a 3 --r 1e-5 --dt 3600
pixi run python scripts/plot_amount_mesh.py \
  -i amount.csv --label "a=3" --data measured.csv -o amount.png
```

`data.hdf` must already exist. The Snakefile can generate it from prepared
source data; building it also requires `evaluation_data.npz`, even though the
simulation itself does not.
