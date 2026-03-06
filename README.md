# Undamaged Case — FEA Beam Simulation Framework

This codebase implements a Finite Element Analysis (FEA) simulation pipeline for an Euler-Bernoulli beam under **undamaged (baseline) conditions**. It generates large batches of synthetic structural vibration datasets consisting of acceleration time histories at 101 virtual sensor nodes, sweeping across beam geometries, materials, boundary conditions, loading scenarios, and noise levels. The output CSV files serve as healthy-state reference data for structural health monitoring (SHM) research and as training data for machine learning models. All computations use the IPS unit system (inches, pound-force, seconds).

---

## File Structure and Code Description

### `config.py` — Central Configuration Registry

The single source of truth for all simulation parameters. Every other module imports constants from here. No computation is performed; it is a pure settings file.

Key parameter groups:
- **Geometry**: `THICKNESS_VALUES`, `LENGTH_VALUES`, `WIDTH_VALUES` — discrete lists of beam dimensions (in).
- **Loading**: `FORCE_VALUES`, `PULSE_DURATION`, `LOADING_TYPES`, `IMPACT_POSITION_VALUES`, `EXCITATION_FREQ_VALUES`, `RANDOM_BANDWIDTH` — controls impact, harmonic, and band-limited random excitation.
- **Time discretization**: `T_TOTAL = 1.0 s`, `N_STEPS = 2000`, `DT` — matched to a 2000 fps acquisition rate.
- **Mesh**: `N_ELEMENTS = 100`, `N_NODES = 101`.
- **Damping**: `RAYLEIGH_ALPHA`, `RAYLEIGH_BETA`, `RAYLEIGH_DAMPING_PAIRS`.
- **Solver**: `NEWMARK_BETA = 0.25`, `NEWMARK_GAMMA = 0.50` (unconditionally stable average-acceleration method).
- **Boundary conditions**: `cantilever`, `simply_supported`, `fixed_fixed`.
- **Noise**: `SNR_VALUES = [40, 60]` dB, `EXPORT_MODE = 'noisy'`.
- **Materials**: `['steel', 'aluminum']`.

Unlike the damaged variant, this configuration contains no damage modeling section — stiffness reductions are never applied and all beams remain structurally intact throughout.

---

### `materials.py` — Material Registry

Defines elastic and mass properties for steel and aluminum in IPS units. Density values are stored in both conventional units (lbm/in³) and consistent FEA units (lbf·s²/in⁴) via division by `G_C = 386.088 in/s²`.

Key functions:
- `get_material(name)` — returns a dict with keys `E`, `rho`, `rho_lbm`, `nu` for a named material.
- `add_material(name, E, rho_lbm, nu)` — registers a new material at runtime with automatic unit conversion.
- `list_materials()` — prints a formatted table of all registered materials.

---

### `beam_element.py` — Element-Level FEA Formulation

Implements the two-node Euler-Bernoulli beam finite element. Each element has 4 degrees of freedom (DOFs): transverse displacement `w` and rotation `θ` at each of its two nodes.

Key functions:
- `element_stiffness(E, I, Le)` — returns the classical 4×4 stiffness matrix using coefficients derived from `E*I/Le³`.
- `element_mass(rho, A, Le)` — returns the 4×4 consistent mass matrix using Hermite cubic interpolation, with leading factor `rho*A*Le/420`.
- `compute_section_properties(b, t)` — returns cross-sectional area `A = b*t` and second moment of area `I = b*t³/12` for a rectangular section.

---

### `assembly.py` — Global Matrix Assembly and Boundary Conditions

Handles mesh generation, direct stiffness assembly of the 202×202 global matrices, and static condensation (boundary condition enforcement). Uses `scipy.sparse` throughout for memory efficiency. Since no damage is modeled, all 100 elements use the same elastic modulus `E` and produce identical stiffness matrices.

Key functions:
- `generate_mesh(L)` — returns 101 equally spaced node coordinates and the uniform element length `Le`.
- `assemble_global_matrices(E, rho, b, t, L)` — assembles the global stiffness and mass matrices by looping over 100 identical elements and scattering their 4×4 contributions into the global 202×202 system via the direct stiffness method. Uses `lil_matrix` for assembly, then converts to `csr_matrix`.
- `apply_boundary_conditions(K_global, M_global, bc_type)` — removes fixed DOFs to produce the reduced free-DOF matrices `K_free`, `M_free`, and the `free_dofs` index array. Supports cantilever (2 fixed DOFs), simply-supported (2 fixed DOFs), and fixed-fixed (4 fixed DOFs).

---

### `damping.py` — Rayleigh Damping

Constructs the proportional damping matrix from the assembled mass and stiffness matrices.

Key functions:
- `build_rayleigh_damping(M_free, K_free, alpha, beta)` — computes `C = alpha*M + beta*K` where `alpha` (1/s) is mass-proportional and `beta` (s) is stiffness-proportional. Returns a `csr_matrix`.
- `rayleigh_from_damping_ratios(zeta1, zeta2, omega1, omega2)` — utility that back-calculates `alpha` and `beta` from physical damping ratios at two known natural frequencies by solving a 2×2 linear system. Useful for engineering calibration (e.g., "2% damping at the first two modes").

---

### `excitation.py` — Force Generation and Application

Generates force time histories for all three supported loading scenarios and distributes the load correctly to the FE DOF system using Hermite shape function interpolation.

Key functions:
- `half_sine_pulse(F0, tau, dt, n_steps)` — generates `F(t) = F0*sin(π*t/tau)` for `t ≤ tau`, zero otherwise. Models a hammer impact.
- `harmonic_load(F0, f_exc, dt, n_steps)` — generates a continuous sinusoidal load `F(t) = F0*sin(2π*f_exc*t)`.
- `random_load(F0, dt, n_steps, f_band, seed)` — generates band-limited Gaussian white noise in the frequency domain via FFT and scales the RMS to `F0`.
- `hermite_shape_functions(xi, Le)` — evaluates the 4 Hermite cubic shape functions at local coordinate `xi ∈ [0,1]`.
- `get_impact_dofs_and_weights(position_frac, L, free_dofs)` — distributes a point load at an arbitrary fractional beam position to the 4 DOFs of the containing element using Hermite weights, avoiding the error of snapping to the nearest node.
- `build_force_vector(F0, free_dofs, L, ...)` — master function that returns the complete force array of shape `(n_free_dofs, n_steps)` for all loading types and positions.
- `get_impact_dof(bc_type)` — legacy function that returns a hard-coded single DOF index (tip node for cantilever; node 22 for other BCs to avoid even-mode nodal points).

---

### `time_integrator.py` — Newmark-Beta Solver

Implements direct time integration of the second-order equation of motion `M·ü + C·u̇ + K·u = F(t)`.

Key function:
- `newmark_beta_solver(M_free, C_free, K_free, F_global, dt, n_steps, beta, gamma)` — uses the average acceleration method (β=0.25, γ=0.5), which is unconditionally stable. The effective stiffness matrix `K_eff = K + (γ/β·Δt)·C + (1/β·Δt²)·M` is precomputed once before the loop. Each timestep solves one sparse linear system via `spsolve`. Returns `accel_history` of shape `(n_free_dofs, n_steps)` in in/s².

---

### `sensors.py` — Sensor Post-Processing

Converts the FE solver output (accelerations at free DOFs, including interleaved rotational DOFs) into a clean physical sensor array.

Key functions:
- `extract_node_accelerations(accel_history, free_dofs, n_nodes)` — maps transverse accelerations (even-indexed global DOFs) back to a `(101, n_steps)` array. Fixed nodes receive zero acceleration. Rotational DOFs are discarded, mimicking real accelerometers that measure translational motion only.
- `get_node_labels(n_nodes)` — returns column header strings `['node_001_accel', ..., 'node_101_accel']` for CSV export.
- `serialize_node_accel(accel_array, delimiter)` — joins a 1D array into a semicolon-delimited string for Encoding A CSV format.

---

### `noise.py` — Sensor Noise Model

Adds independent Gaussian white noise to each sensor channel to simulate measurement imperfection.

Key function:
- `add_sensor_noise(node_accels, snr_db, seed)` — for each of the 101 nodes, computes the signal RMS, derives the required noise RMS from `RMS_noise = RMS_signal / 10^(SNR_dB/20)`, and adds independent Gaussian noise. Nodes with near-zero signal (fixed DOFs) are skipped. Returns both the noisy output and the noise array, each of shape `(101, n_steps)`. Supported SNR values: 40 dB (noise is 1/100 of signal) and 60 dB (noise is 1/1000 of signal).

---

### `sampling.py` — Experimental Design and Parameter Generation

Generates the full list of simulation parameter dictionaries that define the batch. Supports two modes: discrete (full factorial Cartesian product) and continuous (uniform random sampling from parameter ranges).

Key functions:
- `generate_parameter_sets(mode)` — top-level dispatcher, seeds all random generators with `RANDOM_SEED = 42` for full reproducibility.
- `_discrete_mode(...)` — builds all Cartesian combinations of beam geometry, material, BC type, loading type and frequency, impact position, damping pair, and SNR level. Shuffles and slices to `N_SIMULATIONS` (or takes all combinations if `None`).
- `_continuous_mode(...)` — samples each continuous parameter (geometry, force, position) independently from a uniform range and randomly picks discrete parameters (material, BC, loading type, SNR) per simulation.
- `_validate_position_for_bc(pos, bc_type)` — clamps impact positions away from constrained nodes to prevent load application at fixed DOFs.
- `_build_param_dict(...)` — constructs the complete 20+ field parameter dictionary for one simulation, covering geometry, material, loading, damping, noise, and time settings. No damage fields are present in this variant.

---

### `batch_runner.py` — Parallel Simulation Orchestrator

Runs the complete simulation pipeline for every parameter set, in parallel across all available CPU cores.

Key functions:
- `run_single_simulation(params)` — executes the full 9-step pipeline for one simulation: material lookup → matrix assembly → BC enforcement → damping → force vector → Newmark integration → sensor extraction → NaN check → noise addition. Returns a result dict with `status`, `params`, `node_accels`, and `noisy_accels`. Exceptions are caught and returned as failed results without stopping the rest of the batch.
- `run_batch(param_sets, n_jobs)` — uses `joblib.Parallel` + `tqdm` to parallelize across all cores (`n_jobs=-1`). Reports success and failure counts after completion.

---

### `exporter.py` — CSV Export

Serializes simulation results to disk in three encoding formats depending on downstream use.

Key functions:
- `_build_metadata_row(params)` — assembles a flat metadata dict with all simulation parameters (no damage fields in this variant).
- `_write_csv(row, node_accels, ...)` — implements all three encoding formats: **Encoding A** (1 row per simulation, node columns as semicolon-delimited strings), **Encoding B** (1 row per simulation, one explicit column per node-timestep pair — very wide), **Encoding C** (2000 rows per simulation with all metadata repeated on every row — the default ML-friendly format).
- `export_single_simulation(result, ...)` — exports one result according to `export_mode` (`'clean'`, `'noisy'`, or `'both'`).
- `export_all_simulations(results, ...)` — iterates all results, skips failures, and reports file count and estimated disk usage.
- `export_time_vector(output_dir, output_path)` — writes a standalone `time_vector.csv` (2000 rows, one `time_s` column) for use in ML training pipelines that need the time axis separately.

---

### `visualization.py` — Analysis and Plotting

Provides a complete post-processing visualization pipeline that loads a saved CSV and produces six types of diagnostic plots using `matplotlib`.

Key functions:
- `load_simulation_csv(file_path, delimiter)` — auto-detects CSV encoding (A vs C) based on row count and parses `params`, `node_accels (101, n_steps)`, and `time_vector`.
- `compute_fft(node_accels, dt)` and `plot_fft(...)` — computes the one-sided FFT magnitude spectrum for all 101 nodes and plots selected nodes on a log-frequency axis.
- `compute_frf(node_accels, force_time, dt)` and `plot_frf(...)` — computes the Frequency Response Function (FFT of acceleration / FFT of force) with low-force-spectrum masking to suppress noise amplification in division. Plots magnitude and phase in a two-panel figure.
- `plot_frf_per_node(...)` — per-node FRF plots with magnitude and phase side by side, limited to 600 Hz.
- `compute_mode_shapes_from_matrices(params, n_modes)` — rebuilds K and M from saved parameters and solves the generalized eigenvalue problem `K·φ = ω²·M·φ` for natural frequencies and normalized mode shapes using `scipy.linalg.eigh`.
- `find_resonant_frequencies(...)` and `extract_mode_shapes(...)` — peak picking on the FRF to identify resonant frequencies and extract operational mode shapes from FRF magnitude profiles across all nodes.
- `plot_mode_shapes(...)` — grid of mode shape subplots with filled deformation areas and boundary condition symbols (wall lines for fixed ends, triangle markers for pinned ends).
- `plot_waterfall_fft(...)` — 2D heatmap (`pcolormesh`) of FFT magnitude across all 101 nodes vs. frequency (log scale, clamped at 98th percentile).
- `plot_time_history(...)` — stacked time-domain acceleration subplots for selected nodes.
- `run_visualization(csv_file_path, output_dir, n_modes, nodes_to_plot)` — master function that runs all six visualization steps in sequence and optionally saves all plots as PNG files.

---

### `FEA_Simulations.ipynb` — Main Execution Notebook

The top-level entry point and driver for the entire pipeline. Structured as three active cells:

- **Cell 1 — Imports**: Adds the working directory to `sys.path` and imports all module components for interactive exploration.
- **Cell 2 — Batch Pipeline**: A six-step workflow controlled by `OUTPUT_DIR`, `N_JOBS`, `SAMPLING_MODE`, and `ENCODING` variables at the top:
  1. Generate parameter sets via `sampling.py` and print a preview of the first 5 simulations.
  2. Run the full batch in parallel via `batch_runner.py` with elapsed time and success/failure reporting.
  3. Export all results as CSV files (default: Encoding C, noisy only) plus a shared `time_vector.csv`.
  4. Print summary statistics covering BC types, loading types, SNR levels, impact positions, materials, geometry, and force distributions.
  5. Plot sample clean vs. noisy time histories for one simulation per loading type (impact, harmonic, random) and save as PNG files.
  6. Print a final file manifest describing all output files and the CSV column structure.
- **Cell 3 — Visualization**: Calls `run_visualization()` on a specific saved CSV to produce and save all six diagnostic plots.

---

## Pipeline Data Flow

```
config.py
    |
    v
sampling.py         --> generates list of parameter dicts (no damage fields)
    |
    v
batch_runner.py     --> parallel execution across all CPU cores
    |-- assembly.py:        assemble uniform K (202x202), M (202x202)
    |-- assembly.py:        apply BC -> K_free, M_free
    |-- damping.py:         C = alpha*M + beta*K
    |-- excitation.py:      F(t) array (n_free x 2000)
    |-- time_integrator.py: Newmark-beta -> accel (n_free x 2000)
    |-- sensors.py:         extract node accels -> (101 x 2000)
    |-- noise.py:           add SNR-calibrated Gaussian noise -> (101 x 2000)
    |
    v
exporter.py         --> sim_XXXX_noisy.csv  (Encoding C: 2000 rows x ~120 cols)
    |
    v
visualization.py    --> FFT, FRF, mode shapes, waterfall, time history plots
```
