---
name: "python-coding-style"
description: "Python coding conventions and philosophy for scientific software development. Use this skill whenever the user asks Claude to write, review, edit, debug, or discuss Python code — including scripts, notebooks, data pipelines, model training, plotting, analysis, or any task involving Python. Also trigger when the user uploads .py files, Jupyter notebooks, or asks about Python libraries. This skill applies broadly to all Python work, not just specific file types."
---

# Python Coding Style

Conventions for scientific Python software (computer vision, bioinformatics) built for expert users. Goal: **code that is correct, minimal, and fails loudly when something is wrong.** These rules favour caution over speed; for trivial tasks, use judgment.

Core libraries: numpy, scipy, pandas, matplotlib, scikit-image, scikit-learn, PyTorch, PyTorch Lightning, MONAI, zarr, dask, SpatialData, AnnData, scanpy, FAISS. Prefer their idioms.

## 1. Think before coding

- State your assumptions. If there are several interpretations, present them; don't pick silently.
- If something is unclear, stop, name what is confusing, and ask.
- Never make analytical choices silently: method, model architecture, normalization, distance metric, handling of missing annotations, filtering thresholds, pipeline structure. Ask, or propose one with its source (user, paper, documented default, or the data). Aesthetic choices (palettes, figure sizes) you may decide.
- If a simpler approach exists, say so. Push back when warranted.

## 2. Simplicity first

Minimum code that solves the problem. If 200 lines could be 50, rewrite it.

- No features, flexibility, or configurability beyond what was asked.
- Functional by default; classes only for genuine state (PyTorch/Lightning modules, dataloaders).
- No abstractions for single-use code, and no trivial wrappers: if a body is shorter than its signature plus docstring, inline it.
- Reuse before writing: search the user's own code first (current repo, their other packages), then their libraries. If something almost fits, propose extending it rather than writing a near-copy that will drift.

```python
# WRONG — this function adds nothing
def load_adata(path: str) -> ad.AnnData:
    """Load an AnnData object from h5ad."""
    return ad.read_h5ad(path)

# RIGHT
adata = ad.read_h5ad(path)
```

```python
# WRONG — the project already has this
def normalize_channel(image, percentile=1):
    lo, hi = np.percentile(image, [percentile, 100 - percentile])
    return np.clip((image - lo) / (hi - lo), 0, 1)

# RIGHT — reuse the project's own implementation
from myproject.plotting import scale_image
scaled = scale_image(image, percentile=1)
```

```python
# WRONG — scanpy already does this
def plot_umap(adata, color_by):
    coords = adata.obsm["X_umap"]
    fig, ax = plt.subplots()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=adata.obs[color_by], s=1)
    plt.colorbar(scatter)
    return fig

# RIGHT
sc.pl.umap(adata, color=color_by)
```

## 3. Surgical changes

Every changed line should trace directly to the user's request.

- Don't "improve" adjacent code, comments, or formatting, and don't refactor what isn't broken.
- Match the existing style and the repo's tooling (pixi or uv, formatter and linter config, CLI framework).
- Remove imports, variables, and functions that your change made unused. Leave pre-existing dead code alone, but mention it.

## 4. Fail loudly

**Let it crash.** Never use `try`/`except` to hide or recover from an error, and never write defensive checks that skip data, return `None`, or substitute defaults. A traceback beats a subtle wrong result. Two uses are allowed because they still crash: re-raising with added context (`raise ValueError(f"bad tile {tile_id}") from err`), and cleanup through `finally` or a context manager.

```python
# WRONG — silently produces incomplete results
for sample in samples:
    if sample.get("image") is not None:
        results.append(process(sample["image"]))

# RIGHT — crashes immediately if data is malformed
for sample in samples:
    results.append(process(sample["image"]))
```

```python
# WRONG — swallows the real error
try:
    adata = sc.read_h5ad(path)
except Exception:
    adata = None

# RIGHT — if the file can't be read, we need to know why
adata = sc.read_h5ad(path)
```

**Assert where failures are silent.** numpy broadcasts `(N,)` against `(N, 1)` into `(N, N)`, casting negative floats to uint16 is undefined and often wraps (-5 becomes 65531), uint16 subtraction wraps (100 - 200 gives 65436), and compiled libraries (ITK, itk-elastix, OpenCV) cast without warning. Add an assert or explicit conversion at least at dtype downcasts, unsigned arithmetic, compiled-library outputs, pandas merges and index alignment, and tensor shapes entering a model. Checks that raise are welcome anywhere they catch a wrong type, dtype, shape, or overwrite (`assert image.dtype == np.uint16`); what is banned is catching errors, not raising them.

```python
# WRONG — negative values from background subtraction wrap to ~65535
corrected = (image - background).astype(np.uint16)

# RIGHT — compute in a signed dtype, then make the clipping explicit
corrected = image.astype(np.float32) - background.astype(np.float32)
corrected = np.clip(corrected, 0, np.iinfo(np.uint16).max).astype(np.uint16)
```

```python
# WRONG — row counts silently change if keys are missing or duplicated
features = cells.merge(metadata, on="well_id")

# RIGHT — validate crashes on duplicate keys; the assert catches wells missing from metadata
features = cells.merge(metadata, on="well_id", how="left", validate="many_to_one")
assert features["donor_age"].notna().all()
```

**If something is broken, say so.** A failing tool, a bug in the user's code, a misbehaving library, a misconfigured environment, data that looks wrong: stop and report it with the evidence. Never work around it silently (switching tools or methods, patching around another module, dropping failing samples). Report bugs noticed outside the task too.

## 5. Data and reproducibility

- **One source of truth.** Channel lists, sample IDs, thresholds, and paths live in one config or metadata file. Fix wrong values at the source, never with downstream aliases or copies.
- **One data object.** Annotations, metadata, QC flags, labels, and embeddings go inside the AnnData or SpatialData object they describe, never in side CSVs or extra objects. Tell the user before adding anything ("adding `obs['leiden_r0.5']` to the fetal table"); never overwrite an existing key or pile up near-duplicate annotations.
- **Read lazily.** Open zarr, SpatialData, and dask stores lazily and slice; never load a whole store.
- Paths via `pathlib` from config or CLI arguments, never hard-coded.
- Seeds as named parameters: `np.random.default_rng(seed)`, `L.seed_everything(seed)`.
- Pipelines save their YAML config next to their outputs. Notebooks live in a pixi or uv project so each commit captures the environment, and are always run top to bottom from a fresh kernel.

```python
# WRONG — the channel list lives in two places; a metadata typo is aliased downstream
CHANNELS = ["DAPI", "SOX2", "PAX6", "BCL11B"]
stain_names = {"CTIP2": "BCL11B"}

# RIGHT — read it from the single source, fix typos there
channels = pd.read_csv(stain_metadata_path)["stain"].tolist()
```

```python
# WRONG — the dataset is now spread over one object and several side files
adata.write_h5ad("cells.h5ad")
qc.to_csv("cells_qc.csv")
clusters.to_csv("cells_leiden.csv")

# RIGHT — everything about these cells lives in the one object
assert "qc_pass" not in adata.obs and "leiden" not in adata.obs
adata.obs["qc_pass"] = qc.loc[adata.obs_names, "pass"]  # .loc crashes on missing cells instead of NaN-filling
adata.obs["leiden"] = clusters.loc[adata.obs_names]
adata.write_h5ad("cells.h5ad")
```

```python
rng = np.random.default_rng(cfg["seed"])
subset = rng.choice(n_cells, size=cfg["n_subsample"], replace=False)
shutil.copy(config_path, out_dir / "config.yaml")
```

## 6. No magic thresholds

Any number that decides which data are kept, how they are labelled, or what a result says is an analytical decision the user must be able to defend in a paper; a buried, hand-tuned one can amount to academic dishonesty. Never invent one silently: ask, or propose one with its source (user, paper, documented default, or the data). Name it, justify it next to its definition, show the distribution with the cutoff when it is data-derived, and list every threshold you introduced or changed in your reply. Display settings are not thresholds, but must match across compared panels.

```python
# WRONG — where does 0.35 come from? Nobody reviewing this will know it exists
cells = cells[cells["solidity"] > 0.35]

# RIGHT — derived from data, named, justified, visible
MIN_SOLIDITY = np.percentile(curated_nuclei["solidity"], 1)  # 1st percentile of manually curated nuclei, see qc/solidity_hist.png
cells = cells[cells["solidity"] > MIN_SOLIDITY]
```

## 7. Goal-driven execution

Turn each task into a verifiable goal, state a brief plan, and loop until each check passes:

```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
```

- "Fix the bug" -> reproduce it on one sample, write a test that fails because of it, fix, show the test passes and the sample looks right.
- "Add a filter" -> show the distribution with the cutoff, report how much was kept.
- "Refactor X" -> existing tests pass and outputs are identical on one sample, before and after.

**Tracer bullets.** Get one sample running end to end before broadening. While developing, show a quick-look plot after each stage and print how much it kept (`kept 41,203 / 45,880 cells`); finished pipeline code keeps the counts in its log but leaves the plots out unless QC output is asked for, and the visual checks run alongside it instead (see "Check on real data" below). Images: same crop before and after with identical contrast, plus a histogram with the dtype range. Tables: histograms, violins, dotplots. Embeddings: UMAP, t-SNE, or PCA coloured by metadata. Graphs: a drawing of the structure.

```python
registered = register(moving, fixed)

# Quick look: same crop, same contrast, and the value range against the input dtype
crop = np.s_[1000:1512, 1000:1512]
lo, hi = np.percentile(moving[crop], [1, 99.8])
dtype_range = np.iinfo(moving.dtype)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(moving[crop], vmin=lo, vmax=hi)
axes[1].imshow(registered[crop], vmin=lo, vmax=hi)
axes[2].hist(registered.ravel(), bins=256, log=True)
axes[2].axvline(dtype_range.min, color="red")
axes[2].axvline(dtype_range.max, color="red")
```

**Don't program by coincidence.** Know why it works, not just that it ran. Suspect our own code before numpy, torch, or scanpy. Fix root causes; never tweak parameters until the output looks right.

**Tests.** Write tests for the logic you add or change. Use pytest on small synthetic inputs where the right answer is known by construction: an image shifted by a known offset registers back by that offset, a hand-built table merges to the expected rows, values at the dtype edges (0, max, negative before clipping) come out right. Cover the silent-failure spots from section 4 first. Keep tests short, test our code rather than numpy or scanpy, and don't mock the scientific libraries. Run existing tests when you change code they cover; if one fails, report it and never edit a test just to make it pass. If the repo has no test setup yet, ask before adding one.

**Check on real data.** Tests prove the logic; only real data shows whether the result makes sense. After implementing, run the code on one real sample and show the user the quick-look plots from above together with the key numbers (counts kept, value ranges, group sizes). Say what you expect the plots to show and point out anything that looks off, so the user can judge it at a glance. Never call a task done on passing tests alone.

```python
# Synthetic test: the answer is known by construction
def test_background_subtraction_clips_instead_of_wrapping():
    image = np.array([[100, 50]], dtype=np.uint16)
    background = np.array([[20, 80]], dtype=np.uint16)
    corrected = subtract_background(image, background)
    assert corrected.dtype == np.uint16
    np.testing.assert_array_equal(corrected, [[80, 0]])  # 50 - 80 must clip to 0, not wrap to 65506
```

## 8. Style

- Type-hint all signatures. Scanpy-style docstrings: numpy layout, no types in the docstring, no colon after the parameter name.
- Imports at the top, grouped: stdlib, third-party, local.
- No emojis in code, comments, docstrings, or explanations, unless the user uses them.

```python
def filter_cells(
    adata: ad.AnnData,
    min_counts: int,
    max_mito_fraction: float,
) -> ad.AnnData:
    """Filter cells by total counts and mitochondrial fraction.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_counts
        Minimum total counts per cell.
    max_mito_fraction
        Maximum fraction of mitochondrial counts.

    Returns
    -------
    Filtered annotated data matrix.
    """
```