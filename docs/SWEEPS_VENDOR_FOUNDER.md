# Vendor vs Founder Sweep

This sweep runs `analyze` + `anchor` + `classify` on a fixed version and reports how many chunks are labeled vendor vs founder/family. It is intended to help tune thresholds so classification does not drift too liberal or too strict.

## What It Measures

- Vendor count: chunks labeled `vendor`
- Founder count: `founder + family` (family is treated as founder-like in `src/classify_logic.js`)
- Unknown count: chunks that do not receive a category

## Configuration

Edit a sweep config:

- `scripts/sweeps/vendor_founder_2.1.19.json`
- `scripts/sweeps/hone_2.1.19.json`

Each run lists env overrides that map directly to variables in `.env.example`, for example:

- `LIBRARY_MATCH_THRESHOLD`
- `ANCHOR_SIMILARITY_THRESHOLD`
- `SPREADING_THRESHOLD_RATIO`
- `SPREADING_THRESHOLD_COUNT`
- `VENDOR_SPREADING_THRESHOLD_RATIO`
- `VENDOR_SPREADING_THRESHOLD_COUNT`
- `VENDOR_RANGE_WINDOW_SIZE`
- `VENDOR_RANGE_VENDOR_RATIO`
- `VENDOR_RANGE_MIN_VENDOR`
- `VENDOR_BRIDGE_GAP`
- `MARKOV_DAMPING_FACTOR`
- `CHUNKING_TOKEN_THRESHOLD`

You can add additional runs or adjust the version. If you add variables, ensure they already exist in `.env.example`.
The sweep runner accepts multiple configs via `--config`, so you can create new files without changing the script.

Vendor range detection adds a contiguous window check to promote adjacent chunks to vendor when vendor chunks cluster together (useful when vendor libraries appear in runs). Tune via `VENDOR_RANGE_WINDOW_SIZE`, `VENDOR_RANGE_VENDOR_RATIO`, and `VENDOR_RANGE_MIN_VENDOR`.

## Run

```bash
npm run sweep-vendor-founder
```

Or run directly with a custom config:

```bash
node scripts/vendor_founder_sweep.js --config scripts/sweeps/vendor_founder_2.1.19.json
```

To run the honed sweep:

```bash
node scripts/vendor_founder_sweep.js --config scripts/sweeps/hone_2.1.19.json
```

## Output

Reports are written to:

- `cascade_graph_analysis/sweeps/vendor_founder_2.1.19/report.md`
- `cascade_graph_analysis/sweeps/vendor_founder_2.1.19/report.json`

The honed sweep writes to:

- `cascade_graph_analysis/sweeps/honing_2.1.19/report.md`
- `cascade_graph_analysis/sweeps/honing_2.1.19/report.json`

Each sweep run also archives the full output for inspection under:

- `cascade_graph_analysis/sweeps/vendor_founder_2.1.19/<run-name>/`
- `cascade_graph_analysis/sweeps/honing_2.1.19/<run-name>/`

## Notes

- Each run executes analyze and anchor from scratch, so the sweep can take time.
- The analyzer downloads the target bundle if it is not already cached in `claude-analysis/`.
- Vendor spreading and orphan sensitivity runs use `VENDOR_SPREADING_THRESHOLD_RATIO`, `VENDOR_SPREADING_THRESHOLD_COUNT`, and `IMPORTANT_ORPHAN_CENTRALITY` when present in the sweep config.
