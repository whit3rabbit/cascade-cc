# Vendor vs Founder Sweep

This sweep runs `analyze` + `anchor` + `classify` on a fixed version and reports how many chunks are labeled vendor vs founder/family. It is intended to help tune thresholds so classification does not drift too liberal or too strict.

## What It Measures

- Vendor count: chunks labeled `vendor`
- Founder count: `founder + family` (family is treated as founder-like in `src/classify_logic.js`)
- Unknown count: chunks that do not receive a category

## Configuration

Edit the sweep config:

- `scripts/sweeps/vendor_founder_2.1.19.json`

Each run lists env overrides that map directly to variables in `.env.example`, for example:

- `LIBRARY_MATCH_THRESHOLD`
- `ANCHOR_SIMILARITY_THRESHOLD`
- `SPREADING_THRESHOLD_RATIO`
- `SPREADING_THRESHOLD_COUNT`
- `MARKOV_DAMPING_FACTOR`
- `CHUNKING_TOKEN_THRESHOLD`

You can add additional runs or adjust the version. If you add variables, ensure they already exist in `.env.example`.

## Run

```bash
npm run sweep-vendor-founder
```

Or run directly with a custom config:

```bash
node scripts/vendor_founder_sweep.js --config scripts/sweeps/vendor_founder_2.1.19.json
```

## Output

Reports are written to:

- `cascade_graph_analysis/sweeps/vendor_founder_2.1.19/report.md`
- `cascade_graph_analysis/sweeps/vendor_founder_2.1.19/report.json`

Each sweep run also archives the full output for inspection under:

- `cascade_graph_analysis/sweeps/vendor_founder_2.1.19/<run-name>/`

## Notes

- Each run executes analyze and anchor from scratch, so the sweep can take time.
- The analyzer downloads the target bundle if it is not already cached in `claude-analysis/`.
