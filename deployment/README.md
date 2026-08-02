# Deployment bundle

`deployment/current/` is the immutable, checksummed input used by
`streamlit_app.py`. It contains only the active player snapshot, predictions,
reference populations, FPL rules, and selected model artifacts. It does not
contain training data, manager data, or uploaded screenshots.

Rebuild it locally after a validated model or prediction refresh:

```bash
python scripts/build_serving_bundle.py \
  --players data/processed/20260730T162356213385Z/players.parquet \
  --bootstrap data/raw/20260730T162356213385Z/bootstrap-static.json \
  --predictions outputs/predictions/players-20260802-lineup-reconciled.parquet \
  --prediction-metadata outputs/predictions/players-20260802-lineup-reconciled.json \
  --points-models models/20260802T150839931234Z \
  --minutes-models models/20260802T152221018736Z \
  --appearance-models models/20260802T154849873784Z \
  --start-models models/20260802T155252021597Z \
  --reference-size 1000 \
  --output deployment/current
```

The builder refuses to overwrite an existing directory. Build into a new
versioned directory, verify it, and only then replace `deployment/current` in a
normal reviewed Git change.
