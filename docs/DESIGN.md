# Design

Architecture, data flow and extension points. This file grows with each phase; see PLAN.md for
what is built so far. SPEC.md §4–§6 is the normative description.

## Architecture (target)

```
configs/<exp>.yaml ──load_config(profile)──▶ typed dataclass config
                                                   │
superposition run <exp> ──▶ experiments/<exp>.run(config, run_dir, device)
                                   │
            ┌──────────────────────┼──────────────────────────┐
            ▼                      ▼                          ▼
      data.FeatureDistribution  models.TiedModel/AbsModel   train.train(...)
            └──────────────────────┴──────────▶ TrainResult (loss history, snapshots, final loss)
                                                   │
                                   metrics (pure functions of W, b)
                                                   │
                         runs.RunDir: config.yaml, manifest.json, metrics.parquet,
                                      acceptance.json, checkpoints, figures, log.txt
                                                   │
                  plotting/ ◀── superposition figures ──▶ docs/figures/<exp>/ (--publish)
                  report    ◀── superposition report  ──▶ docs/REPRODUCTION.md tables
                  app/      ◀── runs.latest_run(...)   (Streamlit lab, Phase 2+)
```

Details of each layer are written as the layer lands (Phase 1 onward).
