# v2 — Obstacle-Aware Diffusion Trajectory Refiner

Everything in this directory is additive. `planner_node.py` (v1) is not
modified. `v2/` produces `planner_node_v2.py`, run as a separate entry
point, so the old node stays as a known-good fallback the whole time.

## Why v2 exists

Two separate problems, addressed by two separate mechanisms:

1. **No obstacle awareness.** The trained checkpoint
   (`model_v2_epoch_50_sincos.pt`) conditions only on 5-dim pose+steering.
   It cannot see the occupancy map. Fix: EDT-based gradient guidance at
   inference time (Week 2), zero retraining.

2. **Zigzag / non-smooth trajectories.** This is a *sampling* problem, not
   a guidance problem. DDPM is stochastic — every denoising step draws
   fresh noise, and every replanning cycle starts independently with no
   coupling to the previous cycle's solution. So even near-identical robot
   states 100ms apart can produce visibly different trajectories, which
   shows up as left-right oscillation in closed loop. The `alpha=0.8`
   temporal smoothing patch (already applied in v1) blends the symptom
   away after the fact — it doesn't fix the cause.
   Fix: DDIM (deterministic ODE sampler, Week 1), later reinforced by
   warm-starting from the previous cycle's trajectory instead of fresh
   noise (Week 3, only if still needed after DDIM).

## Directory layout

```
v2/
  profiling.py          # Gate 1 — done. Timing instrumentation, stdlib only.
  ddim_scheduler.py      # Gate 2 — next. Deterministic sampler wrapper.
  collision_cost.py      # Gate 3 — EDT field from OccupancyGrid.
  guidance.py             # Gate 4 — gradient guidance term in the DDIM loop.
  fallback.py              # Gate 5/6 — footprint check + A* fallback + status reasons.
  planner_node_v2.py        # Entry point wiring all of the above together.
```

## Gate checklist

- [ ] **Gate 1 — Profiling.** `CycleProfiler` wired into every stage of the
      loop. Know exactly where the 720ms goes before changing anything.
- [ ] **Gate 2 — DDIM + FP16.** 20-step deterministic sampling, trajectory
      geometry visually comparable to the 100-step DDPM baseline, latency
      down, and — this is the one to actually check — the zigzag should be
      visibly reduced in RViz even before any guidance is added.
- [ ] **Gate 3 — EDT field.** `distance_transform_edt` output overlaid on
      RViz matches physical obstacles, frame_id matches the RViz fixed
      frame (this bit you already lost a day to once — don't repeat it),
      coordinate/resolution alignment confirmed.
- [ ] **Gate 4 — Guidance.** Single obstacle placed on the path → trajectory
      bends into free space, no A* anchor term yet, guidance-only.
- [ ] **Gate 5 — Stability.** 30-minute continuous CoppeliaSim run, stable
      publish frequency, no leaks, no dropped cycles.
- [ ] **Gate 6 — Delivery.** A* vs diffusion-refiner screencast across
      Vadim's narrowed test scenarios, metrics packaged for Dmitry.

## Explicitly out of scope

No retraining, no new dataset, no CNN/transformer map encoder, no
action-space diffusion, no A*-anchor loss until Gate 4 passes on
guidance alone. If a step isn't in this list, it isn't happening this
internship.
