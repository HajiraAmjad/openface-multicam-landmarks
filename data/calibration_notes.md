# Calibration & Sync Analysis Notes

## Camera FPS Summary (Faith_cal session)

| Camera | FPS | Frames | Interval Std | Status |
|---|---|---|---|---|
| Front | 120.09 | 2824 | 0.000011s | ✅ Very stable |
| Left | 119.75 | 2814 | 0.000887s | ⚠️ Slight jitter |
| Right | 240.12 | 5630 | 0.000006s | ✅ Very stable |

## Session Sync Quality

| Session | Front fps | Left fps | Right fps | Left jitter | Front jitter | Notes |
|---|---|---|---|---|---|---|
| Faith_cal | 120.09 | 119.75 | 240.12 | ⚠️ 0.887ms | ✅ 0.011ms | Calibration session |
| Faith1 | 120.09 | 120.09 | 240.12 | ✅ 0.011ms | ✅ 0.011ms | Best sync |
| Faith2 | 119.67 | 119.67 | 240.12 | ⚠️ 0.993ms | ⚠️ 0.993ms | Possible thermal throttling |
| Faith3 | 120.09 | 119.67 | 240.12 | ⚠️ 0.988ms | ✅ 0.011ms | Left jitter only |
| Faith4 | 120.09 | 120.09 | 240.12 | ✅ 0.011ms | ✅ 0.011ms | Best sync |

**Best sessions: Faith1 and Faith4** — all cameras stable at 0.011ms jitter.

---

## ESP32 Accelerometer Sync

| Session | Sampling rate | Duration | Issue |
|---|---|---|---|
| Faith1 | — | ~11s | ❌ Epoch mismatch (timestamps start from 1970 — corrupted clock) |
| Faith2 | 110 Hz | 121s | ⚠️ Unusually slow, session much longer than others |
| Faith3 | ~246 Hz | ~11s | ✅ Normal |
| Faith4 | ~246 Hz | ~11s | ✅ Normal |

Faith_cal accelerometer: 6180 samples at 248.25 Hz over 24.9s (09:04:49 → 09:05:13).

---

## Why Only 22 Common Calibration Frames

The calibration used 22 synchronized frames where the ChArUco board was visible in all 3 cameras simultaneously. The bottleneck was the Left camera's 0.887ms jitter during `Faith_cal` — the timestamp alignment step uses strict nearest-timestamp matching, and jitter causes frames to fall outside the sync window.

An additional factor: Right camera runs at 240 fps (2× Front and Left at 120 fps), making timestamp alignment stricter since Right frames are denser and harder to match exactly.

### Result despite only 22 frames
- Reprojection error: **0.086 px** (excellent — threshold < 1.0 px)
- Multi-camera error: **1.12 px**

The calibration quality is excellent and sufficient for landmark extraction.

---

## Should Calibration Be Re-Run?

**Current recommendation: No.**

- 0.086 px reprojection error is well within acceptable range regardless of frame count.
- The Left camera jitter in `Faith_cal` was a one-session issue — Faith1 and Faith4 confirm Left is capable of 0.011ms stability.
- Re-running is unlikely to improve an already sub-pixel result.

**Re-run only if** stereo 3D reconstruction or triangulation across cameras is needed, where geometric accuracy from more common frames would matter.

### If re-running, two options:
1. `--sync-method integer_offset` instead of `nearest_timestamp` — more tolerant alignment, likely to pass more frames through the sync filter.
2. Re-record calibration session with Left camera confirmed stable at 120 fps before starting.

---

## Camera Physical Setup

| Pair | Distance | Angle |
|---|---|---|
| Front ↔ Left | 27.1 cm | 18.1° |
| Front ↔ Right | 45.1 cm | 34.2° |
| Left ↔ Right | 64.0 cm | 52.3° |

**Positions relative to Front:**
- Left: 3 cm left, 23 cm below, 14 cm back
- Right: 3.5 cm right, 40 cm above, 20 cm forward

Full calibration output: [`calibration_results.json`](../calibration_results.json)
