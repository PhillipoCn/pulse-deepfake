# Pulse-based deepfake detection

Built from "Deepfake detection from your pulse: how it works and how to build it" (Hari).
Reads the heart rate from ordinary video (remote photoplethysmography, rPPG), then checks
whether the pulse in different skin patches agrees, the way FakeCatcher does.

## Setup (done once)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models && curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python -m pytest                   # 14 synthetic tests: a known pulse in, the same pulse out
```

## Run order

| Step | Command | What you should see |
|---|---|---|
| 1-2 Face + patches | `python check_face.py VIDEO` | `outputs/*_regions.png`: 5 patches on bare skin |
| Record yourself | `python record.py --seconds 30` | lossless clip in `data/mine/`; count your pulse meanwhile |
| 3 Heart rate | `python heart_rate.py VIDEO --true-bpm 68` | BPM for green / CHROM / POS, plot with a spike near your pulse |
| 4 Check vs real pulse | `python eval_ubfc.py data/ubfc` | MAE table + `outputs/ubfc_scatter.png` |
| 5 Features | `python build_ffpp.py data/ffpp --methods Deepfakes FaceSwap --limit 200` | `outputs/ffpp_features.npz` |
| 6-7 Train + test | `python train_detector.py` | accuracy / AUC on people never seen; `models/detector.joblib` (`--level window` = original v1) |
| 7b Generalisation | `python train_detector.py --holdout-method FaceSwap` | trained without FaceSwap, tested on it |
| 7d Raw vs c23 | `python compare_raw.py` | does compression hide the pulse? (needs raw subset) |
| 7c CNN (best) | `python train_cnn.py` (+ `--holdout-method X`) | PPG-map CNN, `models/ppg_cnn.pt`, used by `detect.py` |
| 8 Use it | `python detect.py VIDEO` | BPM, window votes, REAL/FAKE + caveats |

## Movement-robust heart rate (pulse/robust.py)

`heart_rate.py` prints two answers. The simple one averages the whole clip. The robust one:
skips 10 s windows where the head turns or moves fast, regresses head motion out of the
colour traces, lets the patches vote with their spectra, then tracks the most continuous
heart-rate path (Viterbi). It reports a strength vs noise:
**>= 3x reliable, 2.3-3x weak guess, < 2.3x no pulse** (pure noise reaches ~2.2x).
Patch means also drop the darkest 15% / brightest 10% of pixels (glasses, beard, shine).

### Measured accuracy (UBFC-rPPG DATASET_2, 30 subjects, finger-sensor PPG as truth)

| Method | 10 s windows MAE | 30 s windows MAE |
|---|---|---|
| green | 6.5 BPM | 6.1 BPM |
| CHROM | 1.2 BPM | 1.2 BPM |
| POS | 1.2 BPM | 1.0 BPM |
| robust tracker | 1.4 BPM | - |

`heart_rate.py` prints one BEST ANSWER: strength >= 4x -> POS; 2.3-4x -> tracker (a guess);
< 2.3x -> no pulse. Every UBFC subject scored >= 3.9x; the iPhone home clips 2.2-3.6x.
The motion filter skips windows where the head turns (yaw changes > 0.1), moves fast,
or is turned > 0.3 - a steady slight turn is fine (UBFC subjects 17 and 18 sit like that).

### Recording tips (these matter more than any code)
Sit still facing the camera, light from the front (window or lamp), nothing bright behind,
lock exposure (iPhone: press and hold the face until AE/AF LOCK), face fills the frame,
glasses off if possible, 30 s+, and read a real pulse during the clip.

### Measured detector accuracy (FaceForensics++ c23, people never seen in training)

5-fold GroupKFold on identity groups, video level.

**2 fake methods** (Deepfakes, FaceSwap; 2,912 videos):

| Detector | Balanced acc | AUC |
|---|---|---|
| v1: 14 window features, logreg, window votes (`--level window`) | 0.71 | 0.78 |
| v2: + per-patch/per-colour detail (`pulse/detail.py`), boosted trees | 0.85 | 0.93 |

**All 5 fake methods** (+ Face2Face, NeuralTextures, FaceShifter; 5,825 videos):

| Detector | Balanced acc | AUC | real correct |
|---|---|---|---|
| v2 features + boosted trees (balanced cut-off 0.73) | 0.77 | 0.85 | 0.80 |
| **v3: 1D CNN on PPG maps (`train_cnn.py`, `pulse/ppgmap.py`)** | **0.85** | **0.93** | 0.79 |

**Fake method never seen in training** (trained on the other 4), AUC:

| Held out | v2 (2 methods) | v2 (4 methods) | **v3 CNN (4 methods)** |
|---|---|---|---|
| Deepfakes | 0.75 | 0.83 | **0.87** |
| FaceSwap | 0.69 | 0.76 | **0.86** |
| Face2Face | - | 0.78 | **0.88** |
| NeuralTextures | - | 0.65 | **0.75** |
| FaceShifter | - | 0.84 | **0.88** |

What the numbers say: the pulse-agreement features alone reach only AUC 0.55 at c23; most of
the signal is flicker and spectral shape above the heart-rate band (fakes re-render the face
every frame). Letting a CNN read the full PPG map beats hand-made features, and training on
more fake methods is what helps most on unseen ones. NeuralTextures (mouth-only edits) is
hardest. The feature model reacts to camera/compression: on home iPhone/webcam clips it gave
P(fake) up to 0.88, while the CNN stayed at 0.06-0.34, so `detect.py` uses the CNN.
175 of 6,000 videos were skipped (face in < 80% of frames).

### Does compression hide the pulse? (`compare_raw.py`, same 572 videos raw vs c23)

(The raw subset was deleted after this run to free 53 GB; re-download it with `-c raw -n 100` to rerun.)

| Test | raw | c23 |
|---|---|---|
| Detection from the 13 pulse-agreement features only (5-fold, unseen people) | **AUC 0.76** | 0.62 |
| CNN trained on c23 without these people, tested on them | **AUC 0.94** (real 0.92) | 0.92 (real 0.85) |
| Features + boost, same setup | AUC 0.92 | 0.85 |

The pulse-agreement idea works much better without compression. The pulse in *real* videos is
no clearer at raw (median SNR -3.3 vs -3.1 dB), because FF++ originals were already YouTube
video, so the gain comes from the fakes' pulse defects surviving. The c23-trained CNN holds
up under a change of compression, except on NeuralTextures (0.72 at c23 -> 0.45 at raw).
In this script the boost model's cut-off is picked on its own training data, which is why
it lets only ~47% of real videos through; its AUC is the fair number.

## Credits

- Method and project idea: "Deepfake detection from your pulse: how it works and how to build it" by Hari.
- rPPG: de Haan & Jeanne 2013 (CHROM), Wang et al. 2017 (POS).
- FakeCatcher: Ciftci, Demir & Yin 2020. Face landmarks: Google MediaPipe Face Landmarker.
- Datasets (not included, request access from their authors): UBFC-rPPG (Bobbia et al. 2019),
  FaceForensics++ (Rössler et al. 2019). Trained models are not included either; train your own.

## Datasets

- **UBFC-rPPG**: unzip so subject folders sit under `data/ubfc/` (e.g.
  `data/ubfc/DATASET_2/subject1/vid.avi` + `ground_truth.txt`). DATASET_1 (`gtdump.xmp`) also works.
- **FaceForensics++**: after the form, use their `download-FaceForensics.py`:
  ```bash
  python download-FaceForensics.py data/ffpp -d original  -c c23 -t videos
  python download-FaceForensics.py data/ffpp -d Deepfakes -c c23 -t videos
  python download-FaceForensics.py data/ffpp -d FaceSwap  -c c23 -t videos
  ```
  Start with `--limit 100` in `build_ffpp.py` to check timings before running everything.

## Layout

```
pulse/video.py     OpenCV frames, real fps, MediaPipe 478 landmarks
pulse/roi.py       skin patches (forehead x2, cheeks x2, chin) and their mean RGB
pulse/extract.py   video -> (frames x patches x RGB), glitch removal, cache
pulse/signals.py   detrend, band-pass, green / CHROM / POS, FFT -> BPM, SNR
pulse/features.py  per-window patch agreement: coherence, correlation, lag, BPM spread, SNR, 4-14 Hz power
pulse/ppgmap.py    PPG map (35 rows x time) + 1D CNN
pulse/detail.py    per-video detail: band power + flicker per patch and colour, pose jitter
pulse/robust.py    movement-robust heart rate (motion gate, regression, spectral vote, Viterbi)
```

## Honest expectations

- A homemade detector lands well below FakeCatcher's 96% (and that number was one older
  test set; ~91% on harder video). Above chance on unseen people is a good result.
- Compression (c23/c40), movement and uneven light all weaken the pulse.
- 10 s clips resolve BPM only to ~6 BPM; 30 s to ~2.
- Good face swaps can copy the source pulse (2025): a pulse is not proof a video is real.
- rPPG accuracy drops on the darkest skin types; check your error by subgroup if you can.
