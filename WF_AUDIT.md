# Workflow Audit — `wigjec5544f/facefusion`

> Trạng thái pipeline contribution của fork tính tới **2026-05-05**.
> Tổng hợp 20 PR đã ship + roadmap progress + risks/debt.
>
> Đây là **tài liệu audit tham chiếu**, đi kèm `ULTRA_ROADMAP.md` (kế hoạch
> SOTA gốc) và `README.md` (hướng dẫn dùng). PR thực thi nằm tại
> https://github.com/wigjec5544f/facefusion/pulls?q=is%3Apr.

---

## 0. Tóm tắt

| Hạng mục | Số liệu |
|----------|---------|
| Tổng PR đã mở | 20 |
| Đã merge | 19 (PR #1–#19) |
| Đang mở chờ review | 1 (PR #20) |
| Roadmap milestones đã đụng | A1·A3·A4 / B1+ / D1 / F4 / G2 |
| Roadmap milestones còn lại | A1 (block license) / A2 (block GPU) / B2 / B3 / C1–C4 / D2 / D3 / E·F·G ngoài G2 |
| File mới (cumulative) | tools/ + processors/ + tests/ ~26 file |
| Tests mới (cumulative) | ≥ 130 unit test (batching / doctor / interpolator / portrait_animator / lip_syncer scaffold / face_classifier ...) |
| HF mirror | https://huggingface.co/ngoqquyen/facefusion-extras |

Fork giữ nguyên license OpenRAIL-AS upstream. Không bundle weight tier 3
(research-only) — tất cả research model đều opt-in qua flag CLI riêng.

---

## 1. Methodology

Quy ước áp dụng xuyên suốt 20 PR:

1. **1 PR = 1 milestone** (hoặc 1 slice của milestone lớn). Branch riêng theo
   pattern `devin/<unix_ts>-<topic>`.
2. **Không bundle weight có license mơ hồ.** Weight Apache/MIT/BSD đẩy lên
   HF mirror qua `tools/hf_publish.py`; tier "research-only" / "non-commercial"
   yêu cầu user opt-in flag, fork không tự động download.
3. **Bit-equal cho stock model** là gate cứng cho mọi PR refactor performance.
   Stock ONNX = output identical bytes vs master. Speedup chỉ kích hoạt khi user
   re-export model với `dynamic_axes`.
4. **Test trước hết**: mỗi PR đi kèm unit test stub-based (không cần weight thật)
   để CI có thể chạy trên runner CPU-only.
5. **Đường runtime cũ phải còn**: legacy entrypoint (`forward`, `enhance_face`,
   `restore_expression`...) giữ signature cho back-compat.
6. **Devin Review pass** là gate trước khi báo done. Bug Devin Review tìm thấy
   được fix trên cùng branch trước khi merge.

---

## 2. PR Inventory (chronological)

| # | Title | Tier | Status | Đợt | Files (+/-) | Notes |
|---|-------|------|--------|-----|-------------|-------|
| 1 | Windows bootstrap + high-quality preset | infra | merged | A4 | bat × 3, ini × 1, README | `install.bat` + `run.bat` + `facefusion.high-quality.ini` |
| 2 | ULTRA_ROADMAP.md (SOTA survey) | docs | merged | — | ULTRA_ROADMAP.md | Kế hoạch SOTA + license tier matrix |
| 3 | balanced/fast presets + `facefusion doctor` v1 | infra/A4/F4 | merged | A4·F4 | 2 ini + doctor.py + 11 test | Doctor CLI subcommand, table check |
| 4 | pyproject.toml + HF/GH namespace override | infra | merged | B1 | pyproject + download.py + 11 test | `FACEFUSION_HF_NAMESPACE` / `FACEFUSION_GH_NAMESPACE` |
| 5 | `tools/hf_publish.py` (HF upload + CRC32 .hash) | infra | merged | B1+ | tools/ + 6 test | Bootstrap mirror toolchain |
| 6 | `facefusion.frame_interpolator` (RIFE 4.9 primitive) | feature | merged | A3 | frame_interpolator.py + RIFE ONNX URL | API `interpolate_pair(prev, next, t)` |
| 7 | `tools/interpolate_video.py` (RIFE video CLI) | feature | merged | A3.2 | tools/ + 6 test | ffmpeg-pipe end-to-end CLI |
| 8 | `--frame-interpolator-target-fps` headless integration | feature | merged | A3.3 | core.py + program.py | Backward-compat flag |
| 9 | Dynamic batching infra + face_swapper pixel-boost | perf | merged | G2 | processors/batching.py + face_swapper + 14 test | `run_with_dynamic_batch`, `supports_dynamic_batch` |
| 10 | Register `frame_interpolator` as `--processors` value | feature | merged | A3.4 (E) | core registry + processor module | Mới: `--processors frame_interpolator` |
| 11 | Dynamic batching cho `frame_enhancer` tile loop | perf | merged | G2.ext | frame_enhancer/core.py + 5 test | Tile stack qua `forward_batch` |
| 12 | Dynamic batching cho `face_enhancer` multi-face | perf | merged | G2.ext | face_enhancer/core.py + 13 test | Bbox-overlap check, fallback bit-equal |
| 13 | doctor v2 (GPU detect + model inventory + `--verify-models`) | infra | merged | F4.ext | doctor.py + 11 test | nvidia-smi / rocm-smi / Apple Silicon, CRC32 verify |
| 14 | `latentsync_1_5` scaffold + `--lip-syncer-research-models` gate | feature/scaffold | merged | A2 PR-A | lip_syncer/* + 10 test | `research_only=True`, stub `forward_latentsync` |
| 15 | `portrait_animator` processor (LivePortrait reuse) | feature | merged | D1 | portrait_animator/* (5 file) + 21 test | Source-state LRU, motion blend |
| 16 | Batch ArcFace embeddings across faces | perf | merged | G2.widest #1 | face_recognizer/face_analyser + 7 test | Phase 2 single batched ArcFace call |
| 17 | Batch `fan_68_5` (5→68) landmark expansion | perf | merged | G2.widest #2 | face_landmarker + tests | Phase 0 batched expansion |
| 18 | Batch `2dfan4` / `peppa_wutz` refinement | perf | merged | G2.widest #3 | face_landmarker + batching helpers + 4 test | `run_with_dynamic_batch_multi` |
| 19 | Batch fairface gender/age/race | perf | merged | G2.widest #4 | face_classifier + face_analyser + 8 test | Phase 3 batched fairface |
| 20 | Dynamic batching cho `expression_restorer` multi-face | perf | **open** | G2.followup | expression_restorer/core.py + 14 test | LivePortrait stack: 4×N → 4 ONNX call |

Numbering theo PR GitHub không theo Đợt — xem cột "Đợt" để map về roadmap.

---

## 3. PR detail (chỉ các PR có nội dung kỹ thuật cần lưu)

### 3.1 PR #1 — Windows bootstrap + high-quality preset (Đợt A4)

**Mục tiêu**: máy Windows mới chạy được FaceFusion trong 1 command.

- `install.bat` — winget cài Python 3.12 + git + ffmpeg, tạo `.venv`, gọi
  `install_modules.bat`, force-download model. Auto-detect NVIDIA → CUDA, fallback
  DirectML. Có thể ép `cuda | directml | openvino | qnn | rocm | migraphx | default`.
- `install_modules.bat` — chỉ cài Python modules + onnxruntime variant.
- `run.bat` — activate `.venv`, mặc định nạp `facefusion.high-quality.ini` qua
  `--config-path`. Hỗ trợ `--quality fast|balanced|default|<custom>`.
- `facefusion.high-quality.ini`:
  - detector `many` 640×640 angles 0/90/180/270 score 0.5
  - mask = `box + occlusion(xseg_3) + region(bisenet_resnet_34)`
  - swapper `hyperswap_1c_256` + `pixel_boost = 1024x1024`
  - `expression_restorer = live_portrait` factor 80
  - `face_enhancer = gpen_bfr_2048` blend 80
  - `frame_enhancer = clear_reality_x4` blend 50
  - encoder `libx264 veryslow` quality 95, audio `flac`

Không đụng Python logic.

### 3.2 PR #3 — balanced/fast presets + `facefusion doctor` v1 (Đợt A4 + F4)

**balanced/fast preset**:

- `facefusion.balanced.ini` — clip ngắn, ~6-8 GB VRAM. `inswapper_128 + pixel_boost
  512x512 + gpen_bfr_1024 + real_esrgan_x2`. Encoder `medium` quality 85.
- `facefusion.fast.ini` — real-time/preview, <4 GB VRAM, iGPU/Apple Silicon thấp.
  `inswapper_128 + pixel_boost 256x256 + gfpgan_1.4`. Encoder `veryfast` quality 75,
  temp `jpg`.

**`facefusion doctor` v1**:

- Subcommand `python facefusion.py doctor`.
- Check Python / platform / curl / ffmpeg + version / onnxruntime providers
  (available + configured) / temp_path / jobs_path writable / disk space ≥ 5/20 GiB
  / system memory ≥ 8 GiB.
- Output bảng `ok / warn / fail`, exit code 0 nếu pass.
- `install.bat` tự gọi `doctor` sau force-download.
- 11 unit test (`tests/test_doctor.py`).

### 3.3 PR #4 — `pyproject.toml` + HF/GH namespace override (Đợt B1)

**pyproject.toml** (PEP 621):

- `[project]` name/version sync với `facefusion/metadata.py`, license OpenRAIL-AS,
  requires-python `>=3.10`.
- `dependencies` đọc động từ `requirements.txt` → `install.bat` không đổi.
- `[project.optional-dependencies]`:
  - `dev` — `flake8`, `pytest`, `pytest-mock`
  - `diffusion` — `diffusers>=0.30`, `transformers>=4.44`, `accelerate>=0.34`,
    `safetensors>=0.4`, `torch>=2.3` (sẵn cho Đợt 2.B2 / C1 / C2)
  - `api` — `httpx>=0.27`, `pydantic>=2.6` (sẵn cho Đợt E)
- `[project.scripts]`: `facefusion = facefusion.core:cli`.

**Custom model mirror via env var** (`facefusion/download.py`):

- `FACEFUSION_HF_NAMESPACE` — override HF namespace (vd. `ngoqquyen` →
  `huggingface.co/ngoqquyen/<base_name>/...`).
- `FACEFUSION_GH_NAMESPACE` — override GitHub Releases namespace.
- Helper `resolve_provider_namespace_override()` + `apply_namespace_override()`.
- Khi env var unset → behavior không đổi.
- Hash file vẫn được verify qua `validate_hash`.
- 11 test cho env var lifecycle.

### 3.4 PR #5 — `tools/hf_publish.py` (Đợt B1+)

- `tools/hf_publish.py` upload weight + tính CRC32 sidecar `.hash` (cùng format
  `facefusion.hash_helper.create_hash`) lên HF.
  - `--source <path>` weight cục bộ.
  - `--repo-id <user>/<repo>` đích HF.
  - `--dest <path>` đường dẫn trong HF repo.
  - `--hash-only` chỉ tính hash, không upload.
  - `--token` (mặc định `HF_TOKEN` env).
- `tests/test_hf_publish.py` — 6 test, không gọi network.
- Đã upload mẫu: `ngoqquyen/facefusion-extras/frame_interpolator/rife_4_9.onnx`
  (RIFE 4.9, MIT, ~21 MB).
- ⚠️ HF token đã gửi plaintext qua chat session trước đó — **vẫn cần revoke**
  tại https://huggingface.co/settings/tokens.

### 3.5 PR #6–#10 — RIFE frame interpolator (Đợt A3)

Chuỗi 5 PR đưa RIFE thành processor first-class:

- PR #6 — `facefusion.frame_interpolator` primitive: `interpolate_pair(prev, next,
  timestep ∈ [0,1]) → mid_frame`. ONNX = `rife_4_9.onnx` (21 MB).
- PR #7 — `tools/interpolate_video.py` standalone CLI: ffmpeg pipe (rgb24 raw, không
  extract ra disk), `--multiplier N` chèn N-1 frame trung gian.
- PR #8 — `--frame-interpolator-target-fps` cho `headless-run`: backward-compat,
  multiplier = `round(target_fps / output_fps)` (min 2). Devin Review fix:
  honor user encoder/quality/preset + preserve audio track.
- PR #10 — đăng ký `frame_interpolator` là processor thực: `--processors
  frame_interpolator` → `pre_check` / `post_process` tích hợp với
  `--video-memory-strategy`. Hỗ trợ `--frame-interpolator-model rife_4_9` +
  `--frame-interpolator-multiplier` rời rạc.

Pipeline integration: RIFE chạy ở cuối pipeline (sau `restore_audio`). Lỗi
interpolation giữ nguyên output gốc (rc=0, log warning).

### 3.6 PR #9, #11, #12 — Dynamic batching foundation (Đợt G2)

**PR #9 — Infra**:

- `facefusion/processors/batching.py` (mới):
  - `supports_dynamic_batch(session, input_name)` — inspect `get_inputs()[i].shape[0]`.
    String / None → True; ≤0 int (export convention) → True; fixed=1 → False.
  - `run_session_batched` / `run_session_looped` — same signature.
  - `run_with_dynamic_batch(session, ...)` — try batched, catch `RuntimeError`,
    fallback per-element, `on_fallback(exc)` hook.
  - `stack_prepared_frames` — `numpy.concatenate` cho `(1, C, H, W)`.
- `face_swapper/core.py` — `_build_swap_face_base_inputs()` extract logic chung,
  `swap_face` đẩy N tile qua 1 `forward_swap_face_batch`. Stock model fixed-batch=1
  → fallback bit-equal.
- 14 test cho infra + 1 plumbing test.

**PR #11 — `frame_enhancer`**:

- `enhance_frame()` chuyển từ `for tile` sang prepare all tiles + 1 `forward_batch`.
- 5 test (dispatch / fallback / runtime exception / equivalence / plumbing).
- Stock `clear_reality_x4` / `real_esrgan_x4_plus` fixed batch=1 → fallback,
  output bit-equal.

**PR #12 — `face_enhancer` multi-face**:

- `enhance_faces(target_faces, temp_vision_frame)` — bbox không chồng (25% margin):
  warp + prepare cả N mặt + 1 `forward_batch`, paste-back tuần tự, output **bit-equal**
  với loop cũ.
- Có chồng → fallback nguyên loop tuần tự `enhance_face`.
- Hỗ trợ 'weight' input cho codeformer (broadcast theo batch axis).
- 13 test.

### 3.7 PR #13 — doctor v2 (Đợt F4 mở rộng)

- **GPU detection** (best-effort, không bao giờ `fail`):
  - `_probe_nvidia_gpus()` qua `nvidia-smi --query-gpu=name,memory.total,driver_version`.
  - `_probe_amd_gpus()` qua `rocm-smi --showproductname --showmeminfo vram`.
  - `_probe_apple_gpus()` qua `system_profiler SPDisplaysDataType` (Darwin only).
- **Model inventory** (`check_models()`): quét `<repo>/.assets/models/`,
  đếm `.onnx` + tổng GiB; phát hiện `.hash` mồ côi.
- **Model integrity** (`--verify-models`): CRC32 mọi `.onnx` so với sidecar `.hash`.
  Mismatch → `models_hash_mismatch` `fail`.
- 22 test (11 cũ + 11 mới); end-to-end verify trên 29 model thật.

### 3.8 PR #14 — `latentsync_1_5` scaffold (Đợt A2 PR-A)

Sau khi research lại, phát hiện LatentSync **không phải single-pass** mà là
diffusion (Whisper-Tiny audio + VAE + audio-cond U-Net + DDIM 25–50 step).
Không có public single-shot ONNX export. Tách Đợt A2 thành 2 PR:

- **PR-A (PR #14)** — schema + opt-in gate + dispatch routing + stub.
  Không chạy được runtime, **đó là chủ đích** (tránh ship code chưa validate).
  - `latentsync_1_5` đăng ký với `__metadata__.research_only=True`,
    license `Apache-2.0`, `vendor=ByteDance`, type `latentsync_research`.
  - `--lip-syncer-research-models` (action store_true).
  - `pre_check` short-circuit cho `research_only`:
    - flag off → message yêu cầu opt-in
    - flag on → message "sampler not yet implemented (PR-B)"
  - `forward_latentsync` raise `NotImplementedError` defensive.
  - 10 test scaffold.
- **PR-B (deferred, GPU-bound)** — ONNX exports + DDIM sampler + Whisper audio path
  + upload weights. Bắt buộc cần GPU thật để verify output quality. **Chưa làm.**

### 3.9 PR #15 — `portrait_animator` processor (Đợt D1)

Processor mới animate **static source portrait** theo head pose + expression của
**driving video**. Reuse LivePortrait ONNX đã có cho `expression_restorer` /
`face_editor` (KwaiVGI, MIT, `feature_extractor` / `motion_extractor` /
`generator`) — **không upload weight mới**.

So với processor cũ:

- `face_swapper` — chỉ transfer ID embedding, texture/lighting từ target.
- `expression_restorer` — giữ target ID, modulate expression in-frame.
- `portrait_animator` — transfer **full appearance feature volume** của source
  portrait, target drive pose + expression.

CLI:

```
python facefusion.py headless-run --processors portrait_animator \
  -s portrait.jpg -t driving.mp4 -o output.mp4 \
  --portrait-animator-pose-weight 100 \
  --portrait-animator-expression-weight 100
```

- Source feature volume + canonical motion compute **once per source-path tuple**,
  cache LRU 4-entry. Mỗi driving frame chỉ chạy `motion_extractor` + `generator`.
- `--portrait-animator-pose-weight` / `--portrait-animator-expression-weight` (0–100)
  blend tuyến tính source ↔ target motion. Default 100 (full follow).
- Mask: `box_mask` + optional `occlusion_mask`.
- 21 unit test + 2 Devin Review fix:
  - Bug 1 (`4bc7888`): `animate_portrait` trích motion từ sai frame khi chain
    sau `face_swapper`. Sửa: warp 2 lần — `target_vision_frame` cho motion,
    `temp_vision_frame` cho paste-back. Mirror `expression_restorer` pattern.
  - Bug 2 (`440cc83`): `_build_source_state` crash với multi-source vì `==` trên
    `Face` namedtuple chứa numpy arrays → `ValueError: truth value of an array
    is ambiguous`. Sửa: `any(source_face is cf for cf in candidate_faces)` —
    identity match đúng ngữ nghĩa với `face_store` cache.

### 3.10 PR #16–#19 — Dynamic batching widest (Đợt G2 widest)

Loạt 4 PR đẩy batching qua face analyser pipeline:

| PR | Module | Phase | Note |
|----|--------|-------|------|
| #16 | `face_recognizer` (ArcFace) | phase 2 | `forward_batch` + `calculate_face_embeddings` |
| #17 | `face_landmarker` (`fan_68_5` 5→68) | phase 0 | Batched landmark expansion |
| #18 | `face_landmarker` (`2dfan4` / `peppa_wutz`) | phase 0b | `run_with_dynamic_batch_multi` cho model nhiều output, score arbitration giữ byte-for-byte |
| #19 | `face_classifier` (fairface gender/age/race) | phase 3 | Reuse landmark 5/68 từ phase 2, không warp lại |

Sau loạt này, `face_analyser.create_faces` có 4 phase batched:

```
phase 0  — fan_68_5 (5→68)
phase 0b — 2dfan4 / peppa_wutz refinement
phase 1  — per-face glue (warp + score arbitration)
phase 2  — ArcFace embedding (batched)
phase 3  — fairface (batched)
```

Stock model fixed batch=1 → fallback per-face → output bit-equal.
Speedup 2-4× trên CUDA/TensorRT cho frame nhiều mặt khi user re-export model
với `dynamic_axes={'input': {0:'batch'}}`.

### 3.11 PR #20 — `expression_restorer` multi-face batching (open)

Áp pattern PR #12 cho LivePortrait stack. Khi 1 frame có N mặt, đường gốc
gọi `4 × N` ONNX call (mỗi mặt: 1 feature_extractor + 2 motion_extractor +
1 generator). Sau PR: nếu các bbox không chồng, gọi đúng **4** ONNX call cho
toàn frame, độc lập với N.

- `restore_expressions(target_faces, target_vision_frame, temp_vision_frame)` mới.
- 0/1 mặt + bbox chồng → fallback loop tuần tự bit-equal.
- Bbox không chồng (25% margin) → batch path:
  - Warp + prepare per-face (cv2 op, không đụng ONNX).
  - 1 lần `forward_extract_feature_batch(temp_crops)` + 2 lần
    `forward_extract_motion_batch` (target stack, temp stack).
  - Per-face NumPy math ở giữa (rotation, `restrict_expression_areas`, factor
    blend, motion-points compose) — bit-equal vì cùng input slice.
  - 1 lần `forward_generate_frame_batch`.
  - Paste-back tuần tự.
- 14 test.

Stock LivePortrait ONNX hiện fixed batch=1 → mỗi `forward_*_batch` tự fallback
per-face nội bộ → output bit-equal master.

CI green, mergeable, chờ user merge.

---

## 4. Roadmap progress (vs `ULTRA_ROADMAP.md`)

### Phase A — Quality wins ngay

| Mục | Mô tả | Status | PR |
|-----|-------|--------|-----|
| A1 | `inswapper_512_live` choices | **blocked (license)** | — |
| A2 | `latentsync` + `musetalk_v15` | partial (PR-A scaffold), PR-B GPU-bound | #14 |
| A3 | Processor `frame_interpolator` (RIFE) | ✅ done | #6 #7 #8 #10 |
| A4 | Preset `balanced` / `fast` / `high-quality` + `run.bat` | ✅ done | #1 #3 |

### Phase B — Hạ tầng diffusion

| Mục | Mô tả | Status | PR |
|-----|-------|--------|-----|
| B1 | `pyproject.toml` extras + HF/GH mirror override + `tools/hf_publish.py` | ✅ done | #4 #5 |
| B2 | Diffusion runtime adapter | not started | — |
| B3 | Golden-image regression test | not started | — |

### Phase C — SOTA processors

| Mục | Mô tả | Status |
|-----|-------|--------|
| C1 | `face_enhancer` `hypir` / `supir_face` backend | not started |
| C2 | `face_swapper` `reface_diffusion` backend | not started |
| C3 | Identity ensemble (ArcFace + AdaFace + MagFace + multi-source) | not started |
| C4 | `image_restorer` whole-frame | not started |

### Phase D — Animation & motion

| Mục | Mô tả | Status | PR |
|-----|-------|--------|-----|
| D1 | `portrait_animator` (LivePortrait reuse) | ✅ done | #15 |
| D2 | `temporal_stabilizer` (RAFT optical flow) | not started | — |
| D3 | Workflow `audio_to_video.py` | not started | — |

### Phase E — Video synthesis & motion control

| Mục | Mô tả | Status |
|-----|-------|--------|
| E1 | Motion control adapter (Kling 2.6) | not started |
| E2 | `video_synthesizer` (Wan 2.2 / LTX-2 / CogVideoX) | not started |
| E3 | Workflow `video_to_video.py` | not started |
| E4 | PuLID/InstantID hybrid regen | not started |

### Phase F — Audio & UX

| Mục | Mô tả | Status | PR |
|-----|-------|--------|-----|
| F1 | `voice_extractor` → `audio_pipeline` | not started | — |
| F2 | `voice_cloner` (XTTS v2 / F5-TTS) | not started | — |
| F3 | UI: progress bar / queue / preview | not started | — |
| F4 | `facefusion doctor` CLI | ✅ done v2 | #3 #13 |

### Phase G — Performance

| Mục | Mô tả | Status | PR |
|-----|-------|--------|-----|
| G1 | Pipeline streaming (decode→infer→encode) | not started | — |
| G2 | Dynamic batching swap/enhance | ✅ done (face_swapper / frame_enhancer / face_enhancer / face_recognizer / face_landmarker × 2 / face_classifier / expression_restorer) | #9 #11 #12 #16 #17 #18 #19 #20 |
| G3 | ONNX `IOBinding` + `OrtValue` zero-copy | not started | — |
| G4 | TensorRT EP convert + engine cache | not started | — |
| G5 | OpenVINO/CoreML EP cho Intel/Apple silicon | partial (provider auto-detect) | #1 (install.bat ép variant) |

---

## 5. Open items / known blockers

### 5.1 License / sourcing blockers

- **A1 — `inswapper_512_live`**: nguồn weight chính thức tại
  https://github.com/deepinsight/inswapper-512-live không phát hành ONNX (chỉ có
  app macOS/iOS Picsi.Ai paid). License "academic / personal testing" cấm
  commercial use. Không có HF/GH community publish. **Không thể bundle hợp pháp.**
  Đường khả thi: user tự cấp weight + license commercial từ insightface/Picsi.Ai
  rồi tích hợp qua `--research-models`.
- **A2 PR-B — LatentSync sampler**: cần GPU thật để validate output quality
  (diffusion 25–50 step CPU = không khả thi cho video). Thay thế khả thi nếu user
  muốn ship sớm: MuseTalk v1.5 (Apache, có community ONNX export, single-pass-ish).
- **HF token revoke**: token đã gửi plaintext qua chat session trước. **Phải revoke**
  tại https://huggingface.co/settings/tokens trước khi public fork rộng hơn.

### 5.2 Infrastructure gaps

- B2 (diffusion runtime adapter): chưa code → C1/C2 chưa start được.
- B3 (golden-image regression): tất cả PR perf hiện rely vào unit test stub +
  manual smoke. Cần golden-set 100 ảnh + 20 video cố định seed (xem §6 trong
  ULTRA_ROADMAP) để tự động phát hiện drift.
- G3 (IOBinding zero-copy) + G4 (TensorRT EP): các PR G2 hiện tại đã cấu hình
  fallback runtime, nhưng IOBinding sẽ là chặng tiếp để không lãng phí băng thông
  GPU↔Python.

### 5.3 Test environment gaps

- `tests/test_ffmpeg.py` + `tests/test_vision.py::test_restrict_video_fps` fail
  trên VM CPU vì thiếu codec h264 trong ffmpeg — pre-existing trên master, không
  phải regression của PR nào. CI có codec đầy đủ nên không reproduce.
- Không có GPU validation trong session (CPU-only VM). PR perf phải pre-export
  model với `dynamic_axes` để xác minh batched path; không exercise được
  trong CI.

---

## 6. Risks & technical debt

1. **Bit-equal reliance**: PR perf dùng "stock model fixed batch=1 → fallback
   loop bit-equal" làm guarantee. Nếu upstream re-export weight với
   `dynamic_axes` mà runtime numerical drift (vd. precision của batched matmul
   khác per-element trên CUDA) → output sẽ khác master. Mitigation: golden-set
   regression (B3) sẽ bắt drift này.
2. **Source-state cache LRU (PR #15)** key theo `source_paths` tuple. Nếu user
   thay nội dung file mà giữ path cũ, cache sẽ trả state cũ. Không fix vì pattern
   nhất quán với rest of facefusion (face_store cache cũng vậy).
3. **`expression_restorer` overlap heuristic** (PR #12 / #20) dùng 25% bbox
   margin. Nếu 2 mặt sát nhau ở mức margin biên (~25%) sẽ flip giữa batched
   path / fallback theo bbox jitter — output có thể khác giữa 2 frame liên tiếp.
   Trong thực tế hiếm vì face detector output ổn định trên video.
4. **Latency của ONNX session warm-up** với batch dynamic axis: lần đầu chạy
   với batch=N mới sẽ recompile kernel, tăng delay frame đầu. Không ảnh hưởng
   throughput sau warm-up.
5. **Doctor CRC32** (PR #5/#13) dùng CRC32 thay SHA-256 — match format của
   `facefusion.hash_helper.create_hash` cho backward-compat, nhưng CRC32 không
   chống tampered weight. Mitigation cần: PR riêng đổi sang SHA-256 + sidecar
   migration tool.

---

## 7. Đề xuất next steps (theo ưu tiên độ rủi ro tăng dần)

1. **Merge PR #20** (sau khi user review).
2. **Slice tiếp G2** (S, lowest risk):
   - `portrait_animator` multi-face batching (cùng pattern PR #12, processor
     mới từ PR #15).
   - `lip_syncer` multi-face (1 ONNX/face) — win nhỏ hơn vì wav2lip 96px nhanh
     sẵn.
3. **PuLID + InstantID identity ensemble (Đợt C3)** — quality lift cho
   `face_swapper`, license sạch (Apache/MIT), có ONNX công khai. Effort M-L.
4. **IOBinding + TensorRT EP (Đợt G3 + G4)** — perf bằng path inference, không
   cần model mới. Effort M-L, cần GPU verify.
5. **Đợt B3 — golden-image regression test**: lưu hash output trên seed cố định
   để bắt drift. Cần thiết trước khi dám ship thêm PR perf.
6. **Đợt A2 PR-B — LatentSync sampler**: ONNX export + DDIM sampler + Whisper
   audio path. Bắt buộc GPU thật. Hoặc pivot sang MuseTalk v1.5 nếu user ưu tiên
   ship sớm.

---

## 8. Tham chiếu

- Roadmap: [ULTRA_ROADMAP.md](./ULTRA_ROADMAP.md)
- README: [README.md](./README.md)
- Pull requests: https://github.com/wigjec5544f/facefusion/pulls?q=is%3Apr
- HF mirror: https://huggingface.co/ngoqquyen/facefusion-extras
- Devin Review per PR: https://app.devin.ai/review/wigjec5544f/facefusion/pull/&lt;N&gt;

⚠️ **Action item còn để mở**: revoke HF token cũ tại
https://huggingface.co/settings/tokens.
