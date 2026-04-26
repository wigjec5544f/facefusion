# FaceFusion – Ultra Roadmap

> Khảo sát SOTA 2024–2026 (open-source + API closed-source) và mapping vào kiến
> trúc facefusion. Mục tiêu: nâng chất lượng output lên ngang/vượt các pipeline
> proprietary (Kling, Seedream, Sora) trong khi vẫn chạy local hoặc lai cloud.
>
> Văn bản này là **plan**, không phải mã. PR triển khai sẽ được tách thành
> nhiều bước nhỏ ở phần [Roadmap](#roadmap--milestones) bên dưới.

---

## 0. Bối cảnh

Facefusion hiện tại = pipeline GAN-based hoán đổi mặt + một số processor phụ
(face/frame enhancer, expression restorer, lip syncer wav2lip 96px). Các giới
hạn chính so với SOTA 2025–2026:

| Giới hạn hiện tại | SOTA tương ứng |
|-------------------|---------------|
| swap chạy ở 256×256 (pixel_boost lên 1024 chỉ là tile) | inswapper-512-live, REFace (diffusion), DeepFaceLab successors |
| identity preservation chỉ từ ArcFace embedding | PuLID, InstantID, ConsistentID (multimodal) |
| portrait animation = LivePortrait v1 | LivePortrait v2, X-Portrait, AniPortrait |
| lip-sync wav2lip 96 / edtalk_256 | LatentSync, MuseTalk v2, Hallo3, V-Express |
| face restore = GFPGAN/CodeFormer/GPEN ≤2K | SUPIR, HYPIR, DiffBIR v2.1 |
| frame enhance = Real-ESRGAN family | LTX-2 spatial, SeedVR 2 |
| motion control = none | Kling 2.6 Motion Control, MotionCtrl, DragAnything |
| video synthesis / regen = none | HunyuanVideo, Wan 2.2, CogVideoX, LTX-Video |
| frame interpolation = none | GIMM-VFI, RIFE 4.x, EMA-VFI |
| temporal consistency post-pass = none | StableAnimator, FlowVid, Rerender-A-Video |

---

## 1. Khảo sát thuật toán (cập nhật T4/2026)

### 1.1. Face swap (thay/bổ sung face_swapper)

| Model | Năm | License | Resolution | Đặc điểm |
|------|-----|---------|------------|----------|
| **inswapper-512-live** (deepinsight) | 2025-08 | Custom (free for non-commercial) | 512×512 | 16× pixel của inswapper_128, chỉ cần ~1/10 compute; mục tiêu real-time on-device |
| **REFace** (Sanoojan, WACV 2025) | 2024-09 | MIT | latent | Diffusion-based swap, identity từ ArcFace + structure từ landmarks |
| **HyperSwap 1c/2** | 2025 | research | 256/512 | Đã có trong facefusion (`hyperswap_1c_256`); v2 chưa public |
| **DeepFaceLab2 / Roop** | — | GPL | varies | Workflow truyền thống, không cải tiến chất lượng |
| **PuLID** (ByteDance, NeurIPS 2024) | 2024 | Apache-2.0 | SDXL/Flux | Identity adapter cho diffusion → cho phép swap "diffusion-style" giữ ID |
| **InstantID** (InstantX) | 2024-01 | Apache-2.0 | SDXL | Identity-preserving image generation từ 1 ảnh tham chiếu |
| **ConsistentID** | 2024-04 | research | SD1.5 | Multimodal ID prompt (face description + ArcFace) |

→ Đề xuất: thêm 3 backend cho processor `face_swapper`:
- `inswapper_512_live` — drop-in upgrade cho inswapper_128, có sẵn ONNX.
- `reface_diffusion` — backend chất lượng cao nhất, chậm hơn nhiều (cần SD1.5).
- `pulid_flux` — hybrid: dùng PuLID + Flux để regenerate frame có giữ ID gốc
  (bypass swap GAN, dùng cho ảnh tĩnh chất lượng cực cao).

### 1.2. Face / portrait animation (thay/bổ sung expression_restorer & face_editor)

| Model | License | Đầu vào | Ưu |
|------|---------|---------|----|
| **LivePortrait v1** (Kuaishou) | MIT | image + driving video | Đã có |
| **X-Portrait 2** | research | image + driving | Cinematic head/body, giữ ID tốt |
| **AniPortrait** (Tencent) | Apache-2.0 | image + audio/video | Audio-driven portrait |
| **EMO** (Alibaba) | research closed | image + audio | Talking head chất lượng cao, không OS |
| **V-Express** (Tencent) | Apache-2.0 | image + audio | Conditional control: bbox/pose/audio |
| **Hallo / Hallo2 / Hallo3** | MIT | image + audio | 4K, 1h video, hierarchical audio cond |
| **StableAnimator** | research | image + pose | ID-preserving full-body animation |

→ Đề xuất:
- Mở rộng processor `expression_restorer` để hỗ trợ `liveportrait_v2` (khi
  release).
- Processor mới `portrait_animator` với backends: `aniportrait`, `hallo3`,
  `v_express`, `stable_animator` — cho phép drive 1 ảnh tĩnh thành video bằng
  audio hoặc driving video.

### 1.3. Lip-sync (thay/bổ sung lip_syncer)

| Model | License | Resolution | Note |
|------|---------|------------|------|
| **wav2lip_gan_96** | research | 96×96 | Đã có, thấp |
| **edtalk_256** | research | 256×256 | Đã có |
| **MuseTalk v1.5** (Tencent) | MIT | 256+ | Real-time, latent space |
| **LatentSync** (ByteDance, 2025) | Apache-2.0 | 512 | Stable Diffusion-based, sync rất chính xác |
| **Hallo2/3** | MIT | up to 4K | Long-form audio-driven |
| **DiffTalk** | research | 512 | Diffusion-based |

→ Đề xuất: thêm `latentsync` và `musetalk_v15` làm 2 backend mặc định mới.
Wav2lip giữ lại để dùng cho real-time/streaming.

### 1.4. Identity preservation (module mới)

Hầu hết swap hiện chỉ dùng 1 ArcFace embedding. SOTA 2025 dùng:
- ArcFace + AdaFace + face landmarks + face descriptor LLM (PuLID, ConsistentID).
- Multi-shot reference (3–5 ảnh source) thay vì 1 ảnh duy nhất.

→ Đề xuất module `face_recognizer` mở rộng:
- Hỗ trợ ensemble `arcface_w600k_r50` + `adaface_ir101` + `magface`.
- Cho phép nhiều ảnh source và lấy mean embedding (đã có 1 phần qua
  `get_average_face`); thêm weighted average + outlier rejection.
- Hook để feed embedding mở rộng vào các processor diffusion (PuLID/InstantID).

### 1.5. Face / image restoration (thay frame_enhancer + face_enhancer)

| Model | License | Note |
|------|---------|------|
| **GFPGAN/CodeFormer/GPEN** | đã có | baseline |
| **RestoreFormer++** | research | đã có 1 phần |
| **SUPIR** (CVPR 2024) | dual: research + commercial | SDXL-based, restore "in the wild" cực mạnh, cần ~24 GB VRAM |
| **HYPIR** (XPixelGroup, 2025) | open | Successor DiffBIR, nhanh hơn 10×, chất lượng vượt |
| **DiffBIR v2.1** | Apache-2.0 | Tile-based, MPS support |
| **SeedVR 2** (ByteDance) | research | Video super-resolution |
| **Real-ESRGAN family** | BSD/Apache | đã có |

→ Đề xuất:
- Processor `face_enhancer` thêm backend `supir_face` (sd-based) và `hypir`.
- Processor mới `image_restorer` (toàn ảnh, không chỉ face crop) dùng `hypir`
  hoặc `supir`. `frame_enhancer` cũ vẫn giữ cho video real-time.

### 1.6. Frame interpolation (module mới)

Facefusion hiện không có interpolation → output bị giới hạn ở fps gốc của
target. Để smooth slow-motion / nâng fps:

| Model | License | Note |
|------|---------|------|
| **RIFE 4.26** | MIT | Real-time, 2×–8×, tốt nhất cho tốc độ |
| **GIMM-VFI** (NeurIPS 2024) | Apache-2.0 | Generalizable implicit motion modeling, chất lượng cao hơn RIFE |
| **EMA-VFI** | research | Chất lượng cao, chậm hơn |
| **FILM** (Google) | Apache-2.0 | Cho transition lớn |

→ Đề xuất processor mới `frame_interpolator` với backends `rife_4.26`,
`gimm_vfi`. Đặt sau `frame_enhancer` trong pipeline.

### 1.7. Motion control / video synthesis (module hoàn toàn mới)

Đây là phần Kling/Seedream/Sora đang dẫn đầu. Mapping vào facefusion:

| Capability | Open model | License | Chú ý |
|------------|-----------|---------|------|
| Image-to-video | **Wan 2.2** (Alibaba) | Apache-2.0 | MoE, cinematic, 720p/1080p, ~24 GB VRAM |
| Image-to-video | **HunyuanVideo I2V** (Tencent) | Custom OS | ~60 GB VRAM full, có quantized 24 GB |
| Image-to-video | **LTX-2** (Lightricks) | OpenRAIL-M | Near real-time, 1:192 VAE compression |
| Image-to-video | **CogVideoX 5B** (Zhipu) | Apache-2.0 | Lightweight |
| Motion transfer | **Kling 2.6 Motion Control** | API closed | Tham chiếu video → áp dụng motion lên ảnh; max 30s |
| Motion transfer (OS) | **MotionCtrl / CameraCtrl** | Apache-2.0 | Camera + object motion control cho SVD |
| Motion transfer (OS) | **DragAnything / DragNUWA** | research | Drag-based motion control |
| Pose-driven | **MimicMotion** (Tencent) | Apache-2.0 | Animate from pose sequence |
| Pose-driven | **MagicAnimate / Animate-X** | research | Full-body animation |
| Bilingual T2I | **Seedream 2.0** (ByteDance) | API closed | T2I, mạnh tiếng Trung; có thể dùng làm source generator (không phải swap) |
| 3D awareness | **Seed3D 2.0** | API closed | Single-image → 3D mesh (cho relight/view-synthesis trong tương lai) |

→ Đề xuất 2 module mới (lớn, để P1/P2 thay vì P0):
- **`motion_control` processor**: nhận `(source_image, driving_video)` + prompt
  → output video. Backend: cloud API (`kling_2_6_motion_control` qua fal.ai
  hoặc Volcano Engine) cho chất lượng cao nhất; backend OS:
  `motionctrl_svd`, `mimicmotion`, `dragnuwa` cho self-host.
- **`video_synthesizer` processor**: I2V/T2V regeneration. Backend OS:
  `wan_2_2_i2v`, `ltx_2_i2v`, `hunyuanvideo_i2v`. Backend API:
  `kling_2_6`, `seedream_video` (khi public).

### 1.8. Temporal consistency post-pass (module mới)

Vấn đề lớn của swap GAN trên video: flicker mí mắt / texture da. Các kỹ thuật:

| Method | License | Note |
|--------|---------|------|
| **StableAnimator** | research | Identity-preserving temporal stabilization |
| **FlowVid** | research | Optical flow + diffusion stabilizer |
| **Rerender-A-Video** | Apache-2.0 | Frame-by-frame consistent stylization |
| **CoDeF** (content deformation field) | research | Per-video implicit field |

→ Đề xuất processor `temporal_stabilizer` chạy ở cuối pipeline. Backend
optical-flow-based (RAFT + warping) cho preset lite, diffusion-based cho
preset full.

### 1.9. Audio (mở rộng voice_extractor)

| Capability | Model | License |
|------------|-------|---------|
| Speech enhance | **Resemble Enhance** | Apache-2.0 |
| Vocal separation | **MDX-Net 23C / BS-Roformer** | MIT/research |
| Voice clone | **F5-TTS, OpenVoice v2, Coqui XTTS v2** | varies |
| Lip-sync paired audio gen | **EMOTalk, Hallo3 audio frontend** | research |

→ Đề xuất:
- Mở rộng `voice_extractor` → `audio_pipeline` với `denoise → separate →
  optional re-clone`.
- Thêm processor `voice_cloner` (off mặc định, có watermark + consent gating).

---

## 2. Mapping vào kiến trúc facefusion

```
facefusion/
├── processors/
│   └── modules/
│       ├── face_swapper/         (+ inswapper_512_live, reface, pulid_flux)
│       ├── face_enhancer/        (+ supir_face, hypir)
│       ├── frame_enhancer/       (+ hypir, ltx2_spatial)
│       ├── expression_restorer/  (+ liveportrait_v2)
│       ├── lip_syncer/           (+ latentsync, musetalk_v15, hallo3)
│       ├── frame_interpolator/   ★ MỚI (rife, gimm_vfi)
│       ├── portrait_animator/    ★ MỚI (aniportrait, v_express, hallo3, stable_animator)
│       ├── motion_control/       ★ MỚI (kling_api, motionctrl_svd, mimicmotion)
│       ├── video_synthesizer/    ★ MỚI (wan_2_2_i2v, ltx_2_i2v, hunyuanvideo_i2v)
│       ├── temporal_stabilizer/  ★ MỚI (raft_warp, flowvid)
│       ├── image_restorer/       ★ MỚI (hypir, supir, diffbir)
│       └── voice_cloner/         ★ MỚI (xtts_v2, f5_tts; OFF default + consent gate)
├── face_recognizer.py             (+ ensemble: arcface + adaface + magface, multi-source mean)
├── inference_manager.py           (+ diffusion runtime adapter: diffusers + accelerate)
└── workflows/
    ├── image_to_image.py
    ├── image_to_video.py
    ├── video_to_video.py          ★ MỚI (motion transfer)
    └── audio_to_video.py          ★ MỚI (talking head from still image + audio)
```

Chi tiết runtime:
- Diffusion-based processors cần `diffusers` + `transformers` + `accelerate`,
  không nằm trong onnxruntime stack. Đề xuất tách thành extra:
  `pip install facefusion[diffusion]` (xem PR pyproject.toml roadmap riêng).
- Cloud API (Kling, Seedream) → adapter `processors/modules/.../api_backend.py`
  với secret từ `state_manager.get_item('api_keys')` (load từ env / repo secret).

---

## 3. Preset chất lượng theo use case

Mở rộng `facefusion.high-quality.ini` thành 4 preset:

| Preset | Use case | Pipeline |
|--------|----------|----------|
| `fast` | real-time webcam / preview | inswapper_128 + gfpgan_1.4 |
| `balanced` (= mặc định mới) | clip ngắn, máy 8 GB VRAM | inswapper_512_live + gpen_bfr_1024 + clear_reality_x4 |
| `high` (= preset hiện tại) | video chất lượng cao, 12 GB VRAM | hyperswap_1c + pixel_boost 1024 + gpen_bfr_2048 + clear_reality_x4 |
| `ultra` ★ | tối đa, ≥24 GB VRAM hoặc cloud | reface_diffusion + supir_face + hypir + latentsync + rife_4.26 + temporal_stabilizer (flowvid) + optional motion_control |
| `ultra-hybrid` ★ | local + cloud API | high local + Kling motion_control / Seedream T2I source / SeedVR2 upscale từ cloud |

Mỗi preset có file `.ini` riêng + entry tương ứng trong `run.bat`
(`run.bat --quality ultra`).

---

## 4. Roadmap & milestones

Mỗi milestone là 1 PR độc lập, không phụ thuộc nhau (trừ khi note rõ).

### Phase A – Quality wins ngay (ROI cao, effort thấp)

- **A1.** Thêm `inswapper_512_live` vào `face_swapper` choices. Mới có ONNX,
  drop-in. *Effort: S. VRAM: thấp.*
- **A2.** Thêm `latentsync` và `musetalk_v15` vào `lip_syncer`. Cải thiện
  lip-sync rõ rệt. *Effort: M. Cần diffusers stack.*
- **A3.** Thêm processor `frame_interpolator` (`rife_4.26` + `gimm_vfi`).
  Output 60 fps mượt. *Effort: M. ONNX có sẵn cho RIFE.*
- **A4.** Thêm preset `ultra-fast` & `balanced` vào `.ini` + `run.bat`.
  *Effort: S.*

### Phase B – Hạ tầng diffusion

- **B1.** Tách extras: `pyproject.toml` với `[diffusion]` (diffusers,
  transformers, accelerate, xformers). *Effort: M.*
- **B2.** Diffusion runtime adapter (mirror của `inference_manager` cho HF
  pipelines, với memory offload + sequential CPU offload). *Effort: L.*
- **B3.** Test golden-image regression (lưu hash output trên seed cố định)
  để bắt drift. *Effort: M.*

### Phase C – SOTA processors

- **C1.** `face_enhancer` thêm backend `hypir` + `supir_face` (chỉ qua extras).
  *Phụ thuộc B1, B2. Effort: M.*
- **C2.** `face_swapper` thêm backend `reface_diffusion`. *Phụ thuộc B1, B2.
  Effort: L.*
- **C3.** Identity ensemble (`arcface + adaface + magface`) + multi-source
  weighted mean. *Effort: M.*
- **C4.** Processor `image_restorer` (toàn ảnh, không crop). *Effort: M.*

### Phase D – Animation & motion

- **D1.** Processor `portrait_animator` với `aniportrait`/`hallo3`. *Effort: L.*
- **D2.** Processor `temporal_stabilizer` (RAFT-based optical flow first,
  diffusion later). *Effort: L.*
- **D3.** Workflow mới `audio_to_video.py` (still image + audio → talking
  head). *Effort: M sau D1.*

### Phase E – Video synthesis & motion control

- **E1.** Processor `motion_control` adapter cho Kling 2.6 (cloud API). Yêu
  cầu user nhập API key qua secret. *Effort: M.*
- **E2.** Processor `video_synthesizer` cho Wan 2.2 / LTX-2 / CogVideoX local.
  *Effort: L. Cần ≥24 GB VRAM cho Wan 2.2 full.*
- **E3.** Workflow `video_to_video.py` (driving video + source image → kết
  hợp swap + motion transfer). *Effort: L sau E1, E2.*
- **E4.** PuLID/InstantID hybrid: regen frame qua Flux/SDXL với ID adapter
  thay vì swap GAN, cho ảnh tĩnh chất lượng tối đa. *Effort: L.*

### Phase F – Audio & UX

- **F1.** `voice_extractor` → `audio_pipeline` (denoise + separate). *Effort: M.*
- **F2.** Processor `voice_cloner` (XTTS v2 / F5-TTS), **OFF mặc định** + UI
  consent gate (không bypass content_analyser). *Effort: M.*
- **F3.** UI: progress bar chi tiết, queue persistence, preview side-by-side
  trước/sau. *Effort: M.*
- **F4.** `facefusion doctor` CLI: check ffmpeg/GPU/providers/models/disk,
  in bảng + recommendation. *Effort: S.*

### Phase G – Performance

- **G1.** Pipeline streaming (decode → infer → encode qua queue, bỏ ghi PNG
  trung gian). *Effort: L.*
- **G2.** Dynamic batching cho swap/enhance (gom face cùng frame). *Effort: M.*
- **G3.** ONNX `IOBinding` + `OrtValue` cho zero-copy GPU. *Effort: M.*
- **G4.** TensorRT EP convert + benchmark, lưu engine cache vào `.cache/trt`.
  *Effort: L.*
- **G5.** OpenVINO/CoreML EP cho Intel/Apple silicon. *Effort: M.*

---

## 5. License & feasibility

Phân loại license cho từng dependency mới:

| Tier | Models | Có thể bundle/auto-download? |
|------|--------|------------------------------|
| **OS friendly (Apache/MIT/BSD)** | Wan 2.2, ConsistentID-paper, AniPortrait, V-Express, Hallo, MuseTalk, LatentSync, RIFE, GIMM-VFI, MimicMotion, MotionCtrl, CogVideoX, PuLID, InstantID, DiffBIR | Có, tải qua download.py như hiện tại |
| **OpenRAIL / Custom OS** | LTX-2, HunyuanVideo, inswapper-512-live | Có với điều khoản; cần ghi rõ trong README + UI |
| **Research-only / dual** | SUPIR, REFace, Hallo3, EMO, X-Portrait | Cần opt-in flag, không enable mặc định |
| **Closed / API only** | Kling 2.6, Seedream 2.0, Seed3D 2.0, Sora | Adapter API, không bundle weight; user tự cấp key |

Rules cho repo:
1. Default install (`install.bat` không flag) chỉ tải tier 1+2.
2. Tier 3 (research-only) cần flag `--research-models` + hiển thị license
   prompt 1 lần.
3. Tier 4 (cloud) tách hẳn vào extras `[cloud]`, không tải gì local; key
   lưu qua `state_manager` từ env / config.
4. License OpenRAIL-AS hiện của facefusion phải được duy trì; mọi dependency
   mới phải tương thích (đã check ở bảng trên — tất cả OK).

---

## 6. Đo lường (làm sao biết đã "ultra")

Định nghĩa metric khách quan + golden-set 100 ảnh + 20 video chuẩn.

### 6.1. Quality metrics

| Metric | Đo cái gì | Tool |
|--------|-----------|------|
| **ID similarity** | swap có giữ ID source không | ArcFace cosine giữa output face và source |
| **Pose preservation** | output có đúng pose target không | 6DRepNet head pose error |
| **Expression preservation** | output có đúng biểu cảm target | EmoNet / OpenFace AU error |
| **Image quality** | sắc nét, không artefact | LPIPS, MUSIQ, NIQE, MANIQA |
| **Temporal consistency** (video) | flicker | warp error (RAFT flow) + tOF / FVD |
| **Lip-sync** | audio-mouth alignment | SyncNet score, AVSyncScore |

### 6.2. Performance metrics

- Frames per second per provider (CPU / DirectML / CUDA / TensorRT).
- VRAM peak per processor.
- Cold start time (model load).

### 6.3. CI integration

- Tạo `tests/golden/` với ảnh + hash + threshold per metric.
- PR fail nếu metric tệ hơn baseline > X% hoặc fps tụt > Y%.
- Báo cáo mỗi PR comment kèm bảng diff.

---

## 7. Bảo mật & ràng buộc

- **Giữ nguyên `content_analyser`** ở mọi processor mới. Mọi pipeline
  diffusion (PuLID, REFace, video synth) đều phải qua filter trước khi xuất
  ra disk.
- Cloud API: không gửi ảnh người ra server bên thứ ba khi chưa có consent
  prompt rõ ràng từ user.
- Voice cloner / talking head: thêm watermark có thể tắt được nhưng default
  ON; log file metadata (model, timestamp, prompt).
- Model signing: verify SHA256 + (tương lai) chữ ký sigstore khi tải.

---

## 8. Tóm tắt ưu tiên (TL;DR)

```
P0 (làm ngay, ROI lớn):
  A1 inswapper_512_live
  A2 latentsync + musetalk_v15
  A3 frame_interpolator (RIFE / GIMM-VFI)
  A4 preset balanced/ultra-fast
  G2 dynamic batching

P1 (1-2 tháng, cần extras diffusion):
  B1-B3 pyproject + diffusion runtime + golden test
  C1-C4 hypir / supir_face / reface / image_restorer / identity ensemble

P2 (2-4 tháng, capability mới):
  D1-D3 portrait_animator + temporal_stabilizer + audio_to_video workflow
  E1 motion_control via Kling API (cloud)
  G1 streaming pipeline

P3 (4-6 tháng, đầu tư lớn):
  E2-E4 video_synthesizer (Wan/LTX/Hunyuan) + video_to_video + PuLID hybrid
  G3-G5 IOBinding + TensorRT + OpenVINO/CoreML

Cross-cutting:
  F4 facefusion doctor
  F1-F2 audio pipeline + voice cloner (with consent)
  F3 UX persistence + side-by-side preview
```

---

## 9. Lưu ý triển khai

- Mọi PR triển khai cần đi kèm: golden-image test (Phase B3), benchmark
  trước/sau, license note trong PR description, update README usage.
- Diffusion processors nên fallback gracefully nếu user không cài extras
  `[diffusion]` (in cảnh báo, gợi ý lệnh `pip install facefusion[diffusion]`).
- Cloud processors nên fallback gracefully nếu thiếu API key.
- Mỗi model mới: ghi kích thước weight + VRAM yêu cầu + license vào
  `facefusion/processors/modules/.../README.md` để user dễ tra cứu.

---

## 10. Liên kết tham khảo

- Seedream 2.0 paper: https://arxiv.org/abs/2503.07703
- Kling 2.6 Motion Control: https://fal.ai/learn/devs/kling-video-2-6-motion-control-prompt-guide
- Wan 2.2: https://github.com/Wan-Video/Wan2.2
- HunyuanVideo: https://github.com/Tencent/HunyuanVideo
- LTX-Video / LTX-2: https://github.com/Lightricks/LTX-Video
- inswapper-512-live: https://github.com/deepinsight/inswapper-512-live
- REFace: https://github.com/Sanoojan/REFace
- LivePortrait: https://github.com/KwaiVGI/LivePortrait
- AniPortrait: https://github.com/Zejun-Yang/AniPortrait
- LatentSync: https://github.com/bytedance/LatentSync
- MuseTalk: https://github.com/TMElyralab/MuseTalk
- Hallo: https://github.com/fudan-generative-vision/hallo
- SUPIR: https://github.com/Fanghua-Yu/SUPIR
- DiffBIR / HYPIR: https://github.com/XPixelGroup/DiffBIR
- PuLID: https://github.com/ToTheBeginning/PuLID
- InstantID: https://github.com/InstantID/InstantID
- ConsistentID: https://arxiv.org/abs/2404.16771
- StableAnimator: https://github.com/Francis-Rings/StableAnimator
- MimicMotion: https://github.com/Tencent/MimicMotion
- RIFE: https://github.com/megvii-research/ECCV2022-RIFE
- GIMM-VFI: https://github.com/GSeanCDAT/GIMM-VFI
