# Đợt 1.C3 — PuLID + InstantID identity ensemble: research + plan

## TL;DR

Sau research kỹ, **PuLID** và **InstantID** nguyên gốc đều là *diffusion
conditioner* (inject identity vào Stable Diffusion XL / FLUX), **không
có** path standalone non-diffusion để feed vào inswapper. Để đúng nghĩa
"tích hợp PuLID+InstantID" cần ship SDXL backbone (4-12 GB weight,
GPU-only, license bundling phức tạp) — out of scope cho 1 PR.

**Đề xuất phương án thực tế**: chia thành 3 PR theo độ rủi ro tăng dần.
PR đầu tiên (**slice 1, ship trong PR #23**) làm **Identity Ensemble v1**
— multi-source embedding fusion với weighted averaging + outlier
rejection, license-clean, CPU-testable, bit-equal default. Đây chính
xác là cái user đề cập: "kết hợp embedding của nhiều ảnh source →
identity bền vững hơn inswapper_128 đơn lẻ".

---

## 1. Research findings

### 1.1 PuLID v1.1 (`ToTheBeginning/PuLID`, Apache 2.0)

**Architecture** (từ `pulid/pipeline.py` + nunchaku docs):

- **ArcFace antelopev2** → 512-dim ID embedding.
- **EVA-CLIP image encoder** → multi-scale ViT features (4 scales).
- **IDFormer**: Perceiver transformer (10 layer, 1024-dim, 16 heads)
  fuse ArcFace + EVA-CLIP → **2048-dim output token sequence**.
- 2048-dim output **chỉ** hữu dụng khi inject vào SDXL U-Net
  cross-attention qua `IDAttnProcessor`.
- IDFormer output **không thể** decode về face crop hay feed vào
  inswapper-style swap network — kiến trúc fundamentally incompatible.

**ONNX availability**:

- ✅ ArcFace antelopev2 (đã có sẵn — facefusion đang dùng
  `arcface_w600k_r50` cùng họ InsightFace).
- ✅ EVA-CLIP có ONNX export (community).
- ❌ **IDFormer KHÔNG có ONNX export công khai**. Chỉ có
  `pulid_v1.1.safetensors` PyTorch weight.
- ❌ Không có path bypass diffusion để dùng IDFormer output.

**License**: Apache 2.0 (clean), nhưng dependency stack (SDXL, ByteDance/
SDXL-Lightning) có license riêng cần kiểm tra.

### 1.2 InstantID (`InstantX/InstantID`, Apache 2.0)

**Components**:

- `antelopev2` face pack (5 ONNX file: `1k3d68.onnx`, `2d106det.onnx`,
  `genderage.onnx`, `glintr100.onnx`, `scrfd_10g_bnkps.onnx`) — ✅ tất
  cả ONNX, public trên HF, ~428 MB tổng.
  - Nhưng đây chỉ là InsightFace standard pack — facefusion đã dùng các
    module tương đương (yolo_face / retinaface / scrfd cho detection,
    arcface_w600k_r50 cho recognition).
- **IP-Adapter** (`ip-adapter.bin`) — ❌ PyTorch only, inject identity
  token vào SDXL cross-attention. Không có ONNX export.
- **ControlNetModel** (`diffusion_pytorch_model.safetensors`) — ❌
  PyTorch only, ControlNet receives 5 face landmark keypoints,
  conditioning SDXL.

**License**: Apache 2.0 cho code/weight chính. InstantX không enforce
non-commercial.

**Kết luận**: InstantID's only novel contribution beyond InsightFace =
IP-Adapter + ControlNet, cả hai đều cần SDXL. Phần ONNX có sẵn = phần
facefusion ĐÃ CÓ.

### 1.3 facefusion master (current state)

`facefusion/face_analyser.py:117-139`:

```python
def get_average_face(faces : List[Face]) -> Optional[Face]:
    # ... iterate faces ...
    return Face(
        # ... bounding_box/score_set/landmark_set lấy từ first_face ...
        embedding = numpy.mean(face_embeddings, axis = 0),
        embedding_norm = numpy.mean(face_embeddings_norm, axis = 0),
        # ...
    )
```

`facefusion/processors/modules/face_swapper/core.py:770-781`:

```python
def extract_source_face(source_vision_frames : List[VisionFrame]) -> Optional[Face]:
    source_faces = []
    if source_vision_frames:
        for source_vision_frame in source_vision_frames:
            temp_faces = get_many_faces([source_vision_frame])
            temp_faces = sort_faces_by_order(temp_faces, 'large-small')
            if temp_faces:
                source_faces.append(get_first(temp_faces))
    return get_average_face(source_faces)
```

→ **Multi-source averaging đã tồn tại** dưới dạng naive `numpy.mean`.
Không có:

- Weighting theo face quality (detector score, landmarker score, blur,
  occlusion).
- Outlier rejection (loại bỏ source bất thường — vd. 4 ảnh ID đúng + 1
  ảnh người khác lạc).
- Slerp (spherical interpolation) — đúng hơn về mặt hình học cho
  normalized embedding.
- Cross-encoder fusion (chỉ dùng ArcFace, không có AdaFace/MagFace).

Roadmap §1.4 đã ghi nhận điều này: "thêm weighted average + outlier
rejection".

---

## 2. Phương án đề xuất

### Phương án A — Identity Ensemble v1 (✅ slice 1, PR #23)

**Scope**: Cải tiến `extract_source_face` + `get_average_face` để hỗ
trợ multi-source fusion với 4 mode:

| Mode | Mô tả | Default? |
| --- | --- | --- |
| `mean` | Hiện tại — `numpy.mean` qua tất cả source | ✅ default (bit-equal master) |
| `weighted` | Weighted mean theo `detector × landmarker × resolution_factor` | opt-in |
| `slerp` | Spherical linear interpolation pairwise → đúng hơn cho normalized embedding | opt-in |
| `robust` | Drop outlier (cosine sim với mean < threshold) → weighted mean trên surviving | opt-in |

**CLI flag mới**:

- `--source-fusion-mode {mean,weighted,slerp,robust}` (default `mean`).
- `--source-fusion-outlier-threshold` (float, default 0.65, chỉ cho mode
  `robust`).

**Bit-equal guarantee**: mode `mean` = behavior cũ exact (tested via
stub).

**Effort**: M. ~470 line code add + 32 unit test (stub-based, không cần
ONNX). 1 PR.

**Quality lift**:

- `weighted`: nếu user pass nhiều source với chất lượng khác nhau (vd.
  1 ảnh ID rõ + 4 ảnh blur), mode này ưu tiên ảnh rõ → identity bền hơn.
- `robust`: nếu user vô tình pass nhầm 1 ảnh người khác, mode này tự
  động loại bỏ → tránh contamination.
- `slerp`: cho embedding đã normalize (norm=1 manifold), Slerp giảm hiện
  tượng "average về 0" khi 2 source có angle lớn.

**Validation trên CPU VM**: dùng synthetic embedding (random normalized
512-dim), test:

- Bit-equal vs `numpy.mean` ở mode default.
- Outlier rejection thực sự loại được synthetic outlier (cosine sim 0.1
  vs centroid).
- Weighted ưu tiên đúng score cao.
- Slerp midpoint của 2 ortho vector = đúng 45° (1/sqrt(2)).

### Phương án B — AdaFace cross-encoder ensemble (FUTURE, opt-in research)

**Scope**: Add AdaFace `adaface_ir101` ONNX (research-only flag) → 2 ID
vector per source face → cross-encoder fusion (mean-of-normalized).

**Effort**: M. Cần upload AdaFace ONNX lên HF mirror.

**Quality lift**: ArcFace + AdaFace cross-encoder thường tăng identity
robustness cho face khác chủng tộc / age / lighting (paper PuLID
specifically tested).

### Phương án C — Full PuLID/InstantID diffusion stack (DEFERRED, GPU-only)

**Scope**: Ship SDXL Lightning + IDFormer + ControlNet + IP-Adapter,
integrate vào processor mới `face_diffuser` hoặc `pulid_swap`.

**Effort**: L+. Nhiều PR. Bundle weight 4-12 GB. GPU bắt buộc (CPU sẽ
vô dụng — diffusion 25-50 step/frame). License bundling cần audit kỹ.

**Status**: Defer — sau khi A1 (`inswapper_512_live`) + A2 (LatentSync
sampler) + GPU CI sẵn sàng mới làm.

---

## 3. Quyết định đã chốt cho PR #23

User chọn defaults được recommend:

1. ✅ **Phương án A** — ship slice 1, defer B/C.
2. ✅ **Default mode = `mean`** — bit-equal master, an toàn rollback.
3. ✅ **Outlier threshold = 0.65** (cosine similarity).
4. ✅ **CLI flag = `--source-fusion-mode`** + `--source-fusion-outlier-threshold`.

PR #23 ship đầy đủ phương án A. Phương án B/C ghi vào roadmap, sẽ làm ở
PR follow-up sau khi user approve direction.

---

## 4. Tasks cụ thể của PR #23

✅ `facefusion/face_analyser.py`:

- `get_fused_face(faces, mode, outlier_threshold)` — generic fusion với 4 mode.
- `get_average_face` route qua `get_fused_face(faces, 'mean', None)` để giữ back-compat.
- Helper `_compute_face_quality_weights`, `_weighted_mean`, `_slerp_pair`,
  `_fuse_slerp`, `_reject_outliers`, `_fuse_robust`.

✅ `facefusion/processors/modules/face_swapper/core.py`:

- `register_args`: thêm `--source-fusion-mode` + `--source-fusion-outlier-threshold`.
- `apply_args`: apply state.
- `extract_source_face`: route qua `get_fused_face` với mode từ state.

✅ `facefusion/types.py`: thêm Literal `SourceFusionMode`.

✅ `facefusion/processors/modules/face_swapper/types.py`: register
`SourceFusionOutlierThreshold`.

✅ `facefusion/processors/modules/face_swapper/choices.py`:
`source_fusion_modes` + `source_fusion_outlier_threshold_range`.

✅ `facefusion/processors/modules/face_swapper/locales.py`: help text +
UI label.

✅ `facefusion.ini`: register 2 key config trống (default ở CLI side là
`mean` / `0.65`).

✅ `tests/test_identity_ensemble.py` (mới, 32 test):

- Quality weight (4): detector score / bbox area / landmarker disabled
  / all-zero fallback.
- `_weighted_mean` (2): explicit formula + uniform = arithmetic mean.
- `_slerp_pair` (4): t=0 / t=1 endpoint, unit-norm output, collinear
  handling.
- `_reject_outliers` (4): drops outlier / keeps consistent / keeps
  closest when all flagged / single face short-circuit.
- `get_fused_face` dispatch (10).
- `get_average_face` back-compat (2).
- `extract_source_face` plumbing (3): default mean, robust state, no-face → None.
- `register_args` + `apply_args` (2).
- Module sanity (1).

✅ `README.md`: thêm section "Identity ensemble (multi-source fusion)".

✅ `ULTRA_ROADMAP.md`: mark §1.4 (weighted average + outlier rejection)
✅ done với PR ref. Note PuLID/InstantID full diffusion stack vẫn
pending.

✅ `WF_AUDIT.md`: update PR inventory + roadmap progress; thêm section
3.12 cho PR #23.

**Không làm trong PR này**:

- ❌ AdaFace ONNX (phương án B).
- ❌ SDXL diffusion (phương án C).
- ❌ Thay đổi inswapper inference path — chỉ thay đổi cách tính source
  embedding.
