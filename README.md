FaceFusion
==========

> Industry leading face manipulation platform.

[![Build Status](https://img.shields.io/github/actions/workflow/status/facefusion/facefusion/ci.yml.svg?branch=master)](https://github.com/facefusion/facefusion/actions?query=workflow:ci)
[![Coverage Status](https://img.shields.io/coveralls/facefusion/facefusion.svg)](https://coveralls.io/r/facefusion/facefusion)
![License](https://img.shields.io/badge/license-OpenRAIL--AS-green)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion/master/.github/preview.png?sanitize=true)


Installation
------------

Be aware, the [installation](https://docs.facefusion.io/installation) needs technical skills and is not recommended for beginners. In case you are not comfortable using a terminal, our [Windows Installer](http://windows-installer.facefusion.io) and [macOS Installer](http://macos-installer.facefusion.io) get you started.


Usage
-----

Run the command:

```
python facefusion.py [commands] [options]

options:
  -h, --help                                      show this help message and exit
  -v, --version                                   show program's version number and exit

commands:
    run                                           run the program
    headless-run                                  run the program in headless mode
    batch-run                                     run the program in batch mode
    force-download                                force automate downloads and exit
    benchmark                                     benchmark the program
    job-list                                      list jobs by status
    job-create                                    create a drafted job
    job-submit                                    submit a drafted job to become a queued job
    job-submit-all                                submit all drafted jobs to become a queued jobs
    job-delete                                    delete a drafted, queued, failed or completed job
    job-delete-all                                delete all drafted, queued, failed and completed jobs
    job-add-step                                  add a step to a drafted job
    job-remix-step                                remix a previous step from a drafted job
    job-insert-step                               insert a step to a drafted job
    job-remove-step                               remove a step from a drafted job
    job-run                                       run a queued job
    job-run-all                                   run all queued jobs
    job-retry                                     retry a failed job
    job-retry-all                                 retry all failed jobs
```


Windows quick-start
-------------------

For Windows users a one-shot bootstrap is included:

```
install.bat                :: cài Python (qua winget nếu thiếu), ffmpeg, venv, deps, model
install_modules.bat        :: chỉ cài Python modules + onnxruntime vào venv hiện tại
run.bat                    :: activate venv và chạy app, mặc định preset HIGH QUALITY
run.bat --quality balanced :: clip ngắn, ~6-8 GB VRAM
run.bat --quality fast     :: real-time / preview, máy yếu
run.bat --quality default  :: dùng facefusion.ini gốc
run.bat headless-run -s src.jpg -t tgt.mp4 -o out.mp4
```

`install.bat` tự động phát hiện GPU NVIDIA và chọn `cuda`, ngược lại fallback `directml`. Có thể ép biến thể: `install.bat cuda | directml | openvino | qnn | rocm | migraphx | default`.


Quality presets
---------------

Bốn preset config có sẵn, chọn qua `run.bat --quality <name>` hoặc `--config-path facefusion.<name>.ini`:

| Preset    | File                           | VRAM     | Use case                            |
|-----------|--------------------------------|----------|-------------------------------------|
| `high`    | `facefusion.high-quality.ini`  | 12+ GB   | output chất lượng cao nhất (mặc định) |
| `balanced`| `facefusion.balanced.ini`      | 6-8 GB   | clip ngắn, GPU tầm trung            |
| `fast`    | `facefusion.fast.ini`          | <4 GB    | real-time / preview / iGPU          |
| `default` | `facefusion.ini`               | -        | config gốc của repo                  |

`facefusion.high-quality.ini` là preset đã tinh chỉnh để output ảnh/video chân thực và sắc nét nhất:

- detector `many` 640×640, landmarker `many`, score 0.5 — bắt face nghiêng/nhỏ tốt hơn.
- mask = `box + occlusion (xseg_3) + region (bisenet_resnet_34)` — chống lộ tay/kính, giữ tóc/mắt/miệng.
- `face_swapper_model = hyperswap_1c_256` với `pixel_boost = 1024x1024` — model chạy trên crop 1024 thay vì 256.
- `expression_restorer = live_portrait` factor 80 (upper + lower face) — giữ biểu cảm tự nhiên của target.
- `face_enhancer = gpen_bfr_2048` blend 80 — nâng chi tiết da/mắt mà không bị "plastic skin".
- `frame_enhancer = clear_reality_x4` blend 50 — khử noise toàn frame, giữ texture tự nhiên.
- output: `temp_frame_format = png` (lossless), `libx264 veryslow` quality 95, audio `flac`.

Cách dùng:

```
python facefusion.py run --config-path facefusion.high-quality.ini
python facefusion.py headless-run --config-path facefusion.high-quality.ini -s src.jpg -t tgt.mp4 -o out.mp4
```

Đánh đổi: chậm hơn ~3–5× và VRAM ~1.5–2× so với preset mặc định. Nếu thiếu VRAM, đổi `[memory] video_memory_strategy = moderate` (hoặc `strict`), hoặc giảm `face_swapper_pixel_boost` xuống `512x512`, hoặc dùng `run.bat --quality balanced`.


Environment doctor
------------------

Kiểm tra môi trường (Python, ffmpeg, onnxruntime providers, đường dẫn ghi được, RAM, dung lượng đĩa):

```
python facefusion.py doctor
```

In ra bảng status (`ok` / `warn` / `fail`) và exit code 0 nếu tất cả checks pass, 1 nếu có check fail. Hữu ích sau `install.bat` để xác nhận bootstrap đã đầy đủ.


Custom model mirror
-------------------

Mặc định FaceFusion tải model từ `facefusion/facefusion-assets` (GitHub Releases) và `facefusion/*` (HuggingFace). Nếu bạn muốn dùng mirror riêng (vd: HuggingFace org của chính bạn để host weight bổ sung), set environment variable trước khi chạy:

| Variable | Áp dụng cho provider | Ví dụ |
| --- | --- | --- |
| `FACEFUSION_HF_NAMESPACE` | `huggingface` | `wigjec5544f` (resolve sang `https://huggingface.co/wigjec5544f/<base_name>/resolve/main/<file>`) |
| `FACEFUSION_GH_NAMESPACE` | `github` | `wigjec5544f/facefusion-assets` (resolve sang `https://github.com/wigjec5544f/facefusion-assets/releases/download/<base_name>/<file>`) |

Nếu mirror của bạn vẫn dùng đúng layout repo và đúng file hash như upstream, các processor sẽ download bình thường. Nếu hash không khớp, hash check sẽ fail — bạn cần publish hash file đúng từ weight gốc.

Mirror chỉ thay namespace; URL gốc (`huggingface.co`, `github.com`) không đổi. Để override hoàn toàn URL gốc, sửa `facefusion/choices.py::download_provider_set`.


Publishing weights to your mirror
---------------------------------

`tools/hf_publish.py` upload một file weight + sinh `.hash` (CRC32, format giống `facefusion.hash_helper`) lên HF repo của bạn. Hữu ích để chuẩn bị weight cho các processor mới của Đợt 3-6:

```
export HF_TOKEN=hf_...
python tools/hf_publish.py \
  --source /path/to/rife_4_26.onnx \
  --repo-id ngoqquyen/facefusion-extras \
  --dest frame_interpolator/rife_4_26.onnx
```

Cờ `--hash-only` tính hash mà không upload (để kiểm tra cục bộ). Sau khi upload, kết hợp với `FACEFUSION_HF_NAMESPACE` (xem section trên) để facefusion download về từ mirror của bạn.

Đã upload mẫu: [`ngoqquyen/facefusion-extras/frame_interpolator/rife_4_9.onnx`](https://huggingface.co/ngoqquyen/facefusion-extras/tree/main/frame_interpolator) (RIFE 4.9, MIT, ~21 MB) — dùng cho module `facefusion.frame_interpolator` (Đợt 1.A3).


Frame interpolator (RIFE)
-------------------------

`facefusion.frame_interpolator` cung cấp inference primitive cho RIFE 4.x: lấy 2 frame liên tiếp + timestep ∈ [0,1] → trả về frame trung gian. Dùng để tăng fps (vd: 30 → 60).

```python
from facefusion import frame_interpolator
mid_frame = frame_interpolator.interpolate_pair(prev_frame, next_frame, timestep = 0.5)
```

Đầu vào / đầu ra là `VisionFrame` (HxWx3 BGR uint8) — cùng format với phần còn lại của facefusion. Module tự download `rife_4_9.onnx` (21 MB) lần đầu chạy từ HF mirror (`ngoqquyen/facefusion-extras` mặc định, override bằng `FACEFUSION_HF_NAMESPACE` + `FACEFUSION_EXTRAS_REPO`).

Pipeline integration (chạy interpolation lên cả video output) sẽ ship trong PR riêng — primitive này là building block.

Cho video CLI standalone (chain sau `headless-run`):

```
python tools/interpolate_video.py \
  --input swap.mp4 \
  --output swap_60fps.mp4 \
  --multiplier 2          # 30->60 fps; dùng 3 cho 30->90, 4 cho 30->120
```

Tool đọc video qua ffmpeg pipe (không extract ra disk), gọi `interpolate_pair` cho mỗi cặp frame liên tiếp, ghi thẳng ra encoder. CRF mặc định 18, codec mặc định `libx264`. Override execution provider bằng `--execution-provider cuda` (lặp lại để chain `cuda → cpu`). Lần đầu chạy sẽ download `rife_4_9.onnx` ~21 MB từ HF mirror.


Optional Python extras
----------------------

`pyproject.toml` khai báo extras để chuẩn bị cho các Đợt sau:

```
pip install -e .[dev]         # flake8 + pytest cho contributor
pip install -e .[diffusion]   # diffusers + transformers + torch (Đợt 2)
pip install -e .[api]         # httpx + pydantic (Đợt 4-5, motion control / video synth API)
```

Bootstrap chính (`install.bat` / `install.py`) vẫn là path duy nhất được test cho user cuối — extras chủ yếu để dev và CI dùng.


Documentation
-------------

Read the [documentation](https://docs.facefusion.io) for a deep dive.
