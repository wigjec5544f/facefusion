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
install.bat              :: cài Python (qua winget nếu thiếu), ffmpeg, venv, deps, model
install_modules.bat      :: chỉ cài Python modules + onnxruntime vào venv hiện tại
run.bat                  :: activate venv và chạy app, mặc định preset HIGH QUALITY
run.bat --quality fast   :: chạy với config mặc định (nhanh hơn)
run.bat headless-run -s src.jpg -t tgt.mp4 -o out.mp4
```

`install.bat` tự động phát hiện GPU NVIDIA và chọn `cuda`, ngược lại fallback `directml`. Có thể ép biến thể: `install.bat cuda | directml | openvino | qnn | rocm | migraphx | default`.


High quality preset
-------------------

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

Đánh đổi: chậm hơn ~3–5× và VRAM ~1.5–2× so với preset mặc định. Nếu thiếu VRAM, đổi `[memory] video_memory_strategy = moderate` (hoặc `strict`), hoặc giảm `face_swapper_pixel_boost` xuống `512x512`.


Documentation
-------------

Read the [documentation](https://docs.facefusion.io) for a deep dive.
