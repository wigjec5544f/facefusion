from typing import List, get_args

from facefusion.processors.modules.frame_interpolator.types import FrameInterpolatorModel

frame_interpolator_models : List[FrameInterpolatorModel] = list(get_args(FrameInterpolatorModel))

# Hard-cap interpolation multiplier so we don't accidentally request 100x interpolation; RIFE
# is recursive so 8x already produces 7 synthetic frames per pair.
frame_interpolator_multiplier_range : List[int] = list(range(2, 9))

# Reasonable user-facing target FPS values; the actual target may be any float, this list is
# only used by the UI dropdown / metavar display.
frame_interpolator_target_fps_range : List[int] = [ 24, 25, 30, 48, 50, 60, 90, 120 ]
