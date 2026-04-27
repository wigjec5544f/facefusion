from typing import Literal, TypedDict

from facefusion.types import Mask, VisionFrame

FrameInterpolatorInputs = TypedDict('FrameInterpolatorInputs',
{
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

FrameInterpolatorModel = Literal['rife_4_9']
