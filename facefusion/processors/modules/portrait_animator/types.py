from typing import List, Literal, TypedDict

from facefusion.types import Mask, VisionFrame

PortraitAnimatorInputs = TypedDict('PortraitAnimatorInputs',
{
	'reference_vision_frame' : VisionFrame,
	'source_vision_frames' : List[VisionFrame],
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

PortraitAnimatorModel = Literal['live_portrait']
