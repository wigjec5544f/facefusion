from typing import List, Sequence, get_args

from facefusion.common_helper import create_int_range
from facefusion.processors.modules.portrait_animator.types import PortraitAnimatorModel

portrait_animator_models : List[PortraitAnimatorModel] = list(get_args(PortraitAnimatorModel))

portrait_animator_pose_weight_range : Sequence[int] = create_int_range(0, 100, 1)

portrait_animator_expression_weight_range : Sequence[int] = create_int_range(0, 100, 1)
