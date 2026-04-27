from facefusion.types import Locales

LOCALES : Locales =\
{
	'en':
	{
		'help':
		{
			'model': 'choose the model responsible for animating the source portrait with motion from the driving target',
			'pose_weight': 'specify how strongly the target driving pose (head pitch, yaw, roll) overrides the source portrait pose; 0 keeps the source pose, 100 fully follows the driving video',
			'expression_weight': 'specify how strongly the target driving expression replaces the source portrait expression; 0 keeps the source expression, 100 fully follows the driving video'
		},
		'uis':
		{
			'model_dropdown': 'PORTRAIT ANIMATOR MODEL',
			'pose_weight_slider': 'PORTRAIT ANIMATOR POSE WEIGHT',
			'expression_weight_slider': 'PORTRAIT ANIMATOR EXPRESSION WEIGHT'
		}
	}
}
