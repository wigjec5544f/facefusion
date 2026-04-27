from facefusion.types import Locales

LOCALES : Locales =\
{
	'en':
	{
		'help':
		{
			'model': 'choose the model responsible for synthesising in-between frames',
			'target_fps': 'when set, run RIFE frame interpolation on the rendered video to reach this output fps (e.g. 60); leave unset to skip',
			'multiplier': 'force the interpolation multiplier (overrides target_fps when both are set)'
		},
		'uis':
		{
			'model_dropdown': 'FRAME INTERPOLATOR MODEL',
			'target_fps_slider': 'FRAME INTERPOLATOR TARGET FPS',
			'multiplier_slider': 'FRAME INTERPOLATOR MULTIPLIER'
		}
	}
}
