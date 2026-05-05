from facefusion.types import Locales

LOCALES : Locales =\
{
	'en':
	{
		'help':
		{
			'model': 'choose the model responsible for swapping the face',
			'pixel_boost': 'choose the pixel boost resolution for the face swapper',
			'weight': 'specify the degree of weight applied to the face',
			'source_fusion_mode': 'choose the strategy used to fuse multiple source faces into one identity (mean keeps the legacy behaviour)',
			'source_fusion_outlier_threshold': 'cosine similarity threshold below which a source face is rejected as an outlier (only applies to the robust mode)'
		},
		'uis':
		{
			'model_dropdown': 'FACE SWAPPER MODEL',
			'pixel_boost_dropdown': 'FACE SWAPPER PIXEL BOOST',
			'weight_slider': 'FACE SWAPPER WEIGHT',
			'source_fusion_mode_dropdown': 'SOURCE FUSION MODE',
			'source_fusion_outlier_threshold_slider': 'SOURCE FUSION OUTLIER THRESHOLD'
		}
	}
}
