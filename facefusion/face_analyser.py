from typing import List, Optional, Tuple

import numpy

from facefusion import state_manager
from facefusion.common_helper import get_first
from facefusion.face_classifier import classify_faces
from facefusion.face_detector import detect_faces, detect_faces_by_angle
from facefusion.face_helper import apply_nms, convert_to_face_landmark_5, estimate_face_angle, get_nms_threshold
from facefusion.face_landmarker import detect_face_landmarks_batch, estimate_face_landmark_68_5_batch
from facefusion.face_recognizer import calculate_face_embeddings
from facefusion.face_store import get_static_faces, set_static_faces
from facefusion.types import BoundingBox, Embedding, Face, FaceLandmark5, FaceLandmarkSet, FaceScoreSet, Score, SourceFusionMode, VisionFrame


def create_faces(vision_frame : VisionFrame, bounding_boxes : List[BoundingBox], face_scores : List[Score], face_landmarks_5 : List[FaceLandmark5]) -> List[Face]:
	faces = []
	nms_threshold = get_nms_threshold(state_manager.get_item('face_detector_model'), state_manager.get_item('face_detector_angles'))
	keep_indices = apply_nms(bounding_boxes, face_scores, state_manager.get_item('face_detector_score'), nms_threshold)

	# Phase 0 -- batch every kept face's 5->68 landmark expansion through
	# the `fan_68_5` ONNX model in one call. With a dynamic-batch model
	# this collapses N session.run() into 1; otherwise the helper falls
	# back to N sequential calls (bit-equal output guaranteed).
	kept_face_landmarks_5 = [ face_landmarks_5[index] for index in keep_indices ]
	kept_face_landmarks_68_5 = estimate_face_landmark_68_5_batch(kept_face_landmarks_5)

	# Phase 0b -- batch the (optional) 2dfan4 / peppa_wutz refinement
	# pass for every kept face. We pre-compute the per-face angles (a
	# pure-CPU dependency on the fan_68_5 batch above) then dispatch one
	# batched ONNX call per active refinement model. Stock fixed-batch
	# models fall back to the per-face loop and stay bit-equal; user
	# re-exports with a dynamic batch axis collapse N session.run into
	# one per refinement model.
	face_landmarker_score = state_manager.get_item('face_landmarker_score')
	kept_bounding_boxes = [ bounding_boxes[index] for index in keep_indices ]
	kept_face_angles = [ estimate_face_angle(face_landmark_68_5) for face_landmark_68_5 in kept_face_landmarks_68_5 ]
	if face_landmarker_score > 0:
		kept_refined_landmarks = detect_face_landmarks_batch(vision_frame, kept_bounding_boxes, kept_face_angles)
	else:
		kept_refined_landmarks = [ (None, 0.0) ] * len(keep_indices)

	# Phase 1 -- gather every kept face's per-face state (bounding box,
	# score, refined landmarks) without touching the ArcFace ONNX model
	# yet. This lets us run the recogniser exactly once for the whole
	# frame in phase 2 instead of once per face.
	face_records = []

	for index_position, index in enumerate(keep_indices):
		bounding_box = bounding_boxes[index]
		face_score = face_scores[index]
		face_landmark_5 = face_landmarks_5[index]
		face_landmark_5_68 = face_landmark_5
		face_landmark_68_5 = kept_face_landmarks_68_5[index_position]
		face_landmark_68 = face_landmark_68_5
		face_landmark_score_68 = 0.0
		face_angle = kept_face_angles[index_position]

		if face_landmarker_score > 0:
			refined_landmark_68, refined_score_68 = kept_refined_landmarks[index_position]
			face_landmark_68 = refined_landmark_68
			face_landmark_score_68 = refined_score_68
		if face_landmark_score_68 > face_landmarker_score:
			face_landmark_5_68 = convert_to_face_landmark_5(face_landmark_68)

		face_landmark_set : FaceLandmarkSet =\
		{
			'5': face_landmark_5,
			'5/68': face_landmark_5_68,
			'68': face_landmark_68,
			'68/5': face_landmark_68_5
		}
		face_score_set : FaceScoreSet =\
		{
			'detector': face_score,
			'landmarker': face_landmark_score_68
		}
		face_records.append((bounding_box, face_score_set, face_landmark_set, face_angle))

	# Phase 2 -- compute every kept face's embedding in a single batched
	# ArcFace call. When the ONNX model exposes a dynamic batch axis this
	# collapses N session.run() calls into 1; otherwise it falls back to
	# the original per-face loop (bit-equal output guaranteed).
	face_embedding_landmarks = [ record[2].get('5/68') for record in face_records ]
	face_embeddings = calculate_face_embeddings(vision_frame, face_embedding_landmarks)

	# Phase 3 -- classify (gender, age, race) for every kept face in a
	# single batched fairface call. Uses the same dynamic-batch helper
	# as ArcFace (PR #16) and the landmarker batches (PR #17 / #18); for
	# stock fixed-batch models the helper falls back to the per-face
	# loop and the output stays bit-equal.
	face_classifications = classify_faces(vision_frame, face_embedding_landmarks)

	for record, (face_embedding, face_embedding_norm), (gender, age, race) in zip(face_records, face_embeddings, face_classifications):
		bounding_box, face_score_set, face_landmark_set, face_angle = record
		faces.append(Face(
			bounding_box = bounding_box,
			score_set = face_score_set,
			landmark_set = face_landmark_set,
			angle = face_angle,
			embedding = face_embedding,
			embedding_norm = face_embedding_norm,
			gender = gender,
			age = age,
			race = race
		))
	return faces


def get_one_face(faces : List[Face], position : int = 0) -> Optional[Face]:
	if faces:
		position = min(position, len(faces) - 1)
		return faces[position]
	return None


def get_average_face(faces : List[Face]) -> Optional[Face]:
	# Backwards-compatible entrypoint -- routes through `get_fused_face`
	# in the bit-equal `mean` mode so existing callers (and the historical
	# numpy.mean output) are preserved exactly.
	return get_fused_face(faces, 'mean', None)


def get_fused_face(faces : List[Face], mode : SourceFusionMode = 'mean', outlier_threshold : Optional[float] = None) -> Optional[Face]:
	if not faces:
		return None
	first_face = get_first(faces)

	if mode == 'mean' or len(faces) == 1:
		face_embeddings = [ face.embedding for face in faces ]
		face_embeddings_norm = [ face.embedding_norm for face in faces ]
		embedding = numpy.mean(face_embeddings, axis = 0)
		embedding_norm = numpy.mean(face_embeddings_norm, axis = 0)
	elif mode == 'weighted':
		embedding, embedding_norm = _fuse_weighted(faces)
	elif mode == 'slerp':
		embedding, embedding_norm = _fuse_slerp(faces)
	elif mode == 'robust':
		embedding, embedding_norm = _fuse_robust(faces, outlier_threshold)
	else:
		raise ValueError('unknown source fusion mode: ' + str(mode))

	return Face(
		bounding_box = first_face.bounding_box,
		score_set = first_face.score_set,
		landmark_set = first_face.landmark_set,
		angle = first_face.angle,
		embedding = embedding,
		embedding_norm = embedding_norm,
		gender = first_face.gender,
		age = first_face.age,
		race = first_face.race
	)


def _compute_face_quality_weights(faces : List[Face]) -> 'numpy.ndarray':
	# Per-face quality weight = detector score x landmarker factor x
	# (1 + sqrt(bbox area) / 256). The bbox term rewards source faces
	# that occupy more pixels (more identity signal) without dominating
	# when bboxes are similar in size. When the landmarker is disabled
	# (`face_landmarker_score == 0`) every face has score 0; we apply a
	# constant 0.1 floor so the weight reduces to detector x area.
	weights = []
	for face in faces:
		detector_score = max(float(face.score_set.get('detector', 0.0)), 0.0)
		landmarker_score = max(float(face.score_set.get('landmarker', 0.0)), 0.0)
		landmarker_factor = landmarker_score if landmarker_score > 0.0 else 0.1
		bbox_w = max(float(face.bounding_box[2] - face.bounding_box[0]), 0.0)
		bbox_h = max(float(face.bounding_box[3] - face.bounding_box[1]), 0.0)
		area_factor = 1.0 + numpy.sqrt(bbox_w * bbox_h) / 256.0
		weights.append(detector_score * landmarker_factor * area_factor)
	weights_arr = numpy.array(weights, dtype = numpy.float64)
	if not numpy.any(weights_arr > 0.0):
		# All-zero weights (e.g. detector score absent in a stub) -> uniform.
		weights_arr = numpy.ones(len(faces), dtype = numpy.float64)
	return weights_arr


def _weighted_mean(values : List['numpy.ndarray'], weights : 'numpy.ndarray') -> 'numpy.ndarray':
	stacked = numpy.stack(values, axis = 0)
	normalised = weights / weights.sum()
	return numpy.tensordot(normalised, stacked, axes = 1)


def _fuse_weighted(faces : List[Face]) -> Tuple[Embedding, Embedding]:
	weights = _compute_face_quality_weights(faces)
	face_embeddings = [ face.embedding for face in faces ]
	face_embeddings_norm = [ face.embedding_norm for face in faces ]
	return _weighted_mean(face_embeddings, weights), _weighted_mean(face_embeddings_norm, weights)


def _slerp_pair(a : 'numpy.ndarray', b : 'numpy.ndarray', t : float) -> 'numpy.ndarray':
	# Spherical linear interpolation between two vectors at param t in
	# [0, 1]. Inputs are renormalised first, output is a unit vector.
	a_unit = a / max(float(numpy.linalg.norm(a)), 1e-12)
	b_unit = b / max(float(numpy.linalg.norm(b)), 1e-12)
	dot = float(numpy.clip(numpy.dot(a_unit, b_unit), -1.0, 1.0))
	if dot > 0.9995:
		# Vectors near-collinear -- linear interp + renormalise avoids the
		# 1/sin(omega) singularity.
		result = (1.0 - t) * a_unit + t * b_unit
		return result / max(float(numpy.linalg.norm(result)), 1e-12)
	omega = numpy.arccos(dot)
	sin_omega = numpy.sin(omega)
	return (numpy.sin((1.0 - t) * omega) / sin_omega) * a_unit + (numpy.sin(t * omega) / sin_omega) * b_unit


def _fuse_slerp(faces : List[Face]) -> Tuple[Embedding, Embedding]:
	# Sequentially slerp normalised embeddings with cumulative parameter
	# t = 1/(i+1) so the final vector is the spherical centroid of all
	# sources. Output `embedding_norm` is a unit vector; raw `embedding`
	# is the unit direction scaled by the mean magnitude of source raws
	# so downstream consumers that rescale (e.g. inswapper) keep working.
	face_embeddings_norm = [ face.embedding_norm for face in faces ]
	embedding_norm = face_embeddings_norm[0] / max(float(numpy.linalg.norm(face_embeddings_norm[0])), 1e-12)
	for index in range(1, len(face_embeddings_norm)):
		embedding_norm = _slerp_pair(embedding_norm, face_embeddings_norm[index], 1.0 / (index + 1))
	magnitudes = [ float(numpy.linalg.norm(face.embedding)) for face in faces ]
	mean_magnitude = float(numpy.mean(magnitudes))
	embedding = embedding_norm * mean_magnitude
	return embedding, embedding_norm


def _reject_outliers(faces : List[Face], threshold : float) -> List[Face]:
	if len(faces) <= 1:
		return list(faces)
	face_embeddings_norm = numpy.stack([ face.embedding_norm for face in faces ], axis = 0)
	centroid = numpy.mean(face_embeddings_norm, axis = 0)
	centroid_unit = centroid / max(float(numpy.linalg.norm(centroid)), 1e-12)
	similarities = []
	for face in faces:
		face_norm = face.embedding_norm
		face_unit = face_norm / max(float(numpy.linalg.norm(face_norm)), 1e-12)
		similarities.append(float(numpy.dot(face_unit, centroid_unit)))
	similarities_arr = numpy.array(similarities, dtype = numpy.float64)
	keep_mask = similarities_arr >= threshold
	if not numpy.any(keep_mask):
		# All faces flagged as outliers (vs. their own centroid) -- keep
		# only the single face closest to the centroid so the caller still
		# gets a usable identity instead of `None`.
		closest_index = int(numpy.argmax(similarities_arr))
		return [ faces[closest_index] ]
	return [ face for face, keep in zip(faces, keep_mask) if bool(keep) ]


def _fuse_robust(faces : List[Face], outlier_threshold : Optional[float]) -> Tuple[Embedding, Embedding]:
	threshold = 0.65 if outlier_threshold is None else float(outlier_threshold)
	survivors = _reject_outliers(faces, threshold)
	return _fuse_weighted(survivors)


def get_many_faces(vision_frames : List[VisionFrame]) -> List[Face]:
	many_faces : List[Face] = []

	for vision_frame in vision_frames:
		if numpy.any(vision_frame):
			static_faces = get_static_faces(vision_frame)
			if static_faces:
				many_faces.extend(static_faces)
			else:
				all_bounding_boxes = []
				all_face_scores = []
				all_face_landmarks_5 = []

				for face_detector_angle in state_manager.get_item('face_detector_angles'):
					if face_detector_angle == 0:
						bounding_boxes, face_scores, face_landmarks_5 = detect_faces(vision_frame)
					else:
						bounding_boxes, face_scores, face_landmarks_5 = detect_faces_by_angle(vision_frame, face_detector_angle)
					all_bounding_boxes.extend(bounding_boxes)
					all_face_scores.extend(face_scores)
					all_face_landmarks_5.extend(face_landmarks_5)

				if all_bounding_boxes and all_face_scores and all_face_landmarks_5 and state_manager.get_item('face_detector_score') > 0:
					faces = create_faces(vision_frame, all_bounding_boxes, all_face_scores, all_face_landmarks_5)

					if faces:
						many_faces.extend(faces)
						set_static_faces(vision_frame, faces)
	return many_faces


def scale_face(target_face : Face, target_vision_frame : VisionFrame, temp_vision_frame : VisionFrame) -> Face:
	scale_x = temp_vision_frame.shape[1] / target_vision_frame.shape[1]
	scale_y = temp_vision_frame.shape[0] / target_vision_frame.shape[0]

	bounding_box = target_face.bounding_box * [ scale_x, scale_y, scale_x, scale_y ]
	landmark_set =\
	{
		'5': target_face.landmark_set.get('5') * numpy.array([ scale_x, scale_y ]),
		'5/68': target_face.landmark_set.get('5/68') * numpy.array([ scale_x, scale_y ]),
		'68': target_face.landmark_set.get('68') * numpy.array([ scale_x, scale_y ]),
		'68/5': target_face.landmark_set.get('68/5') * numpy.array([ scale_x, scale_y ])
	}

	return target_face._replace(
		bounding_box = bounding_box,
		landmark_set = landmark_set
	)
