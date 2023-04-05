"""
Implements a tracker for corn crops using SORT algorithm.

SORT is a simple, online and realtime tracking algorithm for 2D multiple
object tracking in video sequences. It works with bounding boxes.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ts_semantic_feature_detector.features_3d.camera import StereoCamera
from ts_semantic_feature_detector.features_3d.crop import CornCrop
from ts_semantic_feature_detector.features_3d.sequence import AgriculturalSequence

class KalmanBoxTracker():
    """
    Implements a bounding box tracker with Kalman filter.

    Attributes:
        crops (:obj:`list`): a list of :obj:`features_3d.crop.CornCrop`
            that this tracker refers to.
    """

    count = 0
    """
    int: static attribute to give unique ID's to the trackers.
    """

    def __init__(
        self,
        crop: CornCrop
    ):
        """
        Initialize a box tracker.

        TODO: update documentation about the state and action vectors.

        Args:
            crop (:obj:`features_3d.crop.CornCrop`): the object containing
                information about a single corn crop.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=6, dim_u=2)

        # State transition matrix.
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Control transition matrix.
        self.kf.B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1],
            [0, 0]
        ])

        # Measurement function.
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ])

        # Process noise 
        self.kf.Q[:, :] *= 1e6

        # Measurement noise
        # Trust a lot position and displacement of matched detections.
        self.kf.R *= 10e-6
        # self.kf.R[:2, :2] *= 10e-3
        # self.kf.R[4:-1, 4:-1] *= 10e-3
        # Do not trust scale and ratio changes too much (original implementation).
        # self.kf.R[2:4,2:4] *= 10.
        # self.kf.R[-1, -1] *= 10.

        # Covariance matrix
        self.kf.P[-1, -1] *= 1000. # High uncertainty to scale velocity.
        self.kf.P *= 10.
        
        self.kf.x[:4] = self._convert_bbox_to_z(crop.crop_box.data)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Change crop cluster.
        self.crops = []
        self.crops.append(crop)
        self.crops[-1].cluster = self.id

    def _convert_bbox_to_z(
        self,
        bbox: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Converts a bounding box in format [x1, y1, x2, y2] into [x, y, s, r].

        The [x1, y1, x2, y2] is segmentation model output format for bounding
        boxes, where (x1, y1) describes the top-left point and (x2, y2) the
        bottom-right point. The [x, y, s, r] is the Kalman filter format, where
        (x, y) describes the bounding box center coordinates, s is the bounding
        box area and r is bounding box size ratio.

        Args:
            bbox: (:obj:`np.ndarray`): containing the crop bounding box.

        Returns:
            converted_data (:obj:`np.ndarray`): the bouding box data converted.
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)

        if len(bbox) > 4:
            return np.array([x, y, s, r, bbox[4], bbox[5]]).reshape((6, 1))
        else:
            return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(
        self,
        x: List,
        score: float = None
    ) -> npt.ArrayLike:
        """
        Converts a bounding box in format [x, y, s, r] into [x1, y1, x2, y2].

        The [x1, y1, x2, y2] is segmentation model output format for bounding
        boxes, where (x1, y1) describes the top-left point and (x2, y2) the
        bottom-right point. The [x, y, s, r] is the Kalman filter format, where
        (x, y) describes the bounding box center coordinates, s is the bounding
        box area and r is bounding box size ratio.

        Args:
            x (:obj:`list`): containing the bounding box data.
            score (float, optional): containing the detection score.

        Returns:
            converted_data (:obj:`np.ndarray`): the bouding box data converted.
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

    def predict(
        self,
        motion_2d_offset: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Executes the Kalman filter predict step.

        Args:
            motion_2d_offset: a :obj:`np.ndarray` containing the 2D motion offset
                calculated from the extrinsics information.

        Returns:
            prediction (:obj:`np.ndarray`): the predicted bounding box.
        """

        # Checks if the new predicted scale will be zero.
        # If yes, ignores the scale improvement.
        if((self.kf.x[4] + self.kf.x[2]) <= 0):
            self.kf.x[4] *= 0.0

        # Advances the state vector informing the motion offset
        # as the control action.
        self.kf.predict(motion_2d_offset)
        self.age += 1

        # Updates the time information
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(
            self._convert_x_to_bbox(self.kf.x)
        )

        return self.history[-1]
    
    def update(
        self,
        detection: npt.ArrayLike
    ) -> None:
        """
        Executes the Kalman filter correction step.

        Args:
            detection (:obj:`np.ndarray`): the detected bounding box
                in format [x1, y1, x2, y2].
        """

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(detection))

    def get_state(
        self
    ) -> npt.ArrayLike:
        """
        Get the current bounding box estimate.

        Returns:
            state (:obj:`np.ndarray`): the current bounding box estimate.
        """
        return self._convert_x_to_bbox(self.kf.x)

class AgricultureSort():
    """
    Modified SORT: A Simple, Online and Realtime Tracker.

    Attributes:
        camera (features_3d.camera.StereoCamera): the camera object that
            contains all the stereo camera information to project 3D points
            back into the 2D plane.
        max_age (int): the maximum number of frames to keep alive a track 
            without associated detections.
        min_hits (int): the minimum number of associated detections before
            track is initialised.
        iou_threshold (float): the minimum IOU for match.
        trackers (:obj:`list`): a list of :obj:`AgricultureTracker` objects.
        frame_count (int): the current frame number.
    """
    def __init__(
        self,
        camera: StereoCamera,
        max_age = 1,
        min_hits = 3,
        iou_threshold = 0.3
    ):
        """
        Initialize the SORT object.

        Args:
            camera (features_3d.camera.StereoCamera): the camera object that
                contains all stereo camera information to project 3D points 
                back into the 2D plane.
            max_age (int, optional): the maximum number of frames to keep alive 
                a track without associated detections.
            min_hits (int, optional): the minimum number of associated detections
                before track is initialised.
            iou_threshold (float, optional): the minimum IOU for match.
        """
        self.camera = camera
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []
        self.frame_count = 0

    def step(
        self,
        sequence: AgriculturalSequence
    ) -> npt.ArrayLike:
        """
        Executes the tracker step.

        Args:
            sequence (:obj:`features_3d.agriculture.AgriculturalSequence`): the sequence
                object that contains all the information about the scenes.

        Returns:
            tracked_bbox (:obj:`np.ndarray`): containing the tracked bounding boxes.
        """

        self.frame_count += 1

        # Update crops with the 2D motion offset from extrinsics
        self.get_crops_motion(sequence)

        # Load information about the Kalman filter prediction step of existing trackers.
        # Also checks if the prediction is valid. If not, saves the tracker index to
        # remove it later. 
        trackers_data = np.zeros((len(self.trackers), 5))
        to_delete_existing_trackers_idxs = []
        for t, tracker_data in enumerate(trackers_data):
            prediction = self.trackers[t].predict(
                self.trackers[t].crops[-1].estimated_motion_2d
            )[0]
            tracker_data[:] = [
                prediction[0], prediction[1], prediction[2], prediction[3], 0
            ]

            if np.any(np.isnan(prediction)):
                to_delete_existing_trackers_idxs.append(t)

        # Filter the extracted data from existing trackers.
        trackers_data = np.ma.compress_rows(np.ma.masked_invalid(trackers_data))

        # Filter the trackers that does not result in good
        for t in reversed(to_delete_existing_trackers_idxs):
            self.trackers.pop(t)

        detections = np.array([crop.crop_box.data for crop in sequence.scenes[-1].crop_group.crops])
        if not detections.any():
            detections = np.empty((0, 5))

        matched, unmatched_detections, unmatched_trackers = self._associate_detections_to_trackers(
            detections,
            trackers_data,
            self.iou_threshold
        )

        # For each founded correspondence, runs the Kalman filter correction step.
        # Also adds the current crop to the tracker.
        for m in matched:
            last_box = self.trackers[m[1]].crops[-1].crop_box.data
            current_box = detections[m[0], :]
            diff_2d = np.array(current_box[:2] - last_box[:2])
            
            observation = np.concatenate([current_box, diff_2d])
            self.trackers[m[1]].update(observation)

            # Adds the crop to the tracker. 
            self.trackers[m[1]].crops.append(sequence.scenes[-1].crop_group.crops[m[0]])
            self.trackers[m[1]].crops[-1].cluster = self.trackers[m[1]].id

        # Create and initialise new trackers for unmatched detections.
        for u in unmatched_detections:
            tracker = KalmanBoxTracker(
                sequence.scenes[-1].crop_group.crops[u]
            )
            self.trackers.append(tracker)

        # Filter the existing trackers by max_age and min_hits.
        ret = []
        t = len(self.trackers)
        for tracker in reversed(self.trackers):
            if tracker.time_since_update < 1:
                if (tracker.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits):
                    d = tracker.get_state()[0]

                    # # Modify the crop's clusters to match tracker ID.
                    # for crop in tracker.crops:
                    #     crop.cluster = tracker.id

                    ret.append(
                        np.concatenate(
                            (d, [tracker.id])
                        ).reshape(1,-1)
                    ) 
                t -= 1

                if tracker.time_since_update > self.max_age:
                    self.trackers.pop(t)

        if len(ret) > 0:
            return np.concatenate(ret)
        else:
            return np.empty((0, 5))
    
    def _iou_batch(
        self,
        bb_test: npt.ArrayLike,
        bb_gt: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Computes the IOU metric between two bounding boxes in the form [x1, y1, x2, y2].

        Args:
            bb_test (:obj:`np.ndarray`): the first bounding box.
            bb_gt (:obj:`np.ndarray`): the second bounding box.

        Returns:
            iou_values (:obj:`np.ndarray`): containing the IOU metric from each 
                tracker/detection pair.
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
        
        return(o)
    
    def _linear_assignment(
        self,
        cost_matrix: npt.ArrayLike
    ) -> npt.ArrayLike:          
        """
        Calculates the best detection/trackers correspondence.
        
        Args:
            cost_matrix (:obj:`np.ndarray`) containing negative IOU values
                for each detection and tracker pair.

        Returns:
            matches (:obj:`np.ndarray`): each line indicates a correspondance 
                between the first column (detection) and the second one (tracker).
        """
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

    def _associate_detections_to_trackers(
        self,
        detections: npt.ArrayLike,
        trackers_data: npt.ArrayLike,
        iou_threshold: float
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Associates the detections to the existing trackers.

        Args:
            detections (:obj:`np.ndarray`): the bounding boxes detected in
                the current frame.
            trackers_data (:obj:`np.ndarray`): the existing bounding boxes
                from the existing trackers.
            iou_threshold (float): the minimum IOU for match.

        Returns:
            matches (:obj:`np.ndarray`): the matches between detections and trackers.
            unmatched_detections (:obj:`np.ndarray`): the unmatched detections.
            unmatched trackers (:obj:`np.ndarray`): the unmatched trackers.
        """

        # If there is not any tracker yet, just return all detections
        # as unmatched ones.
        if len(trackers_data) == 0:
            matched = np.empty((0, 2), dtype=int)
            unmatched_detections = np.arange(len(detections))
            unmatched_trackers = np.empty((0, 5), dtype=int)
            return matched, unmatched_detections, unmatched_trackers
        
        # If there are already some trackers, check their IOU metric
        # with the provided detections
        iou_matrix = self._iou_batch(detections, trackers_data)

        matched_idxs = None
        # Checks if there are detections and trackers overlapping
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)

            # Checks if there is only one good detection and tracker correspondence.
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_idxs = np.stack(np.where(a), axis=1)
            else:
                # If not, try to find the better correspondence with Jonker-Volgenant algorithm.
                matched_idxs = self._linear_assignment(-iou_matrix)
        else:
            # If not, just return a empty Numpy array.
            matched_idxs = np.empty((0, 2))

        # Finds the the detections that don't have a tracker correspondence.
        unmatched_detections = []
        for d, detection in enumerate(detections):
            if d not in matched_idxs[:, 0]:
                unmatched_detections.append(d)

        # Finds the the trackers that don't have a detection correspondence.
        unmatched_trackers = []
        for t, tracker in enumerate(trackers_data):
            if t not in matched_idxs[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_idxs:
            #Filters the matches that have low IOU (when linear_assignment was used)
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def get_crops_motion(
        self,
        sequence: AgriculturalSequence
    ) -> None:
        """
        Calculates a prediction of the crop's position using extrinsics information.

        Args:
            sequence (:obj:`features_3d.sequence.AgriculturalSequence`): object that
                contains all the crop information during several scenes.
        """
        if len(sequence.scenes) > 1:
            current_scene = sequence.scenes[-1]
            prev_scene = sequence.scenes[-2]

            # Get the transformation between the two consecutive scenes and converts it back to origin frame.
            shift_transform = np.linalg.inv(current_scene.extrinsics) @ current_scene.extrinsics @ np.linalg.inv(prev_scene.extrinsics)

            # print('self.camera.size = ', self.camera.size)

            # For each crop in the previous scene, apply the transformation in the average point.
            # Calculates the difference between the previous average point with the transformed one.
            # Project this difference back into the 2D image plane.
            for prev_crop in prev_scene.crop_group.crops:
                # Apply the shift to the average point
                shifted_avg_point = shift_transform @ np.append(prev_crop.average_point, 1)
                shifted_2d_point = self.camera.get_2d_point(shifted_avg_point)

                # print('prev_crop.average_point = ', prev_crop.average_point)
                # print('shifted_avg_point = ', shifted_avg_point)
                # print('shifted_2d_point = ', shifted_2d_point)
                
                prev_box = prev_crop.crop_box.data
                box_center_x = np.average([prev_box[0], prev_box[2]])
                box_center_y = np.average([prev_box[1], prev_box[3]])
                box_center = np.array([box_center_x, box_center_y])

                # print('box_center = ', box_center)

                diff_2d = shifted_2d_point - box_center
                prev_crop.estimated_motion_2d = diff_2d
                # print('diff_2d = ', diff_2d)

                # Correcting direction of the movement.
                # prev_box = prev_crop.crop_box.data
                # if box_center[0] > self.camera.size[0]/2:
                #     prev_crop.estimated_motion_2d[0] *= -1.

                # Forces the array to be 2D to avoid problems when doing matrix multiplication.
                prev_crop.estimated_motion_2d = prev_crop.estimated_motion_2d[:, None]
