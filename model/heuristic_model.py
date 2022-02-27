from .basemodel import BaseModel
import numpy as np

class HeuristicModel(BaseModel):
    '''Heuristic model that serves as the baseline for other CNN models.'''
    def __init__(self, config, **kwargs) -> None:
        super().__init__(**kwargs)
        self.trainable = False
        self.gesture_map = {i: name for i, name in enumerate(config.MODEL.GESTURES)}

    def forward(self, input):
        raise NotImplementedError('Call self.predict() instead for HeuristicModel instance!')

    def predict(self, features, fingers):
        '''Predict gesture class based on heiristic calculation.
        
        Args:
            features (list): List of landmark objects in a single frame.
            fingers (list of list): List of lists of 0/1 indicating detection of fingers.
        '''
        fingers = np.stack(fingers, 0)
        fingers = np.sum(fingers, 0) > (len(fingers) // 2)

        def distance(pt1, pt2):
            return np.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

        pick_dist = distance(features[4], features[8]) / (distance(features[12], features[0]) + 1e-5)
        swipe_dist = np.mean([distance(features[i], features[i+4]) for i in [6, 7, 8]]) \
                     / (distance(features[8], features[0]) + 1e-5)

        # No straight finger detected.
        if not np.any(fingers): return 0

        # Thumb tip to index tip smaller than middle tip to wrist (ok).
        if pick_dist <= 0.09: return 1

        if not fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4] \
            and swipe_dist <= 0.09 \
            and swipe_dist < pick_dist:
            # Only index(1) and middle(2) finger are straight;
            # distances between index finger keypoints and middle finger keypoints;
            # less than the distance from index finger tip to wrist (victory w/ two finger closed, click).
            return 4

        # Full palm.
        if np.all(fingers): return 5

        # Victory.
        if not fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]: return 3

        # Pointing.
        if not fingers[0] and fingers[1] and not fingers[2]: return 2

        # All other cases.
        return 0
