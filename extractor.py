import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True is required to get the iris landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Right eye indices (MediaPipe's right, which is the left side of the image if mirrored)
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_IRIS_CENTER = 468
        
        # Left eye indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS_CENTER = 473
        
    def process_frame(self, frame):
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results
        
    def get_ear(self, landmarks, eye_indices, img_w, img_h):
        """
        Calculate Eye Aspect Ratio (EAR).
        Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        p = []
        for idx in eye_indices:
            p.append(np.array([landmarks[idx].x * img_w, landmarks[idx].y * img_h]))
        
        p = np.array(p)
        
        dist_2_6 = np.linalg.norm(p[1] - p[5])
        dist_3_5 = np.linalg.norm(p[2] - p[4])
        dist_1_4 = np.linalg.norm(p[0] - p[3])
        
        if dist_1_4 == 0:
            return 0.0
            
        ear = (dist_2_6 + dist_3_5) / (2.0 * dist_1_4)
        return ear
        
    def get_iris_ratio(self, landmarks, eye_indices, iris_center_idx, img_w, img_h):
        """
        Calculate the normalized iris position within the eye boundary.
        Projects the iris vector onto the eye-corner vector.
        """
        p1 = np.array([landmarks[eye_indices[0]].x * img_w, landmarks[eye_indices[0]].y * img_h])
        p4 = np.array([landmarks[eye_indices[3]].x * img_w, landmarks[eye_indices[3]].y * img_h])
        iris = np.array([landmarks[iris_center_idx].x * img_w, landmarks[iris_center_idx].y * img_h])
        
        eye_width = np.linalg.norm(p1 - p4)
        if eye_width == 0: 
            return (0.5, 0.5)
        
        # Directional vector from inner to outer corner
        eye_vec = p4 - p1
        eye_vec_dir = eye_vec / eye_width
        
        iris_vec = iris - p1
        
        # X projection: proportion of iris horizontally
        proj_x = np.dot(iris_vec, eye_vec_dir)
        ratio_x = proj_x / eye_width
        
        # Y projection: proportion vertically. Use orthogonal vector.
        ortho_vec_dir = np.array([-eye_vec_dir[1], eye_vec_dir[0]])
        proj_y = np.dot(iris_vec, ortho_vec_dir)
        ratio_y = proj_y / eye_width
        
        return (ratio_x, ratio_y)
        
    def extract_features(self, frame):
        img_h, img_w = frame.shape[:2]
        results = self.process_frame(frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        right_ear = self.get_ear(landmarks, self.RIGHT_EYE, img_w, img_h)
        left_ear = self.get_ear(landmarks, self.LEFT_EYE, img_w, img_h)
        avg_ear = (right_ear + left_ear) / 2.0
        
        right_ratio = self.get_iris_ratio(landmarks, self.RIGHT_EYE, self.RIGHT_IRIS_CENTER, img_w, img_h)
        left_ratio = self.get_iris_ratio(landmarks, self.LEFT_EYE, self.LEFT_IRIS_CENTER, img_w, img_h)
        
        # Average the ratios for a more stable generic "gaze vector"
        avg_ratio_x = (right_ratio[0] + left_ratio[0]) / 2.0
        avg_ratio_y = (right_ratio[1] + left_ratio[1]) / 2.0
        
        return {
            'ear': avg_ear,
            'ratio_x': avg_ratio_x,
            'ratio_y': avg_ratio_y,
            'landmarks': landmarks
        }
