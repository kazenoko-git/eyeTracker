import cv2
import numpy as np
import config

class Calibrator:
    def __init__(self):
        self.calibrated = False
        
        # Define a 3x3 grid for the 9-point calibration
        w, h = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        self.points = [
            (int(w * 0.1), int(h * 0.1)),
            (int(w * 0.5), int(h * 0.1)),
            (int(w * 0.9), int(h * 0.1)),
            (int(w * 0.1), int(h * 0.5)),
            (int(w * 0.5), int(h * 0.5)),
            (int(w * 0.9), int(h * 0.5)),
            (int(w * 0.1), int(h * 0.9)),
            (int(w * 0.5), int(h * 0.9)),
            (int(w * 0.9), int(h * 0.9)),
        ]
        self.current_point_idx = 0
        
        # Store tuples of ((ratio_x, ratio_y), (screen_x, screen_y))
        self.collected_data = [] 
        
        # Mapping coefficients
        self.coef_x = None
        self.coef_y = None

    def get_current_point(self):
        """Returns the current target screen point for calibration."""
        if self.current_point_idx < len(self.points):
            return self.points[self.current_point_idx]
        return None

    def capture_point(self, ratio_x, ratio_y):
        """Records the eye ratio mapped to the current screen point."""
        target_pt = self.points[self.current_point_idx]
        self.collected_data.append(((ratio_x, ratio_y), target_pt))
        self.current_point_idx += 1
        
        if self.current_point_idx >= len(self.points):
            self.compute_mapping()
            self.calibrated = True

    def compute_mapping(self):
        """
        Uses multiple linear regression (least squares) to find mapping coefficients.
        Model:
        ScreenX = A * RatioX + B * RatioY + C
        ScreenY = D * RatioX + E * RatioY + F
        """
        A_mat = []
        bx = []
        by = []
        
        for (rx, ry), (sx, sy) in self.collected_data:
            A_mat.append([rx, ry, 1])
            bx.append(sx)
            by.append(sy)
            
        A_mat = np.array(A_mat)
        bx = np.array(bx)
        by = np.array(by)
        
        # Solve for coefficients [A, B, C] and [D, E, F]
        self.coef_x, _, _, _ = np.linalg.lstsq(A_mat, bx, rcond=None)
        self.coef_y, _, _, _ = np.linalg.lstsq(A_mat, by, rcond=None)
        
    def map_to_screen(self, ratio_x, ratio_y):
        """Translates real-time eye ratio to screen coordinates using the model."""
        if not self.calibrated:
            return config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2
            
        screen_x = self.coef_x[0] * ratio_x + self.coef_x[1] * ratio_y + self.coef_x[2]
        screen_y = self.coef_y[0] * ratio_x + self.coef_y[1] * ratio_y + self.coef_y[2]
        
        # Clamp to screen bounds
        screen_x = np.clip(screen_x, 0, config.SCREEN_WIDTH)
        screen_y = np.clip(screen_y, 0, config.SCREEN_HEIGHT)
        
        return int(screen_x), int(screen_y)
