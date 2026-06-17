import numpy as np

class GazeFilter:
    def __init__(self, filter_type="EMA", ema_alpha=0.3):
        self.filter_type = filter_type.upper()
        self.ema_alpha = ema_alpha
        
        # EMA State
        self.last_val = None
        
        # Kalman filter setup (simplified 2D constant velocity model)
        if self.filter_type == "KALMAN":
            self.dt = 1.0
            
            # State vector [x, y, vx, vy]
            self.x_state = np.zeros((4, 1))
            
            # State transition matrix
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # Measurement matrix (we only measure x and y)
            self.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            
            # Initial state covariance
            self.P = np.eye(4) * 1000.0
            
            # Measurement noise covariance (higher = trust model more, less jitter)
            self.R = np.eye(2) * 50.0
            
            # Process noise covariance
            self.Q = np.eye(4) * 0.1
            
            self.is_initialized = False

    def update(self, mx, my):
        if self.filter_type == "EMA":
            if self.last_val is None:
                self.last_val = (mx, my)
                return mx, my
            
            sx = self.ema_alpha * mx + (1 - self.ema_alpha) * self.last_val[0]
            sy = self.ema_alpha * my + (1 - self.ema_alpha) * self.last_val[1]
            self.last_val = (sx, sy)
            return sx, sy
            
        elif self.filter_type == "KALMAN":
            z = np.array([[mx], [my]])
            
            if not self.is_initialized:
                self.x_state = np.array([[mx], [my], [0], [0]])
                self.is_initialized = True
                return mx, my
                
            # Prediction Step
            self.x_state = np.dot(self.F, self.x_state)
            self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
            
            # Update Step
            y_residual = z - np.dot(self.H, self.x_state)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            
            self.x_state = self.x_state + np.dot(K, y_residual)
            I = np.eye(4)
            self.P = np.dot((I - np.dot(K, self.H)), self.P)
            
            return float(self.x_state[0][0]), float(self.x_state[1][0])
            
        return mx, my
