import numpy as np

class StabilizedKalmanFilter:
    def __init__(self, dt=0.034, dim=3, stabilization_factor=0.95, max_missing=10):
        """
        Stabilized Kalman Filter for 3D points with missing-data handling.
        
        Args:
            dt: Time step in seconds
            dim: Spatial dimensions (3 for 3D)
            stabilization_factor: Damps covariance growth (0 < sf ≤ 1)
            max_missing: Max consecutive frames without update before reset
        """
        self.dt = dt
        self.dim = dim
        self.sf = stabilization_factor
        self.max_missing = max_missing
        
        # State: [x, y, z, vx, vy, vz]
        I = np.eye(dim)
        Z = np.zeros((dim, dim))
        self.A = np.block([
            [I, dt*I],
            [Z, I]
        ])
        self.B = np.zeros((2*dim, 1))      # no control input
        self.H = np.hstack([I, Z])         # measure positions only
        
        # Noise
        self.Q = np.eye(2*dim) * 1e-2
        self.R = np.eye(dim) * 1e-1
        
        # Initial state and covariance
        self.x = np.zeros((2*dim,1))
        self.P = np.eye(2*dim)
        
        # Missing detection tracker
        self.missing_count = 0

    def predict(self, u=0):
        """Prediction step."""
        self.x = self.A @ self.x + self.B * u
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.P *= self.sf
        self.P = (self.P + self.P.T) / 2

    def update(self, z):
        """Update step. z can be None for missing measurements."""
        if z is None or np.allclose(z, 0.0):
            # Missing detection
            self.missing_count += 1
            if self.missing_count > self.max_missing:
                self.reset()
            return
        
        z = np.array(z).reshape(-1,1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        self.P = (self.P + self.P.T) / 2
        self.missing_count = 0

    def reset(self):
        """Reset filter if lost for too long."""
        self.x = np.zeros((2*self.dim,1))
        self.P = np.eye(2*self.dim)
        self.missing_count = 0

    def get_position(self):
        return self.x[:self.dim].flatten()

    def get_velocity(self):
        return self.x[self.dim:].flatten()
