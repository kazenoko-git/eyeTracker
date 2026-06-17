import pyautogui

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Camera Configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Ear (Eye Aspect Ratio) Thresholds for blink/wink detection
EAR_THRESHOLD_BLINK = 0.20  # Below this is considered a closed eye
EAR_CONSEC_FRAMES_WINK = 10 # Number of consecutive frames for a prolonged wink

# Dwell time click configuration
DWELL_TIME_MS = 1000        # Milliseconds of dwelling required to click
DWELL_RADIUS = 20           # Radius in pixels within which cursor must stay to count as dwelling

# Filtering configuration (EMA or Kalman)
FILTER_TYPE = "EMA"         # "EMA" or "KALMAN"
EMA_ALPHA = 0.3             # Smoothing factor for EMA (lower = smoother but more lag)

# Key to trigger midas touch (Physical Modifier)
# e.g., using spacebar to execute a click where the eye is looking
MIDAS_KEY = 'space'
