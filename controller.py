import pyautogui
import time
import math
from pynput import keyboard
import config

# Disable fail-safe so moving cursor to screen corner doesn't crash the program
pyautogui.FAILSAFE = False

class MouseController:
    def __init__(self):
        self.last_x, self.last_y = 0, 0
        self.dwell_start_time = None
        self.wink_frames = 0
        
        # Intent Triggers
        self.midas_triggered = False
        
        # Start background listener for the physical modifier key (Midas Touch)
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        """Callback for physical key presses to execute 'Midas Touch' click."""
        try:
            if hasattr(key, 'char') and key.char == config.MIDAS_KEY:
                self.midas_triggered = True
            elif key == keyboard.Key.space and config.MIDAS_KEY == 'space':
                self.midas_triggered = True
        except Exception:
            pass

    def move_mouse(self, x, y):
        """Moves the OS cursor to the specified screen coordinates."""
        # Using pyautogui's moveTo
        pyautogui.moveTo(x, y)

    def check_dwell_click(self, x, y):
        """Triggers a left-click if gaze stays within a small radius for a threshold duration."""
        dist = math.hypot(x - self.last_x, y - self.last_y)
        
        if dist < config.DWELL_RADIUS:
            if self.dwell_start_time is None:
                self.dwell_start_time = time.time()
            elif (time.time() - self.dwell_start_time) * 1000 > config.DWELL_TIME_MS:
                pyautogui.click()
                print("Dwell Click Executed!")
                # Reset dwell timer to prevent spamming clicks
                self.dwell_start_time = None 
        else:
            self.dwell_start_time = None
            
        self.last_x, self.last_y = x, y

    def check_midas_click(self):
        """Triggers a click if the physical modifier key was pressed."""
        if self.midas_triggered:
            pyautogui.click()
            print("Midas Touch Click Executed!")
            self.midas_triggered = False

    def check_wink_click(self, ear):
        """Triggers a click based on prolonged blinking or intentional winking."""
        if ear < config.EAR_THRESHOLD_BLINK:
            self.wink_frames += 1
            if self.wink_frames == config.EAR_CONSEC_FRAMES_WINK:
                pyautogui.click()
                print("Prolonged Wink Click Executed!")
        else:
            self.wink_frames = 0
            
    def process_actions(self, x, y, ear):
        """Consolidated handler to process mouse movement and all intent triggers."""
        self.move_mouse(x, y)
        self.check_dwell_click(x, y)
        self.check_midas_click()
        self.check_wink_click(ear)
