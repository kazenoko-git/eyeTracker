import cv2
import config
from extractor import LandmarkExtractor
from filter import GazeFilter
from calibrator import Calibrator
from controller import MouseController

def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    extractor = LandmarkExtractor()
    calibrator = Calibrator()
    gaze_filter = GazeFilter(filter_type=config.FILTER_TYPE, ema_alpha=config.EMA_ALPHA)
    controller = MouseController()

    print("=== Eye Tracker Started ===")
    print("Look at the red circles in the video feed and press 'c' to capture calibration points.")
    print("Press 'q' at any time to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        features = extractor.extract_features(frame)
        if features:
            rx, ry = features['ratio_x'], features['ratio_y']
            ear = features['ear']
            
            # Calibration Phase
            if not calibrator.calibrated:
                target = calibrator.get_current_point()
                if target:
                    # Scale screen coordinates down to camera frame size to draw calibration dots
                    # Alternatively, just draw them on a separate fullscreen window.
                    # For simplicity, we draw relative to camera frame dimensions.
                    scale_x = config.CAMERA_WIDTH / config.SCREEN_WIDTH
                    scale_y = config.CAMERA_HEIGHT / config.SCREEN_HEIGHT
                    cam_target_x = int(target[0] * scale_x)
                    cam_target_y = int(target[1] * scale_y)
                    
                    cv2.circle(frame, (cam_target_x, cam_target_y), 15, (0, 0, 255), -1)
                    cv2.putText(frame, f"Look at dot & press 'c' ({calibrator.current_point_idx + 1}/{len(calibrator.points)})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        print(f"Captured point {calibrator.current_point_idx + 1}")
                        calibrator.capture_point(rx, ry)
            
            # Tracking Phase
            else:
                # Map eye ratios to raw screen coordinates
                raw_screen_x, raw_screen_y = calibrator.map_to_screen(rx, ry)
                
                # Apply smoothing
                smooth_x, smooth_y = gaze_filter.update(raw_screen_x, raw_screen_y)
                
                # Update cursor and triggers
                controller.process_actions(smooth_x, smooth_y, ear)

                cv2.putText(frame, "Tracking Active", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show coordinates for debug
                cv2.putText(frame, f"Raw: ({raw_screen_x}, {raw_screen_y})", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Smooth: ({int(smooth_x)}, {int(smooth_y)})", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Eye Tracker Calibration / Debug", frame)

        # Break loop with 'q'
        # Do not block on waitKey so we don't drop frames
        # waitKey(1) waits for 1ms
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
