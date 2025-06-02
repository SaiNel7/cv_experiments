# to run: python detect_blazepose.py path/to/image.jpg
import cv2
import mediapipe as mp
import sys

mp_pose = mp.solutions.pose

image_path = sys.argv[1]
image = cv2.imread(image_path)

with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            print(f"{idx}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, visibility={lm.visibility:.2f}")
    else:
        print("No pose detected.")
