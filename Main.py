import cv2
import numpy as np
import datetime
import os
import time

# Create a directory for storing recordings if it doesn't exist
save_dir = "recordings"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Initialize variables for recording status and the VideoWriter
is_recording = False
out = None
motion_start_time = None
min_record_time = 2  # seconds

# Read two frames to detect motion
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Compute the absolute difference between the two frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if there is any contour detected
    if len(contours) > 0:
        if not is_recording:
            # Start recording if not already recording
            is_recording = True
            motion_start_time = time.time()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_path = os.path.join(save_dir, f'motion_{timestamp}.avi')
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
            print(f"Started recording: {video_path}")
        else:
            # Update the motion start time
            motion_start_time = time.time()
    else:
        if is_recording:
            # Check if the recording has lasted at least 2 seconds
            elapsed_time = time.time() - motion_start_time
            if elapsed_time >= min_record_time:
                # Stop recording if there is no motion and minimum time has elapsed
                is_recording = False
                out.release()
                print("Stopped recording")
            else:
                # Continue recording until minimum time is reached
                print(f"Recording for at least {min_record_time} seconds")

    # If recording, write the frame to the video file
    if is_recording:
        out.write(frame1)

    # Display the frame with contours
    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Motion Detection", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if is_recording:
    out.release()
cv2.destroyAllWindows()
