import cv2
import time
import datetime

cap = cv2.VideoCapture("test0.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 7
desired_width = 960
ignored_frame_number = 1
is_ignore_frame = True

ret, start_frame = cap.read()
aspect_ratio = start_frame.shape[1] / start_frame.shape[0]
desired_height = int(desired_width / aspect_ratio)
start_frame = cv2.resize(start_frame, (desired_width, desired_height))
while cap.isOpened():
    if ignored_frame_number > 10:
        is_ignore_frame = False
    if is_ignore_frame:
        ignored_frame_number += 1
    ret, frame = cap.read()
    if not ret:
        break

    aspect_ratio = frame.shape[1] / frame.shape[0]
    desired_height = int(desired_width / aspect_ratio)
    frame = cv2.resize(frame, (desired_width, desired_height))

    diff = cv2.absdiff(start_frame, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if motion is detected
    if thresh.sum() > 10000 and ignored_frame_number > 10:
        # print(thresh.sum())
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(start_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(start_frame, "Status: {}".format('Movement'),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(
                f"{current_time}.mp4", fourcc, 20, (desired_width, desired_height))
            print("Started Recording!")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording!')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    cv2.imshow("Camera", start_frame)
    start_frame = frame

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
