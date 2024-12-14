import time
import PoseEstimationModuelFP as pem
import numpy as np
import cv2 as cv
import math

cap = cv.VideoCapture(0)
count = 0
direction = 0

prev_time=0
curr_time=0

estimator = pem.PoseEstimationModule()

while True:
    isTrue, frame = cap.read()
    if not isTrue:
        break

    f = cv.flip(frame, 1)
    f = estimator.draw_points(f)
    pList = estimator.return_points(f)

    if pList:
        x1, y1 = pList[12][1:]
        x2, y2 = pList[14][1:]
        x3, y3 = pList[16][1:]

        cv.circle(f, (x1, y1), 10, (255, 0, 0), thickness=-1)
        cv.circle(f, (x1, y1), 15, (255, 0, 0), thickness=1)

        cv.circle(f, (x2, y2), 10, (255, 0, 0), thickness=-1)
        cv.circle(f, (x2, y2), 15, (255, 0, 0), thickness=1)

        cv.circle(f, (x3, y3), 10, (255, 0, 0), thickness=-1)
        cv.circle(f, (x3, y3), 15, (255, 0, 0), thickness=1)

        cv.line(f, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
        cv.line(f, (x2, y2), (x3, y3), (0, 255, 0), thickness=5)

        raw_angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle = raw_angle + 360 if raw_angle < 0 else raw_angle
        angle = min(angle, 360 - angle)

        cv.putText(f, str(int(angle)), (x2 + 10, y2 + 40), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

        per = np.interp(angle, (50, 160), (0, 100))
        bar=np.interp(angle,(50,160),(650,100))

        if per == 100:
            if direction == 0:
                count += 0.5
                direction = 1

        if per == 0:
            if direction == 1:
                count += 0.5
                direction = 0

        cv.rectangle(f, (1650, 100), (1700, 650), (255, 255, 255), cv.FILLED)
        cv.rectangle(f, (1650, int(bar)), (1700, 650), (0, 255, 0), cv.FILLED)
        cv.rectangle(f, (1650, 100), (1700, 650), (0, 0, 0), 3)
        cv.putText(f, f'{int(per)}%', (1600, 90), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), thickness=3)

    curr_time=time.time()
    fps=1/(curr_time-prev_time)
    prev_time=curr_time

    cv.rectangle(f, (30, 250), (300, 320), (0, 0, 0), -1)
    cv.putText(f, f'FPS: {int(fps)}', (50, 300), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=3)

    cv.rectangle(f, (30, 50), (300, 130), (0, 0, 0), -1)
    cv.putText(f, f'Count: {int(count)}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=3)

    cv.imshow('Bicep Curls', f)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
