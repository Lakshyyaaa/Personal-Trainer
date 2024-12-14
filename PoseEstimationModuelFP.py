import cv2 as cv
import mediapipe as mp
import time

class PoseEstimationModule:
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    def draw_points(self, frame):
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(img)
        if results.pose_landmarks:
            pass
            #self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return frame

    def return_points(self, frame):
        lm_list = []
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(img)
        if results.pose_landmarks:
            h, w, c = frame.shape
            for id, lm in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list


def main():
    capture = cv.VideoCapture(0)
    curr_time = 0
    prev_time = 0
    estimator = PoseEstimationModule()

    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break

        frame = estimator.draw_points(frame)
        points = estimator.return_points(frame)

        if points:
            print("Point 0 (nose):", points[0])

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        flipped_frame = cv.flip(frame, 1)
        cv.putText(flipped_frame, f"FPS: {int(fps)}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv.imshow('Pose Estimation', flipped_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
