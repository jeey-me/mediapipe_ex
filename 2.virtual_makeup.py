import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 얼굴 특징점 활용 (예: 입술 영역에 색상 적용)
                lip_points = [face_landmarks.landmark[point_idx] for point_idx in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]]
                lip_coords = np.array([[int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])] for landmark in lip_points], np.int32)
                cv2.fillPoly(image, [lip_coords], (0, 0, 255)) # 빨간색 입술

                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, 
                                          landmark_drawing_spec=None, connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1))

        cv2.imshow('Virtual Makeup', image)

        # q 또는 esc키를 누르면 종료
        if (cv2.waitKey(5) & 0xFF == 27) or cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()