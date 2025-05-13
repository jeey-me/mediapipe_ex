# pip install opencv-python mediapipe simpleaudio
import cv2
import mediapipe as mp
import time
import simpleaudio as sa

# MediaPipe 설정
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 눈 감김 감지 변수
eye_closed_time = None
ALARM_THRESHOLD = 5  # 5초

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 경고음 재생 함수
def play_alarm():
    wave_obj = sa.WaveObject.from_wave_file("assets/alarm.wav")
    wave_obj.play()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 상하 반전 (기존 좌우 반전에서 수정)
    frame = cv2.flip(frame, 1)
    
    # 이미지 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    mesh_results = face_mesh.process(rgb_frame)
    
    current_time = time.time()
    face_present = results.detections is not None
    eyes_closed = False
    
    if face_present:
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                left_eye_top = face_landmarks.landmark[159].y
                left_eye_bottom = face_landmarks.landmark[145].y
                right_eye_top = face_landmarks.landmark[386].y
                right_eye_bottom = face_landmarks.landmark[374].y
                
                left_eye_ratio = abs(left_eye_top - left_eye_bottom)
                right_eye_ratio = abs(right_eye_top - right_eye_bottom)
                
                if left_eye_ratio < 0.02 and right_eye_ratio < 0.02:
                    if eye_closed_time is None:
                        eye_closed_time = current_time
                    elif current_time - eye_closed_time >= ALARM_THRESHOLD:
                        play_alarm()
                        eye_closed_time = None  # 알람 후 리셋
                else:
                    eye_closed_time = None
    else:
        play_alarm()
        eye_closed_time = None  # 사람이 없으면 즉시 알람
    
    # 화면 출력
    cv2.imshow('Eye Tracker', frame)
    
    # q 또는 esc키를 누르면 종료
    if (cv2.waitKey(1) & 0xFF == 27) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()
