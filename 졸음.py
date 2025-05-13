import cv2
import mediapipe as mp
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText

# 이메일 설정
ADMIN_EMAIL = "수신메일주소"
SENDER_EMAIL = "발신메일주소"
SENDER_PASSWORD = "발신메일비번"
SMTP_SERVER = "smtp.naver.com"
SMTP_PORT = 465

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# 졸음 감지 관련 설정
EYE_CLOSED_THRESHOLD = 0.2
EYE_CLOSED_DURATION = 3  # 졸음 지속 시간 (초)

# 현재 단계
STEP_DROWSINESS = 0
STEP_CLAP = 1
STEP_SQUAT = 2
STEP_ALERT = 3
STEP_EXIT = 4
current_step = STEP_DROWSINESS

# 눈 감김을 판단하는 함수
def is_eyes_closed(landmarks, width, height):
    left_eye_top = landmarks[159].y * height
    left_eye_bottom = landmarks[145].y * height
    right_eye_top = landmarks[386].y * height
    right_eye_bottom = landmarks[374].y * height
    
    left_eye_ratio = abs(left_eye_top - left_eye_bottom) / width
    right_eye_ratio = abs(right_eye_top - right_eye_bottom) / width
    
    return (left_eye_ratio + right_eye_ratio) / 2 < EYE_CLOSED_THRESHOLD

# 이메일 전송 함수
def send_email(subject, message):
    global current_step
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ADMIN_EMAIL
    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, ADMIN_EMAIL, msg.as_string())
        server.quit()
        print("✅ 이메일 전송 완료")
        current_step = STEP_EXIT  # 이메일 전송 후 종료 단계로 변경
    except Exception as e:
        print("❌ 이메일 전송 실패:", e)

# 박수 5회 감지 함수
def detect_clap(cap):
    global current_step
    print("🔹 박수를 5회 감지 중...")
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        clap_count = 0
        while clap_count < 5:
            if current_step == STEP_EXIT:
                return
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                clap_count += 1
                print(f"👏 박수 {clap_count}/5")
                time.sleep(0.5)
            
            cv2.imshow('Drowsiness Detection', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                return
        print("✅ 박수 완료! 정상 상태로 복귀")
        current_step = STEP_SQUAT

# 스쿼트 3회 감지 함수
def detect_squat(cap):
    global current_step
    print("🔹 스쿼트 3회를 감지 중...")
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        squat_count = 0
        while squat_count < 3:
            if current_step == STEP_EXIT:
                return
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width, _ = image.shape
                hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2 * height
                knee_y = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y +
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2 * height
                squat_height = (hip_y + knee_y) / 2
                
                if squat_height > height * 0.6:
                    is_down = True
                elif is_down and squat_height < height * 0.5:
                    squat_count += 1
                    is_down = False
                    print(f"🏋️ 스쿼트 {squat_count}/3")
                    time.sleep(0.5)
                    if squat_count >= 3:
                        print("✅ 스쿼트 완료! 정상 상태로 복귀")
                        current_step = STEP_ALERT
                        return
            
            cv2.imshow('Drowsiness Detection', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                return

# Face Mesh 모델 초기화
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = None
    while cap.isOpened():
        if current_step == STEP_EXIT:
            break
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                height, width, _ = image.shape
                landmarks = face_landmarks.landmark
                
                if is_eyes_closed(landmarks, width, height):
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= EYE_CLOSED_DURATION:
                        start_time = None
                        if current_step == STEP_DROWSINESS:
                            print("⚠️ 졸음 감지! 박수를 5회 쳐 주세요!")
                            current_step = STEP_CLAP
                            detect_clap(cap)
                        elif current_step == STEP_SQUAT:
                            print("⚠️ 2회 졸음 감지! 스쿼트 3회 하세요!")
                            detect_squat(cap)
                        elif current_step == STEP_ALERT:
                            print("⚠️ 3회 졸음 감지! 담당자에게 이메일 전송 중...")
                            send_email("학생이 졸고 있습니다.", "조치 부탁드립니다.")
                            break
                else:
                    start_time = None
                
                mp_drawing.draw_landmarks(image, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
        
        cv2.imshow('Drowsiness Detection', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()