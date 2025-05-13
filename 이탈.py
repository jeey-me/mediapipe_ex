import smtplib
import cv2
import time
import mediapipe as mp
from email.mime.text import MIMEText

# 이메일 설정
ADMIN_EMAIL = "수신메일주소"
SENDER_EMAIL = "발신메일주소"
SENDER_PASSWORD = "발신메일비번"

# 얼굴 감지 관련 변수
last_seen = time.time()
time_threshold = 10  # 초 단위, 자리 이탈로 간주할 시간

# MediaPipe 얼굴 감지 모델 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 이메일 전송 함수
def send_email(subject, message):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ADMIN_EMAIL
    
    try:
        server = smtplib.SMTP_SSL('smtp.naver.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, ADMIN_EMAIL, msg.as_string())
        server.quit()
        print("✅ 이메일 전송 완료")
    except Exception as e:
        print("❌ 이메일 전송 실패:", e)

# 카메라 실행
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break
    
    # BGR -> RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    # 얼굴 감지 확인
    if results.detections:
        last_seen = time.time()  # 마지막 얼굴 감지 시간 업데이트
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)
    
    # 자리 이탈 감지 (설정한 시간 이상 얼굴 감지 안 될 경우 이메일 발송)
    if time.time() - last_seen > time_threshold:
        send_email("자리 이탈 감지", "학생이 설정된 시간 동안 자리에 없습니다! 담당자가 확인해주세요.")
        last_seen = time.time()  # 이메일 중복 발송 방지
    
    cv2.imshow('Face Detection', frame)
    
    # ESC 또는 'q' 키를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        print("🔴 얼굴 인식 종료")
        break

cap.release()
cv2.destroyAllWindows()
