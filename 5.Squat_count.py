import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 스쿼트 자세 판별 및 높이 계산 함수
def get_squat_height(landmarks, image_height):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height
    
    # 엉덩이와 무릎의 중간 높이를 계산하여 세로 바 높이로 사용
    return int((left_hip + right_hip + left_knee + right_knee) / 4)

# 스쿼트 횟수 카운트 변수
squat_count = 0
is_down = False

# MediaPipe Pose 모델 초기화
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # 웹캠 또는 동영상 파일 열기
    cap = cv2.VideoCapture(0) # 웹캠 사용

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 처리 및 자세 감지
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 자세 랜드마크 그리기
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width, _ = image.shape
            
            # 스쿼트 자세 높이 계산
            squat_height = get_squat_height(landmarks, height)
            
            # 스쿼트 자세 판별 및 카운트
            if squat_height < height * 0.6 : # 임의의 값으로 스쿼트 다운 판단
                is_down = True
            elif is_down and squat_height > height * 0.7: # 임의의 값으로 스쿼트 업 판단
                squat_count += 1
                is_down = False
                
            # 세로 바 그리기 (수정됨)
            bar_start = (width // 2, height)
            bar_end = (width // 2, squat_height)
            cv2.line(image, bar_start, bar_end, (0, 255, 0), 20) # 화면 아래쪽 기준으로 세로 바 표시

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        # 스쿼트 횟수 화면에 표시
        cv2.putText(image, f"Squat Count: {squat_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 화면 출력
        cv2.imshow('MediaPipe Pose', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()