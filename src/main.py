import cv2
import os

curr_directory = r'D:\python-workspace\Open_Source_SW\src'
resource_directory = r'D:\python-workspace\Open_Source_SW\resource'
file_list = os.listdir(resource_directory)


face_cascade = cv2.CascadeClassifier(os.path.join(curr_directory, 'haarcascade_frontalface_default.xml'))
eyes_cascade = cv2.CascadeClassifier(os.path.join(curr_directory, 'haarcascade_eye.xml'))

for filename in file_list : 
    video_file = os.path.join(resource_directory, filename)
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 12, minSize=(80, 80))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_gray = gray[y:y + h, x:x + w]
                
                # 눈 검출
                eyes = eyes_cascade.detectMultiScale(face_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            
            cv2.imshow('Video', frame)
            
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
