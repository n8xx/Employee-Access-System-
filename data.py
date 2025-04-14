import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
face_id = input("\n Input ID ==>")
count = 0

while True:
    img = cap.read()[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cv2.imwrite("dataset/." + str(face_id) +"."+str(count)+".jpg", roi_gray)
        count += 1
        print("set" + str(count) + " images")
    cv2.imshow("faces",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
    elif count >= 100:
        print("done")
        break
