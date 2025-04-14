import cv2
import telebot

bot = telebot.TeleBot('5135950090:AAEvLB1lJxrQjx3AvEVPBRiaQPmvLqxrjtQ')
chat_id =709571412
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

names = ['Not found','Found in database']
cap = cv2.VideoCapture(0)

while True:
    img = cap.read()[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.2,5,minSize=(int(5),int(5)))

    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),1)
        id,confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if confidence < 50:
            id = names[1]
            confidence = "{0}%".format(round(confidence))
            cv2.putText(img,(str(confidence)+str(id)),(x+5,y+h-5),font,1,(255,255,0,1))
            bot.send_message(chat_id, id)
            cv2.imwrite('saves/1.png', img)
            bot.send_photo(chat_id, open('saves/1.png', 'rb'))

        else:
            id = names[0]
            cv2.putText(img, str(id), (x + 5, y + h - 5), font, 1, (255, 255, 0, 1))
            bot.send_message(chat_id,id)
            cv2.imwrite('saves/1.png', img)
            bot.send_photo(chat_id, open('saves/1.png', 'rb'))


    cv2.imshow("video",img)
    cv2.waitKey(1)
