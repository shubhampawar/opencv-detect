import cv2
import os
import sys
if not os.path.exists("haar/haarcascade_frontalface_alt2.xml"and"haar/frontalEyes35x16.xml"and"haar/haarcascade_smile.xml"):
    sys.exit(0)
face = cv2.CascadeClassifier("haar/haarcascade_frontalface_alt2.xml")
eye = cv2.CascadeClassifier("haar/frontalEyes35x16.xml")
smile = cv2.CascadeClassifier("haar/haarcascade_smile.xml")

vid = cv2.VideoCapture(0)
if (vid.isOpened()==True):
    print("Error in capturing")
count = 1
while count < 200:
    ret,image_frame = vid.read()
    if ret == True:
        grey = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(grey, 1.4, 5)
        count+=1
        for a,b,c,d in faces:
            cv2.rectangle(image_frame,(a,b),(a+c,b+d),(255,0,0),2)
            roi_gray = grey[a:a + c, b:c + d]
            roi_color = image_frame[a:a + c, b:c + d]
            eyes = eye.detectMultiScale(roi_gray,1.2,6)
            smiles = smile.detectMultiScale(roi_gray,1.7,10)
            for ae,be,ce,de in eyes:
                cv2.rectangle(roi_color, (ae, be), (ae + ce, be + de), (255, 0, 0), 2)
            for ae, be, ce, de in smiles:
                cv2.rectangle(roi_color, (ae, be), (ae + ce, be + de), (255, 0, 0), 2)

        cv2.imshow("Show by CV2", image_frame)
        cv2.imwrite("img/shs"+str(count)+".jpg", image_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()