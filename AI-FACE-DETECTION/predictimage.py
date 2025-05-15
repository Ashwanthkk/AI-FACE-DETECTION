from inference import predict_face
from PIL import Image
import cv2 as cv
import urllib.request

url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
urllib.request.urlretrieve(url, 'haarcascade_frontalface_default.xml')


face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

face="IMG_20250507_213930.jpg"

face_read=cv.imread(face)

color=cv.cvtColor(face_read,cv.COLOR_BGR2RGB)

detect_face=face_cascade.detectMultiScale(color,1.3,4)
print(detect_face)
print(detect_face.shape)

idx="idx_to_class.pth"
weight="idx_to_class .pth"

for x,y,w,h in detect_face:
   cv.rectangle(face_read,(x,y),(x+w,y+h),(0,255,0),4)
   # Extract the face region
   face_region = face_read[y:y+h, x:x+w]
   #Convert image to pil type
   face_rgb = cv.cvtColor(face_region, cv.COLOR_BGR2RGB)
   pil_image = Image.fromarray(face_rgb).copy()
   pil_image = pil_image.resize((224, 224))

   face_result=predict_face(pil_image,idx,weight)
   cv.putText(face_read,face_result,(x+25, y - 5),cv.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)

   print(face_result)
cv.imshow(face_read)
