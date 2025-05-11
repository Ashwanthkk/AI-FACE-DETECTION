import cv2 as cv
import numpy as np
import urllib.request
from inference import predict_face
from PIL import Image

urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt', 'deploy.prototxt')
urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel', 'res10_300x300_ssd_iter_140000.caffemodel')

net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

idx="idx_to_class.pth"
weight=" family_cnn.pth"

video_capture=cv.VideoCapture("/content/WIN_20210716_20_28_26_Pro.mp4")

video_codec=cv.VideoWriter_fourcc(*'mp4v')
#output=cv.VideoWriter("Face_detected.mp4", video_codec, 20.0, (640, 480))

width = 640
height = 480
fps = video_capture.get(cv.CAP_PROP_FPS)
print(fps)

#output=cv.VideoWriter("Face_detected.mp4", video_codec, fps, (width,height))

if fps==0:
  fps=20.0

output=cv.VideoWriter("Face_detected.mp4", video_codec, fps, (width,height))

while True:
  ret,frame=video_capture.read()

  if not ret:
    break

  frame=cv.resize(frame,(width,height)) #converts video into lower resolutuion
  blob=cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)) #convert each frame into blob 

  net.setInput(blob)

  detected_values=net.forward()
  (h,w)=frame.shape[:2] #extracts height and width from [height,width,channels]

  for i in range(detected_values.shape[2]):
    high_value=detected_values[0,0,i,2]

    if high_value>0.8: #if face has  more than 80% accuracy
      box=detected_values[0, 0, i, 3:7] * np.array([w, h, w, h])
      #now converts the  float numbers into integer format
      (startx,starty,endx,endy)=box.astype("int")

      #print(startx,starty,endx,endy)
      face_crop=frame[starty:endy,startx:endx] #4 corner coordinates of the detected face [y1:y2,x1:x2]
      print(type(face_crop))
      print(f"face region:{face_crop}")

      '''converts the extracted face colour into rgb since my model weights are trained 
          with rgb color .This can be changed if the model weights are trained with grayscale faces'''
      
      face_rgb=cv.cvtColor(face_crop,cv.COLOR_BGR2RGB)
      pil_image=Image.fromarray(face_crop).copy() #converts np array image to pil image type since predict_face accepts pil image type
      result_name=predict_face(pil_image,idx,weight)#Predicts the correct face

      print(f"detetcted face name : {result_name}")
      
      #Draw rectangle over the face
      cv.rectangle(frame,(startx,starty),(endx,endy),(0,255,0),2)

  output.write(frame)


video_capture.release()
output.release()

