import cv2
thres=0.5 #threshold to detect object
cap=cv2.VideoCapture(0)  #start video capturing using your default camera 
cap.set(3,648)  #3 represents width of camera=648px
cap.set(4,448)  #4 represents height of camera = 448px
cap.set(10,70)   #10 represents brightness that is 70%
#class names loading
ClassName=[] #empty list
classFiles='coco.names'
with open(classFiles,'rt') as f:
 className=f.read().rstrip('\n').split('\n') #this accesses name and store in className file.
configPath="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  #comtains features
weightPath="frozen_inference_graph.pb"  # contains model
net=cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320) #resizes image to 320x320 pixels
net.setInputScale(1.0/127.5)  #normilize pixel values to [-1,1]
net.setInputMean((127.5,127.5,127.5))  #means substraction ,subttract 127.5 from each channel
net.setInputSwapRB(True)  #swap channels from BGR to RGB
while True:
	success,img=cap.read()
	classIds, confs, bbox=net.detect(img, confThreshold=thres)
	if len(classIds)!=0:
		for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
			cv2.rectangle(img,box,color=(0,255,0),thickness=2)
			cv2.putText(img,className[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
	cv2.imshow("Output",img)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break