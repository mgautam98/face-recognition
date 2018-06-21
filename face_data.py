import numpy as np
import cv2

#Initialize the camera
cap = cv2.VideoCapture(0)

#load the haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

#dataset location
skip = 0
face_data = []
dataset_path = './face_dataset/'


file_name = raw_input("Enter the name of the person: ")

while True:
	ret, frame = cap.read()
	if ret == False:
		continue
	#1.Convert to grey scale
	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#2.Detect multi faces in the frame
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	k=1

	#To remove background faces
	faces = sorted(faces, key = lambda x:x[2]*x[3], reverse = True)
	
	skip+=1

	for face in faces[:1]:
		x,y,w,h = face

		#4.Get the face ROI
		offset = 7
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))
		
		if skip%10 == 0:
			face_data.append(face_section)
			print len(face_data)		
		
		#5.display face ROI
		cv2.imshow(str(k), face_section)
		k+=1

		#3.draw rectangel in original image
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
		
		

	cv2.imshow("Video", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#6.convert the dace data to numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print face_data.shape


#7.Store the data
np.save(dataset_path + file_name, face_data)
print "Dataset saved at:{}".format(dataset_path + file_name +'.npy')


cv2.destroyAllWindows()


