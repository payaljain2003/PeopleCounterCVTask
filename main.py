
import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
#from tracker import *
from tracker import Tracker
import math
import platform
import time

count=0
start_point = (70,180)
end_point = (336, 235)
total_count_in = 0
total_count_out =0
#tracker = Tracker()
entry_exit ={}
offset = 20
width = 1020 
height = 500
FRAMES_PER_SECOND = 5



model = YOLO('yolov8s.pt')


tracker = Tracker()


class_file = open('coco.txt', 'r')
class_names = class_file.read()
class_list = class_names.split('\n')
print(class_list)



def mouse_callback(event, x, y, flags, param):
	if event == cv2.EVENT_MOUSEMOVE:
		coord = [x,y]
		print (coord)

cv2.namedWindow('peoplecount')
cv2.setMouseCallback('peoplecount', mouse_callback)



def relative_position(p1, p2, test_point):
    # Calculate vectors from p1 to p2 and p1 to the test point
    vector1 = (p2[0] - p1[0], p2[1] - p1[1])
    vector2 = (test_point[0] - p1[0], test_point[1] - p1[1])

    # Calculate the cross product of the two vectors
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    distance = abs(cross_product) / math.sqrt(vector1[0]**2 + vector1[1]**2)

    # Determine the side of the point based on the sign of the cross product
    if cross_product >= 0:
        return 1, distance
    elif cross_product < 0:
        return -1, distance



def get_entry_exit_count(id,pos, start_point, end_point, entry_exit,entry_count, exit_count):
	#pos = (cx,cy)
	curr_pos, distance = relative_position(start_point, end_point, pos)
	#print("[get_entry_exit_count] id, curr_pos , distance", id, curr_pos, distance)
	if distance <= offset:
		if id in entry_exit:
			if (entry_exit[id] != curr_pos) :
				if curr_pos < 0:
					entry_count = entry_count+1
				else:
					exit_count = exit_count+1
				entry_exit.pop(id)
			entry_exit[id] = curr_pos
		else:
			entry_exit[id]=curr_pos
	return entry_exit, entry_count, exit_count

	








video_path = os.path.join('.','Data','MainGateLuminous.mp4')

output_path = os.path.join('.','Data','out_MainGateLuminous.mp4')
if platform.system() == "Windows":
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
else:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

writer = cv2.VideoWriter(output_path, fourcc, FRAMES_PER_SECOND, (width, height))



cap = cv2.VideoCapture(video_path)

start_time = time.time()
while True:
	ret, frame = cap.read()

	if ret is None:
		break
	if frame is None or frame.size==0:
		break
	frame = cv2.resize(frame,(width, height))
	count = count+1
	if count % 3 !=0:
		continue

	results = model.predict(frame)

	a = results[0].boxes.boxes
	'''
	print(results[0])
	print('**************************************')
	print(results[0].boxes)
	print('#################################')
	'''
	#print(a)
	distance_threshold = 10
	object_dataframe = pd.DataFrame(a).astype('float')
	print (object_dataframe)
	coord_list =[]
	for index, row in object_dataframe.iterrows():
		x1,y1,x2,y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
		score = row[4]
		class_id = int(row[5])
		class_name = class_list[class_id]
		
		if 'person' in class_name and score >= 0.25:
			coord_list.append([x1,y1,x2,y2,score])
			#cv2.rectangle(frame, (x1,y1),(x2,y2), (255,0,255),2)
	
	#print(coord_list)

	tracker.update(frame, coord_list)

	people_count = len(tracker.tracks)
	
	for track in tracker.tracks:
		bbox = track.bbox
		x3,y3,x4,y4 = bbox
		id = track.track_id
		x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
		cx = int(x3 + x4) // 2
		cy = int(y3 + y4) // 2
		#cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)
		cv2.rectangle(frame, (x3,y3),(x4,y4), (0,0,255),2)
		cv2.putText(frame, str(id),(x3,y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
		#cv2.putText(frame, str(score),(x3+40,y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
		if start_point[0] <= cx and end_point[0] > cx:
			entry_exit, total_count_in, total_count_out = get_entry_exit_count(id,(cx,cy), start_point, end_point, entry_exit, total_count_in, total_count_out)
		#entered, exitted = 	get_entry_exit_count(entry_exit)
	
	cv2.line(frame, start_point, end_point, (100,100,200), 2)
	people_count_str = 'Total : ' + str(people_count)
	entered_str = 'Out : ' + str(total_count_in)
	exitted_str = 'In : ' + str(total_count_out)
	cv2.putText(frame, str(people_count_str),(870,423), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
	cv2.putText(frame, str(entered_str),(870,443), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
	cv2.putText(frame, str(exitted_str),(870,463), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


	writer.write(frame)
	cv2.imshow('peoplecount', frame)
	if cv2.waitKey(1) & 0xFF== ord('q'):
		cv2.destroyAllWindows()
		break
	
elapsed_time = time.time() - start_time
fps = count / elapsed_time

print(f"Frames processed: {count}")
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Frames per second (FPS): {fps:.2f}")
	
cap.release()
writer.release()
cv2.destroyAllWindows()
