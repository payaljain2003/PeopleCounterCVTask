
import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
#from tracker import *
from tracker import Tracker   #wrapper Tracker file to ease the use of DeepSORT
import math
import platform
import time

count=0   
start_point = (70,180) #starting point of the reference line
end_point = (336, 235) #end point of the reference line
total_count_in = 0
total_count_out =0

entry_exit ={}
offset = 20

width = 1020 #width of resized input frame
height = 500 #height of resized input frame
FRAMES_PER_SECOND = 10 #FPS of output video writer


model = YOLO('yolov8s.pt') #object detection algorithm


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
	"""
	Calculate the relative position of a test point with respect to a line defined by two points.

	Args:
		p1 (tuple): The first point defining the line.
		p2 (tuple): The second point defining the line.
		test_point (tuple): The point to determine its relative position.

	Returns:
		int: 1 if the test_point is on one side of the line, -1 if on the other side.
		float: The perpendicular distance from the test_point to the line.
	"""
    
	vector1 = (p2[0] - p1[0], p2[1] - p1[1])
	vector2 = (test_point[0] - p1[0], test_point[1] - p1[1])

    
	cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
	distance = abs(cross_product) / math.sqrt(vector1[0]**2 + vector1[1]**2)


	if cross_product >= 0:
		return 1, distance
	elif cross_product < 0:
		return -1, distance



def get_entry_exit_count(id,pos, start_point, end_point, entry_exit,entry_count, exit_count):
	"""
    Update entry and exit counts based on the relative position of a point.

    Args:
        id: Identifier associated with the point.
        pos (tuple): Current position (cx, cy) of the point.
        start_point (tuple): The starting point of a reference line.
        end_point (tuple): The ending point of a reference line.
        entry_exit (dict): A dictionary to track entry/exit status for each point.
        entry_count (int): Count of entry events.
        exit_count (int): Count of exit events.

    Returns:
        tuple: A tuple containing the updated entry_exit dictionary, entry_count, and exit_count.
	"""

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

	








video_path = os.path.join('.','Data','MainGateLuminous_1.mp4')  #input file

output_path = os.path.join('.','Data','out_MainGateLuminous.mp4')  #output file
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

	results = model.predict(frame)  #results [x1,y1,x2,y2,confidence,class_id]

	a = results[0].boxes.boxes  

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
		id = track.track_id  #unique tracking id for each person
		x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)

		#Calculate centre point of bounding region
		cx = int(x3 + x4) // 2  
		cy = int(y3 + y4) // 2

		
		cv2.rectangle(frame, (x3,y3),(x4,y4), (0,0,255),2)
		cv2.putText(frame, str(id),(x3,y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
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


	writer.write(frame)  #write processed video file
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
