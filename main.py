# commands to run
# python main.py --input test.mp4
# python main.py --input test.mp4 --output output.mp4

# importing the necessary packages
import config
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os


# constructing the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# loading the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# deriving the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# loading our YOLO object detector trained on COCO dataset (80 classes)
print("loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# checking if we are going to use GPU
if config.USE_GPU:
	# setting CUDA as the preferable backend and target
	print("setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determining only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initializing the video stream and pointer to output video file
print("accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None


# looping over the frames from the video stream

while True:
	# reading the next frame from the file

	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resizing the frame and then detecting people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# initializing the set of indexes that violate the minimum social
	# distance
	violate = set()

	# ensuring there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extracting all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# looping over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# updating our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)

	# looping over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extracting the bounding box and centroid coordinates, then
		# initializing the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# we update the color
		if i in violate:
			color = (0, 0, 255)

		# drawing (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# drawing the total number of social distancing violations on the
	# output frame
	text = "Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 360),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# showing the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initializing our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, writing the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)
