# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
 
# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		if result['confidence'] > 0.60:
		    # get coordinates
			x, y, width, height = result['box']
			# create the shape
			rect = Rectangle((x, y), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
                        # insert confidence score
			conf= result['confidence']
			b = ['{:g}'.format(float('{:.8g}'.format(conf)))]
			ax.text(x,y,b,fontsize=6, color='green')
	# show the plot
	pyplot.show()
 
filename = 'test13.jpg'
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image

draw_image_with_boxes(filename, faces)


