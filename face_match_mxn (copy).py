import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--db", type = str, required=True)
args = parser.parse_args()

# some constants kept as default from facenet
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.Session()

# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

# read 20170512-110547 model file downloaded from https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk
facenet.load_model("20170512-110547/20170512-110547.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces
def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

file_embedding_dict = {}
for subdir, dirs, files in os.walk(args.db):
	for file in files:
		full_path = os.path.join(subdir, file)
		if full_path.endswith('.jpg'):
			img2 = cv2.imread(full_path)
			face2 = getFace(img2)
			file_embedding_dict[file] = face2[0]['embedding']

print("file_embedding_dict", file_embedding_dict)


def compare2multiple(embed_dict):
	threshold = 1.10    # set yourself to meet your requirement
	compare_result_dict = {}
	dist_list = []
	count = 0
	for kf1, vf1 in embed_dict.items():
		compare_result_dict[kf1] = []
		for kf2, vf2 in embed_dict.items():
			dist = np.sqrt(np.sum(np.square(np.subtract(vf1, vf2))))
			count = count + 1
			if dist <= threshold:
				img_thresh_dict = {kf2: dist}
				compare_result_dict[kf1].append(img_thresh_dict)
	print (count)
	return compare_result_dict


print("compare result: ", compare2multiple(file_embedding_dict))
		




