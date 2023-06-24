from Detector import *
import argparse

detector = Detector()
classFile = "coco.names.txt"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
threshold = 0.5




parser = argparse.ArgumentParser()
parser.add_argument("input_string", type=str, help="Enter a string input.")
args = parser.parse_args()
imagePath = args.input_string

detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath,threshold)