import cv2,time, os, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(777)

class Detector:

    def __init__(self):
        pass

    #read coco file (objects that can be detected)
    def readClasses(self,classesFilePath):
        with open(classesFilePath,"r") as f:
            self.classesList = f.read().splitlines()


        #define colors for each class
        self.colorList = np.random.uniform(low=0,high=255,size=(len(self.classesList),3 ))



    def downloadModel(self,modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index(".")]
        self.cacheDir = "./pretrained_model"
        os.makedirs(self.cacheDir,exist_ok=True)

        get_file(fname=fileName,origin=modelURL,cache_dir=self.cacheDir,cache_subdir="checkpoints",extract=True)


    def loadModel(self):

        print("Loading model" + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints",self.modelName,"saved_model"))

        print("Loading of a: " + self.modelName + " model was successfully finished")

    def createBoundingBox(self,image,threshold = 0.5):
        start = time.time()
        #numpy array
        inputTensor = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        #converting it to a tensor, because model accepts tenosrs
        inputTensor = tf.convert_to_tensor(inputTensor, dtype = tf.uint8)

        #batch will only contain 1 image, (1 image per batch)

        inputTensor = inputTensor[tf.newaxis,...]

        #predictions are dictionaries
        detections = self.model(inputTensor)

        #we extract the bounding boxes
        bboxs = detections["detection_boxes"][0].numpy()
        classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classScores = detections["detection_scores"][0].numpy()


        imH, imW, imC = image.shape

        #allow 50% of overlap between the each bounding box that was created for each object on the image
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold,
                                               score_threshold=threshold)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                # Confidence score for the particular object
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]
                # extracting label for the object
                classLabelText = self.classesList[classIndex]
                # extracting color of boundingbox for the class
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                # unpack bounding box so we can get values of the pixels on x and y axis
                # they are relative to widht and the height of the image and they are not absolute locations

                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=classColor,thickness = 2)
                cv2.putText(image,displayText,(xmin,ymin-10),cv2.FONT_HERSHEY_PLAIN,1,classColor,thickness = 2)


                lineWidth = min(int((xmax-xmin)*0.2),int((ymax-ymin)*0.2))

                cv2.line(image,(xmin,ymin),(xmin+lineWidth,ymin),classColor,thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

                cv2.line(image,(xmax,ymin),(xmax-lineWidth,ymin),classColor,thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)


                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)
            return image






    def predictImage(self,imagePath,threshold=0.5):
        image = cv2.imread(imagePath)
        bboxImage = self.createBoundingBox(image,threshold)
        cv2.imwrite(self.modelName + ".jpg",bboxImage)
        cv2.imshow("Result",bboxImage)
        cv2.waitKey(0)

        cv2.destroyAllWindows()