import numpy
import os
from transformers import AutoFeatureExtractor, DetrForObjectDetection
import torch
import cv2
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String
from cv_bridge import CvBridge


def predict(image: Image):
    ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    ssd_model.to('cuda')
    ssd_model.eval()

    inputs = utils.prepare_input(image)
    tensor = utils.prepare_tensor(inputs)

    with torch.no_grad():
        detections_batch = ssd_model(tensor)

    results_per_input = utils.decode_results(detections_batch)
    best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

    labels = utils.get_coco_object_dictionary()
    
    return labels, best_results_per_input[0]


class RosIO(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.declare_parameter('pub_image', True)
        self.declare_parameter('pub_boxes', True)
        self.declare_parameter('pub_detections', True)
        self.image_subscription = self.create_subscription(
            Image,
            '/ssd/image_raw',
            self.listener_callback,
            10
        )

        self.image_publisher = self.create_publisher(
            Image,
            '/ssd/image',
            1
        )

        self.detection_publisher = self.create_publisher(
            String,
            '/ssd/detections',
            1
        )
    
        self.boxes_publisher = self.create_publisher(
            String,
            '/ssd/detection_boxes',
            1
        )

    def get_detection_arr(self, result):

        bboxes, classes, confidences = result
        
        dda = Detection2DArray()

        detections = []
        self.counter += 1

        for i in range(len(bboxes)):
            left, bot, right, top = bboxes[i]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]

            detection = Detection2D()

            detection.header.stamp = self.get_clock().now().to_msg()
            detection.header.frame_id = str(self.counter)

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = classes[i]
            hypothesis.score = confidences[i]
            hypothesis.pose.pose.position.x = x
            hypothesis.pose.pose.position.y = y

            detection.results = [hypothesis]

            detection.bbox.center.x = x
            detection.bbox.center.y = y
            detection.bbox.center.theta = 0.0

            detection.bbox.size_x = w
            detection.bbox.size_y = h

            detections.append(detection)
    

        dda.detections = detections
        dda.header.stamp = self.get_clock().now().to_msg()
        dda.header.frame_id = str(self.counter)
        return dda


    def listener_callback(self, msg: Image):
        bridge = CvBridge()
        cv_image: numpy.ndarray = bridge.imgmsg_to_cv2(msg)
        converted_image = PilImage.fromarray(numpy.uint8(cv_image), 'RGB')
        labels, result = predict(converted_image)
        bboxes, classes, confidences = result
        print(f'Predicted Bounding Boxes')

        if self.get_parameter('pub_image').value:
            for i in range(len(bboxes)):
                left, bot, right, top = bboxes[i]
                x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
                converted_image = cv2.rectangle(converted_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.image_publisher.publish(bridge.cv2_to_imgmsg(converted_image))

        if self.get_parameter('pub_detections').value:
            result = []
            for label in classes:
                result.append(labels[label - 1])
            detections = ' '.join(result)
            self.detection_publisher.publish(detections)

        if self.get_parameter('pub_boxes').value:
            arr = self.get_detection_arr(result)
            self.boxes_publisher.publish(arr)

        


def main(args=None):
    print('Single Shot Detection Started')

    rclpy.init(args=args)

    minimal_subscriber = RosIO()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
