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

ALGO_VERSION = os.getenv("MODEL_NAME")

if not ALGO_VERSION:
    ALGO_VERSION = '<default here>'


def predict(image: Image):
    feature_extractor = AutoFeatureExtractor.from_pretrained(ALGO_VERSION)
    model = DetrForObjectDetection.from_pretrained(ALGO_VERSION)

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    # Convert output to be between 0 and 1
    sizes = torch.tensor([tuple(reversed(image.size))])
    result = feature_extractor.post_process(output, sizes)
    
    return result[0]


class RosIO(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.declare_parameter('pub_image', True)
        self.declare_parameter('pub_boxes', True)
        self.declare_parameter('pub_detections', True)
        self.image_subscription = self.create_subscription(
            Image,
            '/<name>/sub/image_raw',
            self.listener_callback,
            10
        )

        self.image_publisher = self.create_publisher(
            Image,
            '/<name>/pub/image',
            1
        )

        self.detection_publisher = self.create_publisher(
            String,
            '/<name>/pub/detections',
            1
        )
    
        self.boxes_publisher = self.create_publisher(
            String,
            '/<name>/pub/detection_boxes',
            1
        )

    def get_detection_arr(self, result):
        dda = Detection2DArray()

        detections = []
        self.counter += 1

        for i in range(len(result['boxes'])):
            detection = Detection2D()

            detection.header.stamp = self.get_clock().now().to_msg()
            detection.header.frame_id = str(self.counter)

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = result['labels'][i].item()
            hypothesis.score = result['scores'][i].item()
            hypothesis.pose.pose.position.x = result['boxes'][i][0].item()
            hypothesis.pose.pose.position.y = result['boxes'][i][1].item()

            detection.results = [hypothesis]

            detection.bbox.center.x = result['boxes'][i][0].item()
            detection.bbox.center.y = result['boxes'][i][1].item()
            detection.bbox.center.theta = 0.0

            detection.bbox.size_x = result['boxes'][i][2].item()
            detection.bbox.size_y = result['boxes'][i][3].item()

            detections.append(detection)
    

        dda.detections = detections
        dda.header.stamp = self.get_clock().now().to_msg()
        dda.header.frame_id = str(self.counter)
        return dda


    def listener_callback(self, msg: Image):
        bridge = CvBridge()
        cv_image: numpy.ndarray = bridge.imgmsg_to_cv2(msg)
        converted_image = PilImage.fromarray(numpy.uint8(cv_image), 'RGB')
        result = predict(converted_image)
        print(f'Predicted Bounding Boxes')

        if self.get_parameter('pub_image').value:
            for box in result['boxes'].tolist():
                x = result['boxes'][0]
                y = result['boxes'][1]
                w = result['boxes'][2]
                h = result['boxes'][3]
                converted_image = cv2.rectangle(converted_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.image_publisher.publish(bridge.cv2_to_imgmsg(converted_image)

        if self.get_parameter('pub_detections').value:
            labels: torch.Tensor = result['labels']
            detections = ' '.join(labels.tolist())
            self.detection_publisher.publish()

        if self.get_parameter('pub_boxes').value:
            arr = self.get_detection_arr(result)
            self.boxes_publisher.publish(arr)

        


def main(args=None):
    print('<name> Started')

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
