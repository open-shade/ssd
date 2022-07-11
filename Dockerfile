ARG ROS_VERSION=humble
FROM shaderobotics/pytorch:${ROS_VERSION}

ARG ROS_VERSION=humble
ENV ROS_VERSION=$ROS_VERSION

WORKDIR /home/shade/shade_ws

# install additional dependencies here
RUN apt update && \
    apt install -y \
      python3-colcon-common-extensions \
      python3-pip \
      ros-${ROS_VERSION}-cv-bridge \
      ros-${ROS_VERSION}-vision-opencv && \
    echo "#!/bin/bash" >> /home/shade/shade_ws/start.sh && \
    echo "source /opt/shade/setup.sh" >> /home/shade/shade_ws/start.sh && \
    echo "source /opt/ros/${ROS_VERSION}/setup.sh" >> /home/shade/shade_ws/start.sh && \
    echo "source ./install/setup.sh" >> ./start.sh && \
    echo "ros2 run ssd ssd" >> /home/shade/shade_ws/start.sh && \
    chmod +x ./start.sh

COPY . ./src/ssd

RUN pip3 install ./src/ssd && \
    : "Install the model" && \
    python3 -c "impot torch; ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')" && \
    colcon build

ENTRYPOINT ["/home/shade/shade_ws/start.sh"]
