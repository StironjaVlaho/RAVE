# ROS Noetic
FROM ros:noetic

# Prevent console from interacting with the user
ARG DEBIAN_FRONTEND=noninteractive

# This is required else apt-get update throws Hash mismatch error
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update -yqqq

# Source ROS environment
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc

# Install catkin_tools for catkin build, RViz and Gazebo
RUN apt-get install --no-install-recommends -yqqq \
    python3-catkin-tools \
    python3-pip \
    build-essential \
    cmake \
    curl \
    gedit \
    ros-$ROS_DISTRO-rviz \
    ros-$ROS_DISTRO-gazebo-ros


# Install scipy
RUN pip3 install scipy

# Copy the rave folder into the catkin_ws/src directory
COPY ./rave /catkin_ws/src/rave

# Source the catkin workspace in the bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

