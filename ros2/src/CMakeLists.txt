cmake_minimum_required(VERSION 3.5)
project(harmony_segmentation)


install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
