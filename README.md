<H1 align="center"> YOLOv8_Realtime_Car_Detection_Tracking_and_counting</H1>

## Description:
  In this realtime car detection we are using YOLOV8 model also known as Ultralytics, for the detection of vehicles and deep_sort_pytorch.
  Then we use Flask from python to transfer the realtime photage of the source given by the user on to the webpage along with the Vehicle In/Out count.
  The realtime photage is basically the prediction done on the frames. All the function build in it along with all the methodologies have proper commenting, which give breif info about that function.
## Prerequisite:
 1. YOLOv8
 2. Deep_sort Algorithm
 3. Flask
 4. Python
 5. SocketIO
 6. HTML
 7. CSS

## Steps To RUN it:
  - Clone the repository of MuhammadMoinFaisal
  ```
  git clone https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking.git
  ```
  - Goto the cloned folder.
  ```
  cd YOLOv8-DeepSORT-Object-Tracking
  ```
  - Setting the Directory.
  ```
  cd ultralytics/yolo/v8/detect
  ```
  - Downloading the DeepSORT Files From The Google Drive 
  ```
  https://drive.google.com/drive/folders/1kna8eWGrSfzaR6DtNJ8_GchGgPMv3VC8?usp=sharing
  ```
  - After downloading the DeepSORT Zip file from the drive, unzip it go into the subfolders and place the deep_sort_pytorch folder into the yolo/v8/detect folder
  - Now Clone this git repository Which I have
  ```
  git clone https://github.com/Shifu34/YOLOv8_Realtime_Car_Detection.git
  ```  
  - Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the dependecies
  ```
  pip install -r requriement.txt
  ```
  - Run the code with mentioned command below.
  - For yolov8 realtime object detection and tracking and counting
  ```
  python app.py source="test3.mp4"
  ```
## Bonus Points:
  - If you change the run command with this specific command:
  ```
  python app.py source="0"
  ```
  - This will change the source form video to the camera
  - If you want to use your own video or camera then you should draw the intersector line on the frame captured by the camera
  - You just have to change the starting and ending point of the intersector line which is present in the utils.py file
## Final Output on Webpage:
![Webpage_output_1 ](https://github.com/Shifu34/YOLOv8_Realtime_Car_Detection_Tracking_and_counting/assets/140503589/d6ad6440-dbab-4f8b-8b83-c632bf088f31)
![Webpage_output_2](https://github.com/Shifu34/YOLOv8_Realtime_Car_Detection_Tracking_and_counting/assets/140503589/6ef8a55f-316b-4bc3-b6a7-85bf2200354d)
