# drowsiness_detector_IR

Theme : eye blink detector in  infrared ray

**-Purpose**

The purpose is to detect blinkers to prevent drowsy driving. Infrared cameras (Waveshare / RPi IR-CUT Camera) extract eye areas through dlib to detect blinking even inside dark vehicles. Basically, the goal is to prevent drowsy driving, but as non-face-to-face content is increasing due to Corona, attendance and concentration of classes can also be secured as class participation in large-scale online classes and video conferences can be seen.

: 졸음운전방지를 위한 눈깜빡임 감지가 목적이다. 어두운 차량 내부에서도 눈깜빡임을 감지 할 수 있게 적외선 카메라(Waveshare사의 RPi IR-CUT Camera)로 dlib를 통해 눈 영역을 추출해 낸다.
기본적으로 졸음운전방지를 목표로 하지만 코로나로 인해 비대면성 컨텐츠가 늘어나고있어 대규모의 온라인수업, 화상회의 등에서의 수업참여, 졸음 유무를 알 수 있어 출석, 수업집중도 또한 확보 할 수 있다.

**-supply**

1.Raspberry Pi 4 (raspberry pi 4 ram 4gb)

2. Infrared Camera (RPi IR-CUT Camera)

3. Piezo Buzzer

<img src = "https://user-images.githubusercontent.com/82746560/129918777-5925cc86-9df0-4b8f-9263-153b2db676fa.png" width="80%" height="80%">

**-datasets**

Closed Eyes In The Wild (CEW)

<img src = "https://user-images.githubusercontent.com/82746560/129921166-e5fd5a0a-8be4-46bc-9db1-37a6e964e47d.png" width="80%" height="80%">

**-environment(requirements)**

Window 10

Cuda toolkit = 11.0

Cudnn = v8.1.1

Tensorflow = 2.4.1

Keras = 2.4.3

Csv = 1.0

Dlib = 19.21.1


**-Operation**

1. opencv & keras & dlib : face detection, eye roi

2. train cnn : train with left eye ( and symmetry ), open is 1 close is 0

3. detection : raspberry pi 4 4gb + raspberry pi IR-CUT camera

4. operation : When the car starts. (the engine is on)
               If it remains 'eye closed' for one second(25 frames), we'll give you a first warning,
               and if it's over 2 seconds (60 frames), we'll give you a second warning.
               
               (※ If you continue to detect drowsiness after a secondary warning, the vehicle will speed up.
               I thought it would be good to automatically reduce the speed of the car or turn on the blinker automatically,
               and stop on the shoulder of the road after self-driving at low speed.)

<img src = "https://user-images.githubusercontent.com/82746560/129919091-73890da8-5bf3-4c8c-90bb-474485cd74be.png" width="80%" height="80%">           


<img src = "https://user-images.githubusercontent.com/82746560/129994922-d85bd62b-fef8-465b-9807-96be2337e213.png" width="80%" height="80%"> 

다운로드 해서 파일에 추가
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

**-processing**

1. dlib cnn방식으로 CEW datasets을 훈련
 <train result>
  
  
  <img src = "https://user-images.githubusercontent.com/82746560/129990981-5316c6cb-e88b-47be-960c-d5aed1da2783.png" width="80%" height="80%"> 
 <videostream>
   
   
   <img src = "https://user-images.githubusercontent.com/82746560/129991032-cf53a220-48ef-4a86-86a7-c8b114a229e8.png" width="80%" height="80%"> 
   
   
  => IR영역에서 detection이 잘 이뤄지지 않아 HOG + EAR 방식으로 진행함.

  2016년 Tereza Soukupova & Jan ´ Cech에 의해 제시된 Eyes Aspect Ratio(이하 EAR) 방식을 사용한 code가 있어 인용함.
   <img src = "https://user-images.githubusercontent.com/82746560/129991250-958d61ea-ff3f-4a5c-8614-f233a75eff38.png" width="80%" height="80%"> 
 
  Implementation of Haar Cascade Classifier and Eye Aspect Ratio for Driver Drowsiness Detection Using Raspberry Pi
  https://www.hrpub.org/download/20191230/UJEEEB9-14990984.pdf
 
   
2. HOG + EAR
 <videostream>
   <img src = "https://user-images.githubusercontent.com/82746560/129991338-680d3d3b-c601-4148-8b22-c8afb2f7c4ad.png" width="80%" height="80%"> 


3. raspberry pi settings

   opencv, dlib install and run code

 <img src = "https://user-images.githubusercontent.com/82746560/130000204-e8698e20-c20a-4ee9-96e4-c73dd07cbda8.png" width="50%" height="50%"> 

   
**-result**
<pc>
  
https://user-images.githubusercontent.com/82746560/129997122-6ad471b1-236a-4da1-890b-a06270997661.mp4
  
1초(25frame)동안 eye close이면 1차 경고. 그 후 2초(60frame) 동안 다시 감지되면 2차 경고  


<raspberry pi>

https://user-images.githubusercontent.com/82746560/130000348-53d7066c-0714-4ab0-87aa-4205a9a8431d.mp4
   
밝은 곳에서 얼굴인식을 통한 눈 깜빡임 탐지를 하여 졸음운전 경고 프로세스까지 작동하는 것을 확인

   
---------------------------------------------------------------------------------------------------------------------------------------------------
   
 RPi IR-CUT Camera s/w 자체 오류로 ir영역이 제대로 나오지 않아 추가적인 detection은 못함
   
