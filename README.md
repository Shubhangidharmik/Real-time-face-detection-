# Real-time-face-detection
Real time face detection

A Convolutional Neural Network based Tensorflow implementation on facial expression recognition (FER2013 dataset) and achieving 72% accuracy. 


In this Model 'Convolutional Neural Network' is used for Real time face emotion recognition through webcam so based on that, a streamlit app is created which is deployed on Google cloud platform.
The model is trained on the dataset 'FER-13 dataset', which had five emotion categories which are 'Happy', 'Sad', 'Neutral','Angry' and 'Disgust' in which all the images were 48x48 pixel grayscale images of face. This model gave an accuracy of approximately 55% at 6th epoch and it will be given much better accuracy, if we increase the number of epoch.

![Screenshot (1142)](https://user-images.githubusercontent.com/85070726/134208281-299fa3e1-30da-4b0b-84df-7cf5a3f19036.png)


### Dependencies:
- python 3.8
- OpenCV
- Keras with TensorFlow as backend<br/>
- Streamlit framework for web implementation

### FER2013 Dataset:
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data<br/>
- Image Properties: 48 x 48 pixels (2304 bytes)<br/>
- Labels: 
> * 0 - Angry :angry:</br>
> * 1 - Disgust :anguished:<br/>
> * 2 - Fear :fearful:<br/>
> * 3 - Happy :smiley:<br/>
> * 4 - Sad :disappointed:<br/>
> * 5 - Surprise :open_mouth:<br/>
> * 6 - Neutral :neutral_face:<br/>
- The training set consists of 28,708 examples.<br/>
- The model is represented as a json file : fer.json
- The model is represented by : fer2013.h5

### Try it out:
* Prepare data
    * Dataset : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
* Train 
    * CNN Model : https://github.com/Shubhangidharmik/Realtime-Face-Emotion-Recognition/blob/main/src/emotion_recognition.py.ipynb 
* Test it
    * Real time face emotion recognition through webcam : https://github.com/Shubhangidharmik/Realtime-Face-Emotion-Recognition/blob/main/src/test.py.ipynb
* Deploy Your app
* Access it via streamlit
     * Run on streamlit app : https://github.com/Shubhangidharmik/Realtime-Face-Emotion-Recognition/blob/main/app.py



## Here is my deployed app link:
### Google Cloud Platform 
  * [https://face-emotion-detection-326315.as.r.appspot.com]


### Streamlit app link
  * [Network URL: http://192.168.43.165:8501]
![Screenshot (1078)](https://user-images.githubusercontent.com/85070726/133897025-8de3e1e3-c8c9-4064-9411-7f84a4b0048a.png)




## Demo on streamlit:
![Screenshot (1090)](https://user-images.githubusercontent.com/85070726/133998418-ab9465eb-57b7-42f1-9cf8-84f1f27f7d74.png)


![Screenshot (1088)](https://user-images.githubusercontent.com/85070726/133998071-f744fbb2-0235-4a67-9ab6-55f71bf3fbb9.png)


![Screenshot (1091)](https://user-images.githubusercontent.com/85070726/133998638-7e5c7857-8e11-438d-b58f-f1be162a2843.png)


