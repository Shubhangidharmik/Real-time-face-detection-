# Real-time-face-detection
Real time face detection

A Convolutional Neural Network based Tensorflow implementation on facial expression recognition (FER2013 dataset) and achieving 72% accuracy. 


In this Model 'Convolutional Neural Network' is used for Real time face emotion recognition through webcam so based on that, a streamlit app is created which is deployed on Google cloud platform.
The model is trained on the dataset 'FER-13 dataset', which had five emotion categories which are 'Happy', 'Sad', 'Neutral','Angry' and 'Disgust' in which all the images were 48x48 pixel grayscale images of face. This model gave an accuracy of approximately 72% at 30th epoch and it will be given much better accuracy, if we increase the number of epoch.

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
    * Test file : https://github.com/Shubhangidharmik/Realtime-Face-Emotion-Recognition/blob/main/src/test.py.ipynb
* Deploy Your app
* Access it via streamlit
     * File : https://github.com/Shubhangidharmik/Realtime-Face-Emotion-Recognition/blob/main/main.py



## Here is my deployed app link:
### Google Cloud Platform 
  * [https://face-emotion-detection-326315.as.r.appspot.com]


### Streamlit app link
  * [Network URL: http://192.168.43.165:8501]
![Screenshot (1078)](https://user-images.githubusercontent.com/85070726/133897025-8de3e1e3-c8c9-4064-9411-7f84a4b0048a.png)




## Demo on streamlit:
![Screenshot (1076)](https://user-images.githubusercontent.com/85070726/133896895-09e02159-9405-468b-87da-ea3b4bcd0593.png)

