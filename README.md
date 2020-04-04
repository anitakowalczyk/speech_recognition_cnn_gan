# Speech Recognition - CNN and GAN
Convolutional Neural Network and Generative Adversarial Network for Speech Recognition

## Table of contents
* [Technologies](#technologies)
* [Results](#Results)

## Technologies
* Python 3.7
* Tensorflow 1.13.1
* Speech Commands Dataset (http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

## Results
Wave signal and spectrogram of the sample audio file from Speech Commands Dataset

![screenshot](./png/spectogram.PNG)



#### Results of classifier with 78% correctness

<p align="center" style="margin-top: 150px;">
  <img style="display: inline-block;" width="45%" src="./png/classifier1_acc.PNG">
  <img style="display: inline-block;" width="45%" src="./png/classifier1_val.PNG">
 </p>

#### Results of classifier with 83% correctness
<p align="center" style="margin-top: 150px;">
  <img style="display: inline-block;" width="45%"  src="./png/classifier2_acc.PNG">
  <img style="display: inline-block;" width="45%" src="./png/classifier2_val.PNG">
 </p>


#### Results of classifier with 92% correctness
<p align="center" style="margin-top: 150px;">
  <img style="display: inline-block;" width="45%" src="./png/classifier3_acc.PNG">
  <img style="display: inline-block;" width="45%" src="./png/classifier3_val.PNG">
 </p>


#### Results of generative adversarial network using discriminator loss and generator loss
<p align="center" style="margin-top: 150px;">
  <img style="display: inline-block;" src="./png/gan.PNG">
 </p>


