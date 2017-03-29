## Grad-CAM implementation in Keras ##

Gradient class activation maps are a visualization technique for deep learning networks.

See the paper: https://arxiv.org/pdf/1610.02391v1.pdf

The paper authors torch implementation: https://github.com/ramprs/grad-cam


This code assumes Tensorflow dimension ordering, and uses the VGG16 network in keras.applications by default (the network weights will be downloaded on first use).


Usage: `python grad-cam.py <path_to_image>`


##### Examples

![enter image description here](https://github.com/jacobgil/keras-grad-cam/blob/master/examples/boat.jpg?raw=true) ![enter image description here](https://github.com/jacobgil/keras-grad-cam/blob/master/examples/persian_cat.jpg?raw=true)

Example image from the [original implementation](https://github.com/ramprs/grad-cam):  

'boxer' (243 or 242 in keras)

![](/examples/cat_dog.png)
![](/examples/cat_dog_242_gradcam.jpg)
![](/examples/cat_dog_242_guided_gradcam.jpg)

'tiger cat' (283 or 282 in keras)

![](/examples/cat_dog.png)
![](/examples/cat_dog_282_gradcam.jpg)
![](/examples/cat_dog_282_guided_gradcam.jpg)



