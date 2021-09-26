Environment name : exam_mnist

Required Packeges:

Tensorflow-GPU - 2.2
Pillow -  8.1.0
open-cv  
TensorboardX 
cuda - 10.1

the packeges of this env is exported at exam_mnist.yml file.



Synthetic Image Generation:

run the synthetic_image_gen.py file to generation 0 - 9 digit.
N.B. I could install times new roman font in Linux thus i had generated synthetic image using Ubuntu's default font.

Train, Test, Validation split:
As this is autoencoder image construction I didn't set any test data. I put the test data of Mnist into validation data. 


Training:

Run train.py file to train the model.

Logs and tensorboard:
Logs of the model has been saved at logs directory and evaluated them at tensorboard.

Test:

To test single image output run the single_image_prediction.py file.



