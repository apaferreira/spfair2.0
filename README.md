At the Skip's app people can take a picture of the delivered food (as we all do at restaurants).

One question comes: The taken picture is a proper food photo or not a food (maybe a nude picture)?

I am proposing a feature to automatically classify the picture.

This task can not be done in the cloud. This processing has to be done locally on the app.

The advantages of local processing are:

- not to waste processing on the cloud, this task does not need to be scaled up.

- save storage and data traffic on the cloud once only valid food image will be uploaded

To do so, I propose the use of Deep Learning (Deep Neural Network) as the classifier for this task.

This technique has been proved to have excellent performance on image processing area, mostly in object classification, image segmentation. For some of those applications, a Deep Neural Network can overperform the human performance.
This project has important non-functional requirements like the response time (the Deep Learning model cannot be too big).

In this project, I used the Food-5k dataset (https://mmspg.epfl.ch/food-image-datasets) which is composed of three parts: Training set
( 3k images), Validation set (1k images) and Testing set (1k images). Both the classes are balanced (50% positive, 50% negative).

The input images dimention varies a lot, so a image resize had to be done. The final size of the RGB images had a 224X224 dimension.


The proposed model is detailed as follows:

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 222, 222, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 111, 111, 32)      0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 111, 111, 32)      128       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 109, 109, 32)      9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 54, 54, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 52, 52, 64)        18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 26, 26, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 24, 24, 64)        36928     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               2359552   
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              263168    
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 2050      
=================================================================
Total params: 2,690,466
Trainable params: 2,690,402
Non-trainable params: 64

The used model includes only four 2d convolutional layers followed by two fully connected layers. The output is categorial to maximaze output separation. It has total number of parameters of aroud 2 Mega weights.

Using this shallow model I achieved a training performance of 0.8747 and validation performance of 0.8262. On the testing set
the reported performance was llllll.



