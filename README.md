# spfair2.0 Food/No food - deep learning classifier

At the Skip's app people can take a picture of the delivered food (as we all do at restaurants).

One question comes: The taken picture is a proper food photo or a nude picture?

I am proposing a feature to automaticly classify the picture.

This task can not be done on the cloud. This processing have to be done locally on the app. 

The advantages of local processing are: 
    - not wasting processing on the cloud, this task do not need to be scaled up.
    - save storage and data traffic on the cloud once only valid food image will be uploaded

To do so, I propose the use of Deep Learning (Deep Neural Network) as the classifier for this task.

This technique has been proved to have excelent performance on image processing area, mostly in object classification,
image segmentation. For some of those applications a Deep Neural Network can overperform a human at the same conditions.

