# gaussian_cat_classifier
Image classification problem by classifying foreground and background regions in an image, using a Gaussian classifier
Image classification is an important problem in computer vision and (probably) the most widely used test
bed problem in artificial intelligence. I am using a Gaussian classifier for image classification by classifying foreground and background regions in the image.
 

The image consists of a cat and some grass. The size of this image is 500 ×375 pixels. The left hand side of Figure 1 shows the image, and the right hand side of Figure 1 shows a manually labeled “ground truth”. The aim is to extract the cat from the grass, and compare with the ” ground truth”. However, before proceeding on to tackling the task, let us establish

![alt text](https://github.com/aguram11/gaussian_cat_classifier/blob/master/readme_images/1.png)

The classifier is based on the Maximum A Posteriori(MAP) method assuming it to be a gaussian. More specifically, the MAP procedure is as follows:

For purposes of testing, the image will be extracted in 8 × 8 patches at pixel (i, j) and value of 1 is given to the patch x if it is classified as ”cat”, and 0 otherwise (this is the convention the ”ground truth” image is constructed on). Furthermore, training data files train_cat.txt and train_grass.txt are available. The sizes of the arrays in these files are 64 × N, where N corresponds to the number of training samples and 64 corresponds to the size of the block 8×8. You will need to use these data to compute the necessary parameters for your Gaussian classifier.


  
