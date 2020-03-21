# Detecting disease on beans leaves using image classification

> *This article was originally written for [Meritocracy.is](https://meritocracy.is/blog/)*


Food security is one of the United Nations Sustainable Goals. Yet still, every day more than 25,000 people die of hunger. Overall, there are more than [800 million](https://www.worldometers.info/) undernourished people in the world because of food shortages or inefficient crop yield. Early disease detection on plant leaves can reduce world's crop losses.

In this project I looked at `beans` [dataset](https://github.com/AI-Lab-Makerere/ibean), where each image contains bean plants grown on small farms in East Africa. Beans are a staple in many East African cousines and represent a significant source of protein for school-aged children. Each image from the `beans` dataset is associated with exactly one condition:

1. **angular leaf spot disease**
2. **bean rust disease**
3. **healthy**

<img class="two" src="assets/visualization.png" width="90%" >

Our goal is to **develop a classifier that can predict one of these conditions**. Every image in this dataset is 500-by-500 pixels large and was taken by a smartphone camera on the farm. 

This code contains a basic convolutional neural network architecture and achieves 80 % accuracy on the test set (so far).

## Extensions:

- **Network depth**: what happens if we add or remove convolutional layers?
- **Image augmentation**: what if we apply geometric transformations, such as scaling, cropping, to training images to add diversity and increase the training set size?
- **Image resizing**: what is we crop images with `tf.image.resize_with_crop_or_pad` to reduce background noise?
- **Fine-tuning the hyper-parameters**: what if we try other hyperparameter values?
