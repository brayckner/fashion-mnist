# Fashion MNIST Classification with TensorFlow

This TensorFlow project is focused on training a neural network model to classify articles of clothing using the Fashion MNIST dataset.

![Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)

## Dataset Description

The Fashion MNIST dataset is a popular dataset for image classification tasks. It contains 60,000 training images and 10,000 test images. These images are grayscale with pixel values ranging from 0 to 255. The dataset is split as follows:

- Training Data:
  - 60,000 28x28 images
  - 60,000 corresponding labels (values 0-9) for clothing categories

- Test Data:
  - 10,000 28x28 images
  - 10,000 corresponding labels (values 0-9) for clothing categories

## Project Overview

In this project, we perform the following steps:

1. **Data Normalization**: We normalize the pixel values of the images to fall within the range [0, 1].

2. **Neural Network Architecture**: We define a neural network with three layers:

    - **Input Layer**: This layer takes a 2D array, representing a 28x28 pixel image, and flattens it into a 1D array.
    
    - **Middle or Hidden Layer**: This layer consists of 128 neurons with randomly initialized parameters, enabling the model to learn complex patterns.

    - **Output Layer**: The output layer consists of 10 neurons, each representing a different clothing category. The neuron with the highest activation corresponds to the predicted clothing category.

3. **Model Compilation**: We compile the model by specifying an optimizer, a loss function suitable for classification tasks, and metrics to track accuracy.

4. **Model Training**: We fit the model to the training data, attempting to learn the relationship between the training images and their corresponding labels. We use 50 epochs for training, with a callback that stops training once the accuracy reaches 95%.

5. **Model Evaluation**: Finally, we evaluate the trained model by testing it on previously unseen data—using the test images and their associated labels—to assess its performance.

By following these steps, we aim to create an efficient clothing classification model using TensorFlow and the Fashion MNIST dataset.

Feel free to explore the code and adapt it for your own projects!

[Link to Fashion MNIST Repository](https://github.com/zalandoresearch/fashion-mnist)
