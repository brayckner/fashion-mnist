import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('sparse_categorical_accuracy') > 0.95):
            print('\nReached 95% accuracy so cancelling training')
            self.model.stop_training = True

callbacks = myCallback()
# Getting MNIST Data
data  = tf.keras.datasets.fashion_mnist

# Fashion MNIST has 60,00 training images and 10,000 test images (Images are grey scale with pixel values between 0 - 255)
# the return from load_data
# - an array of 60,000 28x28 => training images
# - an array of 60,000 values (0-9) => training_labels

# - an array of 10,000 28x28 => test_images | This will be unseen data that will test the efficiency of the network after training
# - an array of 10,000 values (0-9) => test_labels

(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Goal: To fit the training images to the training labels

# 1) Normalizing the images to be values of 0 and 1
training_images = training_images / 255.0 
test_images = test_images / 255.0 

# 2) Defining the Neural Network that makes up our model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 3) Compiling the model
# Adam - Faster and More efficient than stochastic gradient descent (sgd)
# Sparse categorical crossentropy - We are trying to categorize images of clothes into 1 of 10 categories so this works best for loss
# Metrics - We want to report back on the accuracy of the network so we choose accuracy
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy, 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# 4) Fitting
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

# 5) Evaluate the model (Pass the 10,000 images that the model has not seen and compare it's result with the labels given)
model.evaluate(test_images, test_labels)

# Exploring the Model Output
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])