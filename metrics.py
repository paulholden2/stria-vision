import tensorflow as tf

mnist = tf.keras.datasets.mnist

_,(x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

model = tf.keras.models.load_model('models/mnist_digits.h5')
score = model.evaluate(x_test, y_test)

print('Loss:', score[0])
print('Accuracy:', score[1])
