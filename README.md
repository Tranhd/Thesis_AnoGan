## Anomaly Detection GAN  
An reconstruction based anomaly detection algorithm using DCGANs.

## User Guide 

#### Create a AnoGan instance
The model is initiated with an Tensorflow session and a directory where to save the parameters after training (optional). After this stage the graph is assembled and are ready to be trained.
```python
tf.reset_default_graph()
sess = tf.Session()
net = AnoGan(sess, save_dir='./MnistCNN_save/')
```
#### Train the model
The model is trained using the 'train_model' function. The function expect training data: 'x_train' with the shape [n_examples, 28, 28, 1]. In addition optional parameters includes 'batch_size', 'epochs', 'learning_rate' and 'verbose'.
```python
net.train_model(x_train, epochs=100, learning_rate=2e-4, verbose=1)
```

#### Inference
When the model is trained, it is ready to perform inference on new data. Prediction is performed with the function 'predict'. The function expects new images with the same shape as x_train, i.e [n_examples, 28, 28, 1]. The function returns 
* The prediction for each of the examples in 'predictions' [n_examples, 1],
* The probability distribution over all classes for each of the examples in 'probs' [n_examples, 10], 
* The activations from each of the layers in the networks as a list in 'activations', where the ith object is the activations for layer i for all examples [n_examples, dimension of activation map of layer i]. 

When predicting the test data:
```python
predictions, probs, activations = net.predict(x_test)
```
 To calculate the accuracy of the predictions the following code-snippet can be used:
 ```python
accuracy = np.sum(np.argmax(y_test, 1) == preds)
print(f'Test accuracy {accuracy/100} %')
```