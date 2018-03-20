## Anomaly Detection GAN  
An reconstruction based anomaly detection algorithm using DCGANs.

## User Guide 

#### Create a AnoGan instance
The model is initiated with an Tensorflow session and a directory where to save the parameters after training (optional). After this stage the graph is assembled and are ready to be trained.
```python
tf.reset_default_graph()
sess = tf.Session()
anogan = AnoGan(sess, save_dir='./AnoGan_save/')
```
#### Train the model
The model is trained using the 'train_model' function. The function expect training data: 'x_train'. In addition optional parameters includes 'batch_size', 'epochs', 'learning_rate' and 'verbose'.
```python
anogan.train_model(x_train, epochs=100, learning_rate=2e-4, verbose=1)
```

#### Inference
When the model is trained, it is ready to perform anomaly detection on new data. First the anomaly detection graph need to be assembled using the function 'init_anomaly()'. Then an anomaly score can be obtained (the higher the more anomalous) using 'anomaly(query_img)'. The function returns
* All the reconstructed images 'samples'
* The sum of all latent vector reconstruction scores 'losses'
* The index of the best reconstruction 'best_index'
* The loss divided over the different reconstructions 'loss_w'

When performing anomaly detection on new data:
```python
anogan.init_anomaly()
samples, losses, best_index, loss_w = anogan.anomaly(query_img)
```
The reconstruction with the lowest score/loss and its loss can then be accessed using  
```python
anomaly_score = loss_w[best_index]
reconstruction = samples[best_index]
```