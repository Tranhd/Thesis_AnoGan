## Anomaly Detection GAN  
An reconstruction based anomaly detection algorithm using DCGANs.

## User Guide 

#### Create a AnoGan instance
The model is initiated with an Tensorflow session and a directory where to save the parameters after training (optional). After this stage the graph is assembled and are ready to be trained.
```python
tf.reset_default_graph()
sess = tf.Session()
net = AnomalyGAN(sess, save_dir='./AnoGan_save/')
```
#### Train the model
The model is trained using the 'train_model' function. The function expect training data: 'x_train'. In addition optional parameters includes 'batch_size', 'epochs', 'learning_rate' and 'verbose'.
```python
net.train_model(x_train, epochs=100, learning_rate=2e-4, verbose=1)
```

#### Inference
When the model is trained, it is ready to perform anomaly detection on new data. First the anomaly detection graph need to be assembled using the function 'init_anomaly()'. Then an anomaly score can be obtained (the higher the more anomalous) using 'anomaly(query_img)'. The function returns
* All the reconstructed images 'samples'
* The sum of all latent vector reconstruction scores 'losses'
* The index of the best reconstruction 'best_index'
* The loss divided over the different reconstructions 'loss_w'

When performing anomaly detection on new data:
```python
optim, loss, resloss, discloss, w, samples, query, grads = init_anomaly(sess, net)
im, losses, r_loss, d_loss, noise = anomaly(sess, query_img, optim, loss, resloss, discloss, w, query)
```
Returns reconstructions and anomaly scores (losses) from 12 different initial latent vectors.
