# mnistpure

<b>This project is based on the mnist dataset - using TensorFlow in a convoluted neural network</b>

<p>The mnist dataset is a collection of handwritten numbers, being loaded into a cNN</p>
 
<p>For this project I've trained the model with 7058 steps, being able to get a loss = 0.29477254(which is okay, I might say)
<p>The images are loaded into the model 28*28 pixel on colorchannel 1 -> black/white (grayscale)
<p>The model is using the ReLu activation function, to filter out negative values, and normalise the data.
<p>Information about dead neurons due to ReLu being used is not provided. 

<p>The datatype is set to an int32. (standard) 

<p>The last training provided follow information: INFO:tensorflow:loss = 0.12128864, step = 7701 (18.603 sec)</p>
