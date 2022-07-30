# Deterministic behavior of Tensorflow

This code is provided as a support for [this blog post](https://ai-researcher.com/2022/07/30/how-to-perform-repeatable-experiments-with-tensorflow/).

One of the significant drawbacks of Keras and Tensorflow is their non-deterministic behavior. 
This makes the debugging of the code hard, and the replication of training results impossible. 
This repository provides a solution to perform repeatable experiments with Tensorflow and Keras.

### Enabling the deterministic behavior of Tensorflow

By default, Tensorflow is non-deterministic. 
When using Tensorflow with a GPU, two tricks activate its deterministic behavior:

One trick resides in the environment variables of the system. 
We can remove the non-deterministic behavior by setting the following flags to 1:

```python
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

The second trick consists in setting the seed of tensorflow before instantiating the model:

```python

def create_model():
    input_x = Input((28,28,1))
    
    x = Conv2D(filters=32, kernel_size=3, activation="relu")(input_x)
    x = AvgPool2D()(x)
    x = Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = AvgPool2D()(x)
    x = Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(10, activation="sigmoid")(x)
    model = Model(inputs=input_x, outputs=x)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics="accuracy")
    return model
    
# fixing the seed
... # load data

tf.random.set_seed(42)
np.random.seed(1234) # optional, only necessary if using numpy random functions

model = create_model()
# on GPU using CUDA
model.fit(x_train, y_train, batch_size=512, shuffle=True, epochs=5, verbose=0)

```

### License 
The code of this repository is MIT-licensed.
