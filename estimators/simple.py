import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# from tensorflow.contrib import learn
# learn.estimators.DNNRegressor
# learn.monitors.BaseMonitor
# from tensorflow.contrib import layers

tf.logging.set_verbosity(tf.logging.ERROR)


img = plt.imread('../photo-128x96.jpg')
WIDTH = 128
HEIGHT = 96


def get_input_fn(x, y, num_epochs=None, shuffle=False):
    return tf.estimator.inputs.numpy_input_fn(
        x={'x': x},
        y=y,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


class CustomMonitor(tf.contrib.learn.monitors.BaseMonitor):
    def step_begin(self, step):
        pass
        # if step % 100 == 0:
        #     pred = estimator.predict({'x': X})
        #     arr = np.array(list(pred)).reshape((WIDTH, HEIGHT, 3))
        #     print(arr.shape)
        #     # tf.summary.image('img-' + str(step), img)
        #     image.imsave('./img-' + str(step) + '.png', arr)

    def step_end(self, step, output):
        pass


X = np.array([[y, x] for y in range(HEIGHT) for x in range(WIDTH)], dtype=np.float32)
X = (X - np.mean(X)) / np.std(X)
Y = img.reshape(HEIGHT * WIDTH, 3)

configs = [
    {'learning_rate': 0.01, 'units': [64, 64, 64]},
    {'learning_rate': 0.001, 'units': [64, 64, 64]},
    {'learning_rate': 0.0005, 'units': [64, 64, 64]},

    {'learning_rate': 0.01, 'units': [64, 64, 64, 64, 64, 64]},
    {'learning_rate': 0.001, 'units': [64, 64, 64, 64, 64, 64]},
    {'learning_rate': 0.0005, 'units': [64, 64, 64, 64, 64, 64]},

    {'learning_rate': 0.01, 'units': [64, 64, 64, 64, 64, 64, 64, 64, 64]},
    {'learning_rate': 0.001, 'units': [64, 64, 64, 64, 64, 64, 64, 64, 64]},
    {'learning_rate': 0.0005, 'units': [64, 64, 64, 64, 64, 64, 64, 64, 64]},

    {'learning_rate': 0.01, 'units': [128, 128, 128, 128, 128]},
    {'learning_rate': 0.001, 'units': [128, 128, 128, 128, 128]},
    {'learning_rate': 0.0005, 'units': [128, 128, 128, 128, 128]},
]

for loop in range(100):
    for i, config in enumerate(configs):
        feature_columns = [tf.feature_column.numeric_column('x', shape=[2])]
        estimator = tf.contrib.learn.DNNRegressor(
            hidden_units=config['units'],
            feature_columns=feature_columns,
            label_dimension=3,
            model_dir='./models/dnn-' + str(i + 1)
            , optimizer=tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
        )
        estimator.fit(input_fn=get_input_fn(X, Y, 1))
        res = estimator.predict({'x': X})
        img_pred = np.array(list(res), dtype=img.dtype).reshape(img.shape)
        plt.imsave('pred-config-' + str(i) + '.' + str(loop + 1) + '.png', img_pred)

        estimator.fit(input_fn=get_input_fn(X, Y, 250))
        res = estimator.evaluate(input_fn=get_input_fn(X, Y, 1))
        print('Eval ' + str(i) + '/' + str(loop + 1) + '  = ' + str(res['loss']))
    print('---')
    # for i in range(1024):
    #     estimator.fit(input_fn=get_input_fn(X, Y, 16), monitors=[CustomMonitor()])
    #     res = estimator.predict({'x': X})
    #     img_pred = np.array(list(res), dtype=img.dtype).reshape(img.shape)
    #     print(img.shape)
    #     print(img_pred.shape)
    #     print('---')
    #     print(img[10][10])
    #     print(img_pred[10][10])
    #     plt.imsave('original.png', img)
    #     plt.imsave('aaa-pred-' + str(i + 720) + '.png', img_pred)
