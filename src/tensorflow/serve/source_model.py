from spark_tensorflow_distributor import MirroredStrategyRunner
import tensorflow as tf


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.const_matrix = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.train_datasets = []

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name='input')])
    def get_matrix(self, x):
        return self.const_matrix + x

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name='train_datasets')])
    def distributed_train(self, train_datasets):
        self.train_datasets = train_datasets
        MirroredStrategyRunner(num_slots=8).run(self.train)

    def train(self):
        multi_worker_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

        multi_worker_model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            metrics=['accuracy'],
        )

        multi_worker_model.fit(x=self.train_datasets, epochs=3, steps_per_epoch=5)


if __name__ == '__main__':
    model = MyModel()

    signatures = {
        "get_const_matrix": model.get_matrix,
        "distributed_train": model.distributed_train
    }

    tf.saved_model.save(obj=model, export_dir='model', signatures=signatures)
