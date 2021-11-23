from spark_tensorflow_distributor import MirroredStrategyRunner


def train_custom_strategy():
    import tensorflow as tf

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)

    with strategy.scope():
        import uuid

        BUFFER_SIZE = 10000
        BATCH_SIZE = 64

        def make_datasets():
            (mnist_images, mnist_labels), _ = \
                tf.keras.datasets.mnist.load_data(path=str(uuid.uuid4())+'mnist.npz')

            dataset = tf.data.Dataset.from_tensor_slices((
                tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
                tf.cast(mnist_labels, tf.int64))
            )
            dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
            return dataset

        def build_and_compile_cnn_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])
            model.compile(
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                metrics=['accuracy'],
            )
            return model

        train_datasets = make_datasets()
        multi_worker_model = build_and_compile_cnn_model()

        # Specify the data auto-shard policy: DATA
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.DATA
        train_datasets = train_datasets.with_options(options)

        multi_worker_model.fit(x=train_datasets, epochs=3, steps_per_epoch=5)


if __name__ == '__main__':

    # Use the local mode to verify `CollectiveCommunication.NCCL` is printed in the logs
    runner = MirroredStrategyRunner(num_slots=1, use_custom_strategy=True, local_mode=True, use_gpu=True)
    runner.run(train_custom_strategy)
