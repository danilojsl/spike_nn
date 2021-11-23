import tensorflow as tf


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.const_matrix = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.train_datasets = []

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name='input')])
    def get_matrix(self, x):
        return self.const_matrix + x

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name='inputs')])
    def distributed_train(self, inputs):
        from pyspark.sql.session import SparkSession
        data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]
        spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
        result = spark.sparkContext.parallelize(data).toDF()
        return inputs


if __name__ == '__main__':
    model = MyModel()

    signatures = {
        "get_const_matrix": model.get_matrix,
        "distributed_train": model.distributed_train
    }

    tf.saved_model.save(obj=model, export_dir='model', signatures=signatures)
