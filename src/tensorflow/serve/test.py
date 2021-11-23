from pyspark.sql.session import SparkSession

if __name__ == '__main__':
    data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]
    spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
    result = spark.sparkContext.parallelize(data).toDF()
    result.show()
