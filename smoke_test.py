# smoke_test.py
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('smoke_test').master('local[*]').getOrCreate()
df = spark.createDataFrame([(1,'a'),(2,'b')], ['id','val'])
df.show()

from graphframes import GraphFrame
v = spark.createDataFrame([('a','Alice'),('b','Bob')], ['id','name'])
e = spark.createDataFrame([('a','b','friend')], ['src','dst','relationship'])
g = GraphFrame(v, e)
g.vertices.show()
print('SMOKE TEST PASSED')
