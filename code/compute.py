import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import pandas as pd


spark = SparkSession \
    .builder \
    .getOrCreate()

ratings = spark.read.csv('../data/small/ratings.csv',header=True,inferSchema=True)

model = ALSModel.load('CF')

def prediction(userId, movieId):
	df = pd.DataFrame([[userId,movieId]],columns=['userId','movieId'])
	df.to_csv('input.csv')
	data = spark.read.csv('input.csv',header=True,inferSchema=True)
	prediction = model.transform(data)
	num = prediction.select("prediction").collect()[0]

	return num[0]

print prediction(1,31)