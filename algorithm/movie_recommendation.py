import findspark
findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .getOrCreate()

movies = spark.read.csv('../data/small/movies.csv',header=True,inferSchema=True)
ratings = spark.read.csv('../data/small/ratings.csv',header=True,inferSchema=True)
links = spark.read.csv('../data/small/links.csv',header=True,inferSchema=True)
tags = spark.read.csv('../data/small/tags.csv',header=True,inferSchema=True)


from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator



train, test = ratings.randomSplit([0.8,0.2])
als = ALS(maxIter=3, regParam=0.2, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(train)
prediction = model.transform(test)
prediction.show()


evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
print "Mean Square Error : " + str(evaluator.evaluate(prediction))


ratings.schema



def K_fold_Cross_Validation(data, k=5, verbose=False, **kwags):
    mse_list = []
    k_size = [1.0/k] * k
    k_set = data.randomSplit(k_size)
    for i in xrange(k):
        test = k_set[i]
        train = spark.createDataFrame(sc.emptyRDD(),ratings.schema)
        for j in xrange(k):
            if j!=i:
                train = train.union(k_set[j])
        als = ALS(**kwags)
        model = als.fit(train)
        prediction = model.transform(test)
        mse = evaluator.evaluate(prediction.na.drop())
        mse_list.append(mse)
        if verbose:
            print "Fold : " + str(i+1) + "  MSE : " + str(mse)
    if verbose:
        print "Mean MSE : " + str(sum(mse_list)/k)
    return sum(mse_list)/k



import sys
def Grid_Search():
    ranks = [20, 30]
    maxIters = [10, 20]
    regParams = [0.25, 0.4]
    best_param = {"rank":0,"maxIter":0,"regParam":0}
    min_mse = sys.maxint
    for rank in ranks:
        for maxIter in maxIters:
            for regParam in regParams:
                mse = K_fold_Cross_Validation(ratings,
                                        rank=rank,
                                        maxIter=maxIter,
                                        regParam=regParam,
                                        userCol="userId",
                                        itemCol="movieId",
                                        ratingCol="rating")
                print "rank : " + str(rank) +                 " maxIter : " + str(maxIter) +                 " resParam : " + str(regParam) +                 " MSE : " + str(mse)
                if mse<min_mse:
                    min_mse = mse
                    best_param["rank"] = rank
                    best_param["maxIter"] = maxIter
                    best_param["regParam"] = regParam
    return best_param
                


best_param = Grid_Search()


print best_param


regParam = best_param['regParam']
rank = best_param['rank']
maxIter = best_param["maxIter"]
mse = K_fold_Cross_Validation(ratings,
                              rank = rank,
                              maxIter = maxIter,
                              regParam = regParam,
                              verbose = True,
                              userCol = "userId",
                              itemCol="movieId",
                              ratingCol="rating")

train, test = ratings.randomSplit([0.8,0.2])
als = ALS(rank = rank,
          maxIter = maxIter,
          regParam = regParam,
          userCol="userId",
          itemCol="movieId",
          ratingCol="rating")
model = als.fit(train)
prediction = model.transform(test)
prediction.show()

