#cluster mode
# spark-submit --master yarn --deploy-mode cluster binomial_logistic_regression.py

#local mode
# spark-submit binomial_logistic_regression.py


from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from datetime import datetime

start_time = datetime.now()
print("Program started : ",start_time)

#spark = SparkSession.builder.appName("ML-DEMO").getOrCreate()

#training = spark.read.format("libsvm").load("data_mllib_sample_libsvm_data.txt")    
# Load training data
training_lr = spark.read.format("libsvm").load("data_mllib_sample_libsvm_data_large.txt")

logreg = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = logreg.fit(training_lr)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# We can also use the multinomial family for binary classification
mlogreg = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlogreg.fit(training_lr)

# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))


end_time = datetime.now()
print("Program end : ",end_time)

time_taken = end_time -start_time
print("Total time taken in minuntes",int(time_taken.seconds))



