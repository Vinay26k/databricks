# Databricks notebook source
# MAGIC %md
# MAGIC ## Spark Machine Learning Assignment

# COMMAND ----------

# MAGIC %md
# MAGIC ### ETL
# MAGIC 1. Extract tables from different datasets
# MAGIC 2. Transform them as per requirements
# MAGIC 3. Load /Pipeline data to required format

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extract
# MAGIC 1. Check which datasets are being provided by databricks 
# MAGIC   - Task: Display all the existing datasets
# MAGIC 2. Extract the given following tables 
# MAGIC    - Task: Extract specific tables given in tables list object

# COMMAND ----------

# DBTITLE 1,🔍 Display Datasets
## display all the available datasets
## Ex: path='dbfs:/databricks-datasets/adult/README.md', name='README.md', size=2672 => output: README.md
## Hint: Use dbutils


# COMMAND ----------

# DBTITLE 1,📝 Required Setup
## Import Required package for the next tasks like the one mentioned in this cell
from pyspark.sql.types import * ## Required for Data types and StructField

Adult_Schema = StructType([
  StructField('age',DoubleType(),True),
  StructField('workclass',StringType(),True),
  StructField('fnlwght',DoubleType(),True),
  StructField('education',StringType(),True),
  StructField('educational-num',DoubleType(),True),
  StructField('marital-status',StringType(),True),
  StructField('occupation',StringType(),True),
  StructField('relationship',StringType(),True),
  StructField('race',StringType(),True),
  StructField('gender',StringType(),True),
  StructField('capital-gain',DoubleType(),True),
  StructField('capital-loss',DoubleType(),True),
  StructField('hours-per-week',DoubleType(),True),
  StructField('native-country',StringType(),True),
  StructField('income',StringType(),True)
])

# COMMAND ----------

# DBTITLE 1,🗜️ Extract required datasets
# # Get path from above task for adult dataset => file name : adult.data, type : csv 
# # Using above schema extract 
# # save file data to variable [data]


# data = [Uncomment and finish this line]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform
# MAGIC 1. Find categorical columns
# MAGIC   - To find out whether a column is categorical or numerical => check schema and classify
# MAGIC 2. Encode the categorical columns
# MAGIC   - Why do we need to encode categorical columns ? - Fill it in the cell
# MAGIC   - Encode using spark ml features

# COMMAND ----------

# DBTITLE 1,🅰&1️⃣ Classify Categorical & numerical columns
## Hint: use dtypes function
## Other working answers also accepted
## Classify categorical columns into cat_cols and numerical to num_cols
cat_cols, num_cols = [], []

## fill your logic here

# COMMAND ----------

# DBTITLE 1,⁉ Why do we need to encode ?
# MAGIC %md
# MAGIC - Remove this and fill the answer here

# COMMAND ----------

# DBTITLE 1,⁉ LabelEncoding VS OneHotEncoding
# MAGIC %md
# MAGIC - Fill your answer here

# COMMAND ----------

# DBTITLE 1,📝Required Setup
## Import required packages for pipeline, string indexer, vector assembler, onehotencoder


stages = [] ## stages in our Pipeline or transformations in dataset
categoricalColumns = [x for x in cat_cols if x not in ('income')] ## Adding this to not to onehotencode this label column

# COMMAND ----------

# DBTITLE 1,🅰 Encoders for categoricalColumns
## Follow below rules
## While label encoding append _Index to the output column
## while onehot encoding append _vec to the output column

## Uncomment and start writing your code
# for categoricalColumn in categoricalColumns:
  ## Add String Indexer logic to variable [stringIndexer]
  
  ## Add One Hot Encoder logic to variable [encoder]
  
  ## Add above transformations to stages using append / + operator
  

# COMMAND ----------

# DBTITLE 1,💰 Encoding - Income column [Example]
# label_stringIdx = StringIndexer(inputCol = "income", outputCol = "label") # use this as example and perform for above task
# stages += [label_stringIdx] # adding stage to stages variable

# COMMAND ----------

# DBTITLE 1,🔢 Feature Assembling
## consider only _vec and numerical columns for featurs
## Use VectorAssembler to assemble all values to features column
## outputCol = 'features'
## Uncomment and complete the code

# features = consider only _vec + numerical columns
# assembler = 
#stages += [assembler]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline

# COMMAND ----------

# DBTITLE 1,[ || ] Create Pipeline 
# pipeline = create pipeline with the stages

# COMMAND ----------

# DBTITLE 1,🎭 Perform all transformations present in pipeline
## now the data will flow in spark and it will take some seconds to complete. 
# pipelineModel = [fit the data to pipeline]
# dataset = [tranform the data from pipeline]

# COMMAND ----------

# DBTITLE 1,📦 Pack / Keep Only relevant columns
## Keep relevant columns 
## label column is the transformation of Income
## Features are the numerical representation of all the data numerical+categorical
## Remove all vec or index suffix columns

# selectedcols = [Remove all vec or index suffix columns]
# dataset = [select only above mentioned columns]
# display(dataset)

# COMMAND ----------

# DBTITLE 1,🪓Random split data to 7:3
# print(dataset.count())
# (train, test) = [Random split this code to 7:3 and use seed=2611]
# print(train.count())
# print(test.count())
# print("Split Matched: ", dataset.count()==train.count()+test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Machine Learning
# MAGIC 1. Train a model and evaluate results
# MAGIC 2. Hyper Parameter Selection - Optional
# MAGIC   * If you want to test yourself, finish this as well

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training & Evaluation

# COMMAND ----------

# DBTITLE 1,📝Required Setup
# Import your packages here for Logistic Regression


# COMMAND ----------

# DBTITLE 1,👩🏻‍🏫 Train - Logistic Regression
## maxIter : 50
# regressor = [create object for regression class]
# regressorModel = [fit above regressor object on to train data]

# COMMAND ----------

# DBTITLE 1,🔮 Predict 
## Hint: Databricks uses transform keyword inplace of predict
# predictions  = [Use above regressorModel to predict income label on test data]

# COMMAND ----------

# DBTITLE 1,[ ℹ ] Information
# selected = predictions.select("label", "prediction", "probability", "age", "occupation")
# display(selected)

# COMMAND ----------

# DBTITLE 1,✔ Model Evaluation
## Import Required package here
## Hint: Use BinaryClassificationEvaluator


## Evaluate Model
# evaluator = [use binaryclassification evaluator]
# evaluator.evaluate(predictions)

# COMMAND ----------

# DBTITLE 1,[ ℹ ] 🔍 Display Metric Name
# evaluator.getMetricName()

# COMMAND ----------

# DBTITLE 1,[ ℹ ] Information
# # Explain Params of model
# regressor.explainParams()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Optional - Hyper Parameter Selection

# COMMAND ----------

# DBTITLE 1,📝 Required Setup
# Import Required package here
# Hint: use ml tuning package


# COMMAND ----------

# DBTITLE 1,💻 Create ParamGrid for Cross Validation
## Create ParamGrid for Cross Validation. In Sklearn gridsearchCV is available for this
# paramGrid = [fill this]

# COMMAND ----------

# DBTITLE 1,🔀 Cross Validator
## Create 5-fold CrossValidator
# cv = [create crossvalidator with numFolds=5]

## Run cross validations
# cvModel = [fit cv on train data]
# this will take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

# DBTITLE 1,🔮 Predict With CrossValidator Model
# Use test set here so we can measure the accuracy of our model on new data
# predictions = [predict with above cvModel]

# COMMAND ----------

# DBTITLE 1,✔ Evaluate CV Model
# cvModel uses the best model found from the Cross Validation
# Evaluate best model using evaluator evalaute function for above predictions


# COMMAND ----------

# DBTITLE 1,[ ℹ ] Information
# # https://www.theanalysisfactor.com/interpreting-the-intercept-in-a-regression-model/
# print('Model Intercept: ', cvModel.bestModel.intercept)

# COMMAND ----------

# DBTITLE 1,[ ℹ ] Information
# # View best model's predictions and probabilities of each prediction class
# selected = predictions.select("label", "prediction", "probability", "age", "occupation")
# display(selected)
