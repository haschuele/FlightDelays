# EDA & Data Cleaning
```python
# Mean Impute Missing Data
numerical_cols = [c for c, d_type in df_60M.dtypes if d_type in ['int','float', 'double']]

df_training = df_60M.filter(col('YEAR') < 2019)

imputer = Imputer(inputCols= numerical_cols, outputCols=numerical_cols)
imputer.setStrategy("mean")

imputer_model = imputer.fit(df_training)
df_60M = imputer_model.transform(df_60M)


# Correlation Analysis
corr_df = pd.DataFrame(list(corr_matrix_df.DEP_DELAY.items()),columns=['Column', 'Correlation'])
corr_df = corr_df[corr_df.Column != 'DEP_DELAY']

plt.figure(figsize=(20, 6))
sns.barplot(x='Column', y='Correlation', data=corr_df)
plt.title('Correlation with DEP_DELAY')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=90)
plt.show()


# One Hot Encode Categorical Data and Split into Train/Test Sets
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

dummy_source_columns = ['OP_UNIQUE_CARRIER', 'DAY_OF_WEEK', 'region', 'origin_type']

# Creating a list to store stages of the Pipeline for one-hot encoding and vector assembly
stages = []

# Process each integer column for one-hot encoding
for c in dummy_source_columns:
    # First convert integer IDs to indexed categories
    indexer = StringIndexer(inputCol=c, outputCol=c + "_indexed")
    encoder = OneHotEncoder(inputCol=c + "_indexed", outputCol=c + "_dummy") #<- this expands category index above into one hot encoding e.g. southwest becomes vector 0,1,0,0 representing midwest = 0, southweest = 1 etc
    stages += [indexer, encoder]

# Create a list of all feature columns, excluding original ID columns and the target variable
feature_columns = [c + "_dummy" for c in dummy_source_columns] + [c for c in df_60M.columns if c not in [*dummy_source_columns,*[c + '_indexed' for c in dummy_source_columns], 'DEP_DEL15', 'FL_DATE']]

# Add the VectorAssembler to the stages
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features") # collects each of the features into a single column
stages += [assembler]

# Create the pipeline
pipeline = Pipeline(stages=stages)

# Fit the pipeline to the data and transform
df_60M_transformed = pipeline.fit(df_60M).transform(df_60M)

# Split the data into training and test sets on flights prior to 2019, and 2019 and later
train_df = df_60M_transformed.filter(df_60M_transformed["YEAR"] < 2019)
test_df = df_60M_transformed.filter(df_60M_transformed["YEAR"] >= 2019)

# Create a pipeline for standardization
std_pipeline = Pipeline(stages=[
    StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
])

# Fit the standardization pipeline to the training data
std_pipeline_model = std_pipeline.fit(train_df)

# Transform both training and test data using the standardization pipeline model
train_df_transformed = std_pipeline_model.transform(train_df)
test_df_transformed = std_pipeline_model.transform(test_df)
```

# Modeling Snippets
```python
# Define Logistic Regression Function
def trainLogRegModel(folds, label, feature, final_val, seeds):
    trainingMetrics = {}
    validationMetrics = {}
    finalValidationMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        validationMetrics[seed_name] = []
        finalValidationMetrics[seed_name] = []
        fold_num = 0

        # Create a Logistic Regression model instance
        lr = LogisticRegression(labelCol=label, featuresCol=feature, weightCol = 'combined_weight')

        # Evaluation metrics
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

        for fold in folds:
            train_df, val_df = fold
            
            # Fit the model on the training data
            model = lr.fit(train_df)

            # Print model coeficients
            print(f"Seed #{seed} Fold #{fold_num} Coefficients: {model.coefficients}")
            fold_num += 1

            # Make predictions on the training data and evaluate
            train_predictions = model.transform(train_df)
            trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

            # Make predictions on the validation data and evaluate
            val_predictions = model.transform(val_df)
            validationMetrics[seed_name].append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

            if (final_val != None):
                final_val_predictions = model.transform(final_val)
                finalValidationMetrics[seed_name].append([evaluator_f1.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])

    return (trainingMetrics, validationMetrics, finalValidationMetrics)


# Run logistic regression on k folds
seeds = [5]
initlog_trainingMetrics, initlog_validationMetrics, initlog_ValMetrics = trainLogRegModel(folds, "DEP_DEL15", 'scaledFeatures', val_df, seeds)


# Define LASSO Function
def trainLassoRegModel(folds, label, feature, final_val, seeds):
    trainingMetrics = {}
    validationMetrics = {}
    finalValidationMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        validationMetrics[seed_name] = []
        finalValidationMetrics[seed_name] = []
        fold_num = 0

        # Create a Logistic Regression model instance
        lr = LogisticRegression(labelCol=label, featuresCol=feature, weightCol = 'combined_weight', regParam =0.01, elasticNetParam=1.0)

        # Evaluation metrics
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

        for fold in folds:
            train_df, val_df = fold
            
            # Fit the model on the training data
            model = lr.fit(train_df)

            # Print model coeficients
            print(f"Seed #{seed} Fold #{fold_num} Coefficients: {model.coefficients}")
            fold_num += 1

            # Make predictions on the training data and evaluate
            train_predictions = model.transform(train_df)
            trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

            # Make predictions on the validation data and evaluate
            val_predictions = model.transform(val_df)
            validationMetrics[seed_name].append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

            if (final_val != None):
                final_val_predictions = model.transform(final_val)
                finalValidationMetrics[seed_name].append([evaluator_f1.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])

    return (trainingMetrics, validationMetrics, finalValidationMetrics)


# Define Random Forest Function
def trainRandomForestModel(folds, label, feature, final_val, numTrees, seeds, featureSubsetStrategy, maxDepth):
    trainingMetrics = {}
    validationMetrics = {}
    finalValidationMetrics = {}

    # Run for multiple seeds    
    for seed in seeds:
        seed_name = f"Seed #{seed}" 
        trainingMetrics[seed_name] = []
        validationMetrics[seed_name] = []
        finalValidationMetrics[seed_name] = []
        fold_num = 0

        # Create a Random Forest model instance
        rf = RandomForestClassifier(labelCol=label, featuresCol=feature, numTrees=numTrees, weightCol='combined_weight', seed=seed, featureSubsetStrategy=featureSubsetStrategy, maxDepth=maxDepth)

        # Evaluation metrics
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
        evaluator_auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

        # Fit model to each fold
        for fold in folds:
            train_df, val_df = fold

            model = rf.fit(train_df)

            # Access feature importances
            feature_importances = model.featureImportances
            print(f"Seed #{seed} Fold #{fold_num} Feature Importances: {feature_importances}")
            fold_num += 1

            # Make predictions on the training data and evaluate
            train_predictions = model.transform(train_df)
            trainingMetrics[seed_name].append([evaluator_f1.evaluate(train_predictions), evaluator_auc.evaluate(train_predictions)])

            # Make predictions on the validation data and evaluate
            val_predictions = model.transform(val_df)
            validationMetrics[seed_name].append([evaluator_f1.evaluate(val_predictions), evaluator_auc.evaluate(val_predictions)])

            if (final_val != None):
                final_val_predictions = model.transform(final_val)
                finalValidationMetrics[seed_name].append([evaluator_f1.evaluate(final_val_predictions), evaluator_auc.evaluate(final_val_predictions)])
    
    return (trainingMetrics, validationMetrics, finalValidationMetrics)
```
