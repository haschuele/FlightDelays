# Modeling Snippets

## Linear Regression

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
```
