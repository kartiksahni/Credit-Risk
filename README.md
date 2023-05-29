# Credit Card Default Prediction

This code performs credit card default prediction using logistic regression and WoE (Weight of Evidence) transformation. It follows the following steps:

1. Import necessary libraries: pandas, numpy, matplotlib.pyplot, and seaborn.
2. Load the dataset from the file 'default of credit card clients.csv'.
3. Explore the dataset to understand its characteristics.
4. Analyze variable distributions using histograms.
5. Identify missing values in the dataset.
6. Preprocess categorical variables:
   - Bin 'LIMIT_BAL' into different categories.
   - Bin 'EDUCATION' into categories: graduate school, university, high school, and others.
   - Bin 'MARRIAGE' into categories: married, single, and others.
   - Bin 'AGE' into different age groups.
   - Bin 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' into categories: on time, 30DPD, 60DPD, and >=90DPD.
   - Bin 'BILL_AMT1' - 'BILL_AMT6' into categories based on percentiles.
   - Bin 'PAY_AMT1' - 'PAY_AMT6' into categories: <2000, 2000-5000, 5000-10000, 10000-20000, and >20000.
7. Calculate WoE (Weight of Evidence) and IV (Information Value) for each binned variable.
8. Plot the WoE and IV for each binned variable.
9. Select variables with IV > 0.02.
10. Replace binned variables with the WoE values.
11. Split the dataset into training and testing sets.
12. Apply logistic regression to the training data.
13. Predict the target variable for the testing data.
14. Evaluate the model by calculating accuracy and AUC.
15. Plot the ROC curve.

## Dependencies

- pandas
- numpy
- matplotlib.pyplot
- seaborn
- sklearn

