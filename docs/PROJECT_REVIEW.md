# What changed from the class project

The original project has a useful starting point: a clear churn question, stratified splitting, mixed-feature preprocessing and a random forest. The portfolio version keeps that idea and makes the experiment easier to reproduce and defend.

## Corrections

| Original issue | Change in this version | Why it matters |
| --- | --- | --- |
| The final tree cell reloads the workbook and keeps `Churn Score` and `CLTV`. | One explicit list of 19 allowed inputs is used everywhere. | Churn Score is another model's output. Its provenance and availability are not established for this experiment. CLTV is also excluded because its construction and timing are unclear. |
| The ROC and score-density cell uses training predictions. | ROC, precision–recall, calibration and error figures use held-out test predictions. | Training performance can give an optimistic impression. |
| The notebook evaluates a threshold of 0.35 on the test set. | A separate validation set chooses the threshold before test evaluation. | Choosing settings from test results makes the final evaluation less independent. |
| Missing values are filled before splitting. | Imputation is learned inside each fitted preprocessing pipeline and CV fold. | Validation and test data do not contribute medians or category modes. |
| The report and notebook mix original and tuned model results. | Tables, figures and written results come from one saved experiment. | A reader can trace each result back to the same model and split. |
| Random forest is the only main model. | A majority-class baseline and logistic regression are compared with it. | Complexity has to earn its place. A simple model may perform better. |
| City and numeric ZIP code remain in the main model. | Location fields are excluded from the first portfolio version. | ZIP codes are identifiers, not ordered quantities. Sparse local categories also make generalization harder to assess. |
| Importance comes from the original forest's impurity scores. | Permutation importance evaluates original columns in the selected pipeline. | It is easier to interpret and avoids favoring high-cardinality features, though correlations remain a limitation. |
| Broad warnings are suppressed and cells repeat splits and model setup. | Functions own loading, splitting, training, threshold selection and evaluation. | Errors surface clearly, and the notebook can run from a fresh kernel. |

## Writing changes

The class report includes website engagement variables in the midterm summary, but those variables are not in the Telco workbook. Those claims are removed. The revised write-up also removes grading language and unsupported statements that the model has already improved retention or saved revenue.

The reported confusion matrix has 280 true positives and 94 false negatives, which imply recall of about 0.749, rather than 0.70. The revised report uses freshly calculated values throughout. Historical class results and new results should not be compared as a controlled improvement because the feature set, split and evaluation protocol changed.

IBM defines Churn Score as an SPSS Modeler output. It is too strong to state that this field is necessarily measured after churn; the reason for excluding it here is the unverified score provenance and timing. CLTV is excluded on the same conservative availability principle, rather than treated as proven leakage.

## What this project can honestly claim

It compares churn classifiers, documents feature exclusions, chooses an operating threshold independently of the test labels and quantifies a retention shortlist on a fictional sample. It does not prove that a customer will churn in a future period or that an offer will prevent churn. There are no dated observations to establish a future prediction window, and no treatment outcomes to estimate the effect of retention offers.

For a production extension, define a scoring date and prediction horizon, collect dated customer snapshots, evaluate on later periods, audit subgroup performance, and test retention offers experimentally.
