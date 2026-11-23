# STEP 1: Import libraries
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
f1_score,
accuracy_score,
classification_report,
confusion_matrix
)
# STEP 2: Load dataset
data = load_breast_cancer()
X = pd.DataFrame(
data.data,
columns=data.feature_names
)
y = pd.Series(data.target, name=&quot;target&quot;)
print(&quot;Dataset shape:&quot;, X.shape)
print(&quot;\nFirst 5 rows:&quot;)
print(X.head())
print(&quot;\nTarget values:&quot;)
print(y.value_counts())
# STEP 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)
print(&quot;\nTraining samples:&quot;, X_train.shape[0])
print(&quot;Testing samples:&quot;, X_test.shape[0])
# STEP 4: Define classification algorithms
models = {
&quot;Logistic Regression&quot;: LogisticRegression(
max_iter=5000
),
&quot;KNN&quot;: KNeighborsClassifier(),
&quot;Decision Tree&quot;: DecisionTreeClassifier(
random_state=42
),
&quot;Random Forest&quot;: RandomForestClassifier(
random_state=42
),
&quot;SVM&quot;: SVC()
}
# STEP 5: Define hyperparameter grids
param_grids = {
&quot;Logistic Regression&quot;: {
&quot;model__C&quot;: [0.1, 1, 10],
&quot;model__solver&quot;: [&quot;liblinear&quot;]
},
&quot;KNN&quot;: {
&quot;model__n_neighbors&quot;: [3, 5, 7, 9],
&quot;model__weights&quot;: [&quot;uniform&quot;, &quot;distance&quot;]
},
&quot;Decision Tree&quot;: {
&quot;model__max_depth&quot;: [None, 3, 5, 10],
&quot;model__min_samples_split&quot;: [2, 5, 10]
},
&quot;Random Forest&quot;: {
&quot;model__n_estimators&quot;: [100, 200],
&quot;model__max_depth&quot;: [None, 5, 10],
&quot;model__min_samples_split&quot;: [2, 5]
},
&quot;SVM&quot;: {
&quot;model__C&quot;: [0.1, 1, 10],
&quot;model__kernel&quot;: [&quot;linear&quot;, &quot;rbf&quot;],
&quot;model__gamma&quot;: [&quot;scale&quot;, &quot;auto&quot;]
}
}
# STEP 6: Run GridSearchCV for every algorithm
results = []
best_models = {}
for model_name, model in models.items():
print(&quot;\n&quot; + &quot;=&quot; * 60)
print(&quot;Running:&quot;, model_name)
print(&quot;=&quot; * 60)
pipeline = Pipeline([
(&quot;scaler&quot;, StandardScaler()),
(&quot;model&quot;, model)
])
grid_search = GridSearchCV(
estimator=pipeline,
param_grid=param_grids[model_name],
scoring=&quot;f1&quot;,
cv=5,
n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_models[model_name] = grid_search.best_estimator_
results.append({
&quot;Model&quot;: model_name,
&quot;Best F1 Score&quot;: grid_search.best_score_,
&quot;Best Parameters&quot;: grid_search.best_params_
})
print(&quot;Best F1 Score:&quot;,
round(grid_search.best_score_, 4))
print(&quot;Best Parameters:&quot;,
grid_search.best_params_)
# STEP 7: Compare all models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
by=&quot;Best F1 Score&quot;,
ascending=False
)
print(&quot;\n&quot; + &quot;=&quot; * 60)
print(&quot;MODEL COMPARISON&quot;)
print(&quot;=&quot; * 60)
print(
results_df[
[&quot;Model&quot;, &quot;Best F1 Score&quot;]
].to_string(index=False)
)
# STEP 8: Select the best model
best_model_name = results_df.iloc[0][&quot;Model&quot;]
best_model = best_models[best_model_name]
best_cv_f1 = results_df.iloc[0][&quot;Best F1 Score&quot;]
print(&quot;\nBest model based on CV F1-score:&quot;)
print(best_model_name)
print(&quot;\nBest CV F1-score:&quot;)
print(round(best_cv_f1, 4))
# STEP 9: Evaluate on unseen test data
y_pred = best_model.predict(X_test)
print(&quot;\n&quot; + &quot;=&quot; * 60)
print(&quot;FINAL MODEL EVALUATION&quot;)
print(&quot;=&quot; * 60)
print(&quot;Accuracy:&quot;, accuracy_score(y_test, y_pred))
print(&quot;F1 Score:&quot;, f1_score(y_test, y_pred))
print(&quot;\nClassification Report:&quot;)
print(
classification_report(
y_test,
y_pred,
target_names=data.target_names
)
)
print(&quot;\nConfusion Matrix:&quot;)
print(confusion_matrix(y_test, y_pred))
# STEP 10: Save the best model
joblib.dump(best_model, &quot;best_model.joblib&quot;)
print(&quot;\nBest model saved successfully!&quot;)
print(&quot;Model:&quot;, best_model_name)
