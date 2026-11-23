# predict.py
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
# STEP 1: Load the saved model
def load_model(model_path=&quot;best_model.joblib&quot;):
model = joblib.load(model_path)
return model
# STEP 2: Create a reusable prediction function
def make_prediction(model, input_data):
prediction = model.predict(input_data)
probabilities = model.predict_proba(input_data)
result = &quot;Malignant&quot; if prediction[0] == 0 else &quot;Benign&quot;
return {
&quot;prediction&quot;: result,
&quot;probabilities&quot;: {
&quot;malignant&quot;: probabilities[0][0],
&quot;benign&quot;: probabilities[0][1]
}
}
# STEP 3: Run a sample prediction when executed directly
if __name__ == &quot;__main__&quot;:
model = load_model()
data = load_breast_cancer()
feature_names = data.feature_names
sample_input = dict(zip(feature_names, data.data[0]))
# Modify a few values for demonstration
sample_input[&quot;mean radius&quot;] = 15.0
sample_input[&quot;mean texture&quot;] = 18.0
sample_input[&quot;mean perimeter&quot;] = 100.0
input_df = pd.DataFrame(
[sample_input],
columns=feature_names
)
output = make_prediction(model, input_df)
print(&quot;Model loaded successfully!&quot;)
print(&quot;\nPrediction:&quot;, output[&quot;prediction&quot;])

print(&quot;\nPrediction Probabilities:&quot;)
print(&quot;Malignant:&quot;, output[&quot;probabilities&quot;][&quot;malignant&quot;])
print(&quot;Benign:&quot;, output[&quot;probabilities&quot;][&quot;benign&quot;])
