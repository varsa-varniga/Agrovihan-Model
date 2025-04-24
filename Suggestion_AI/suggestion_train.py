import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os

# Load dataset
df = pd.read_csv("Datas/tamilnadu_crop_suggestions_detailed.csv")

# Rename columns for consistency
df.columns = ['crop_type', 'phase', 'temperature', 'humidity', 'rain', 'wind_speed', 'suggestion', 'category']

# Encode target labels
le = LabelEncoder()
df['suggestion_encoded'] = le.fit_transform(df['suggestion'])

# Save the label encoder
if not os.path.exists('model'):
    os.makedirs('model')

with open("model/suggestion_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Filter out rare classes with <2 samples
label_counts = df['suggestion_encoded'].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df_filtered = df[df['suggestion_encoded'].isin(valid_labels)]

# Relabel the target classes to be continuous integers starting from 0
df_filtered['suggestion_encoded'] = le.fit_transform(df_filtered['suggestion'])

# Features and target
features = df_filtered[['crop_type', 'phase', 'temperature', 'humidity', 'rain', 'wind_speed']]
features_encoded = pd.get_dummies(features)

# Save the encoded column names
with open("model/columns.pkl", "wb") as f:
    pickle.dump(features_encoded.columns.tolist(), f)

target = df_filtered['suggestion_encoded']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, target, test_size=0.25, random_state=42, stratify=target)

# Normalize features for better performance with certain models (e.g., XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Improved Random Forest with advanced hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

rf = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
rf_cv.fit(X_train_scaled, y_train)

best_rf_model = rf_cv.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_scaled)

print("✅ Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\n📊 Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# Cross-validation score
cv_scores_rf = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5)
print("📈 Random Forest CV Accuracy:", cv_scores_rf.mean())

# Save the best RF model
with open("model/agrovihan_model_rf.pkl", "wb") as f:
    pickle.dump(best_rf_model, f)

# Try XGBoost with improved hyperparameters
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(df_filtered['suggestion_encoded'].unique()),
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=300,
    learning_rate=0.05,  # Lower learning rate
    max_depth=12,  # Increased depth for more complex patterns
    subsample=0.8,  # Prevent overfitting by sampling
    colsample_bytree=0.8,  # Use feature subsampling
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

print("\n🚀 XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("\n📊 XGBoost Report:\n", classification_report(y_test, xgb_pred))

# Save the XGBoost model
with open("model/agrovihan_model_xgb.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
