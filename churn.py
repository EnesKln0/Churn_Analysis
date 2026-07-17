import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, roc_curve

# Ignore future warnings for cleaner output
warnings.filterwarnings('ignore')

print("[INFO] Loading and preprocessing data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values and correct data types
# Step 1: Force hidden empty strings to NaN (Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# -----MISSING DATA VISUALIZATION ---
print("[INFO] Plotting missing values distribution...")
# Calculate the fraction of missing values for each variable
missing_fraction = df.isnull().sum() / len(df)
missing_df = missing_fraction.reset_index()
missing_df.columns = ['variables', 'percent_missing']

# Sort the values for better visualization in the bar plot
missing_df = missing_df.sort_values(by='percent_missing', ascending=True)

# Plot the graph
plt.figure(figsize=(10, 6))
sns.barplot(x='percent_missing', y='variables', data=missing_df, color='red')

# Design and layout settings
plt.title('Percentage of Missing Values per Variable', fontsize=14, fontweight='bold')
plt.xlabel('percent_missing', fontsize=12)
plt.ylabel('variables', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# --------------------------------------------------------

# Step 2: Drop the missing values and unnecessary columns after showing the plot
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Feature Engineering
replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in replace_cols:
    df[col] = df[col].replace({'No internet service': 'No'})

df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

# Grouping the continuous tenure variable into structural time blocks
def group_tenure(t):
    if t <= 12: return '0-12 Months'
    elif t <= 24: return '12-24 Months'
    elif t <= 48: return '24-48 Months'
    elif t <= 60: return '48-60 Months'
    else: return '> 60 Months'

df['tenure_group'] = df['tenure'].apply(group_tenure)
df.drop('tenure', axis=1, inplace=True)

# Keep a copy for Exploratory Data Analysis (EDA)
df_eda = df.copy()

# Map target variable and apply One-Hot Encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

# Split features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']
print("[SUCCESS] Data is ready for the model. Number of features:", X.shape[1])

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("[INFO] Generating Comprehensive Exploratory Data Analysis (EDA)...")

# Grafikler için verinin orijinal halini çekiyoruz
df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
df_raw.dropna(subset=['TotalCharges'], inplace=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 1. OVERALL CHURN PERCENTAGE
# ==========================================
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Churn', data=df_raw, palette=['#ff5722', '#ffc107'])
plt.title('Overall Churn Percentage', fontsize=16, fontweight='bold')
plt.xlabel('Churn Status', fontsize=12)
plt.ylabel('Count', fontsize=12)

total = len(df_raw)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.2f}%'
    # Değişken isimlerini x_pos ve y_pos yaparak 'y' tablomuzun ezilmesini önledik
    x_pos = p.get_x() + p.get_width() / 2
    y_pos = p.get_height() + 50
    ax.annotate(percentage, (x_pos, y_pos), ha='center', fontsize=12, fontweight='bold')
plt.show()

# ==========================================
# 2. NUMERICAL VARIABLES vs CHURN
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Distribution of Numeric Variables by Churn', fontsize=16, fontweight='bold')

sns.boxplot(x='Churn', y='tenure', data=df_raw, palette='Set2', ax=axes[0])
axes[0].set_title('Tenure vs Churn', fontsize=14)

sns.boxplot(x='Churn', y='MonthlyCharges', data=df_raw, palette='Set2', ax=axes[1])
axes[1].set_title('Monthly Charges vs Churn', fontsize=14)

sns.boxplot(x='Churn', y='TotalCharges', data=df_raw, palette='Set2', ax=axes[2])
axes[2].set_title('Total Charges vs Churn', fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# 3. CATEGORICAL VARIABLES vs CHURN
# ==========================================
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'InternetService', 'Contract']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Categorical Variables vs Churn Target', fontsize=18, fontweight='bold')

for i, col in enumerate(categorical_cols):
    row = i // 3
    col_idx = i % 3
    sns.countplot(x=col, hue='Churn', data=df_raw, palette='pastel', ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} vs Churn', fontsize=14)
    axes[row, col_idx].set_xlabel('')

plt.tight_layout()
plt.show()

# ==========================================
# 4. CORRELATION HEATMAP
# ==========================================
plt.figure(figsize=(8, 6))
numeric_df = df_raw[['tenure', 'MonthlyCharges', 'TotalCharges']]
corr_matrix = numeric_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
            annot_kws={"size": 12, "weight": "bold"})
plt.title('Correlation Matrix of Numeric Variables', fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Clip to prevent overflow in exp
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        y = np.array(y)

        # Gradient Descent loop
        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / m) * np.dot(X.T, (y_predicted - y))
            db = (1 / m) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_predicted = self.predict_proba(X)
        return np.array([1 if i > threshold else 0 for i in y_predicted])

print("[SUCCESS] Custom Logistic Regression class defined successfully.")

# Split the dataset into train and test sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_df = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_df = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Convert to numpy arrays for the custom model
X_train_np = X_train_df.values
X_test_np = X_test_df.values
y_train_np = y_train.values
y_test_np = y_test.values

print("[INFO] Training the custom model (This might take a few seconds)...")
model = CustomLogisticRegression(learning_rate=0.1, num_iterations=4000)
model.fit(X_train_np, y_train_np)
print("[SUCCESS] Model training completed!")

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print("[INFO] Calculating Optimal Probability Cutoff...")

# Get predicted probabilities from the model (values between 0.0 and 1.0)
y_prob = model.predict_proba(X_test_np)

# Create an array of threshold values to test, from 0.0 to 1.0 with a step of 0.01
thresholds = np.arange(0.0, 1.0, 0.01)
accuracies, sensitivities, specificities = [], [], []

# Calculate evaluation metrics for each threshold
for t in thresholds:
    # Classify as 1 (Churn) if probability is greater than or equal to the threshold
    y_pred_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred_t).ravel()

    # Calculate Accuracy, Sensitivity (Recall), and Specificity
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0  # Ability to correctly identify churners
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0  # Ability to correctly identify non-churners

    accuracies.append(acc)
    sensitivities.append(sens)
    specificities.append(spec)

# Find the optimal threshold where Sensitivity and Specificity intersect (minimum difference)
diffs = np.abs(np.array(sensitivities) - np.array(specificities))
optimal_idx = np.argmin(diffs)
optimal_threshold = thresholds[optimal_idx]

print(f"[SUCCESS] Optimal Cutoff found: {optimal_threshold:.2f}")

# ==========================================
# 1. OPTIMAL CUTOFF GRAPH (Intersection)
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, label='Accuracy', color='blue', lw=2)
plt.plot(thresholds, sensitivities, label='Sensitivity', color='red', lw=2)
plt.plot(thresholds, specificities, label='Specificity', color='green', lw=2)

# Draw a vertical line to highlight the optimal cutoff point
plt.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Optimal Cutoff = {optimal_threshold:.2f}')

plt.xlim([0.0, 0.8])
plt.xlabel('Probability Cutoff', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Finding the Optimal Probability Cutoff', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==========================================
# 2. FINAL DASHBOARD WITH OPTIMAL CUTOFF
# ==========================================
print(f"\n[INFO] Generating Final Dashboard using Optimal Cutoff ({optimal_threshold:.2f})...")

# Generate final predictions using the optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

print("="*45)
print(f"  CLASSIFICATION REPORT (Cutoff = {optimal_threshold:.2f})  ")
print("="*45)
print(classification_report(y_test_np, y_pred_optimal, target_names=['Stay (0)', 'Churn (1)']))

# Calculate Confusion Matrix and AUC Score
conf_matrix = confusion_matrix(y_test_np, y_pred_optimal)
test_auc = roc_auc_score(y_test_np, y_prob)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f'Model Evaluation Dashboard (Optimal Cutoff: {optimal_threshold:.2f})', fontsize=18, fontweight='bold', y=1.05)

# --- Subplot 1: Confusion Matrix ---
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: Stay', 'Pred: Churn'],
            yticklabels=['Actual: Stay', 'Actual: Churn'],
            annot_kws={"size": 14}, ax=axes[0])
axes[0].set_title('Confusion Matrix', fontsize=14)

# --- Subplot 2: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test_np, y_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Our Model (AUC = {test_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve', fontsize=14)
axes[1].legend(loc="lower right")

# --- Subplot 3: Feature Importances ---
importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.weights})
importances['Abs_Importance'] = importances['Importance'].abs()
top_features = importances.sort_values(by='Abs_Importance', ascending=False).head(10)

sns.barplot(x='Importance', y='Feature', data=top_features, ax=axes[2], palette='coolwarm')
axes[2].set_title('Top 10 Feature Importances', fontsize=14)

plt.tight_layout()
plt.show()

import numpy as np

class CustomSVC:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # SVM expects labels to be -1 and 1 instead of 0 and 1
        y_transformed = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condition: y_i * (w*x_i - b) >= 1
                condition = y_transformed[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    # Only apply regularization gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Apply both regularization and hinge loss gradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_transformed[idx]))
                    self.b -= self.lr * y_transformed[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.where(approx >= 0, 1, 0)

    def predict_proba(self, X):
        """
        SVM doesn't naturally give probabilities.
        We use a Sigmoid function (Platt Scaling) to squash the output between 0 and 1.
        """
        decision_function = np.dot(X, self.w) - self.b
        # Squashing values into [0, 1] for our cutoff analysis
        return 1 / (1 + np.exp(-decision_function))

print("[SUCCESS] CustomSVC class defined with Gradient Descent.")

# --- STEP 1: TRAIN ---
np.random.seed(42)
# (Raising iterations to 5000 so your custom math actually has time to learn)
my_svc = CustomSVC(learning_rate=0.0001, lambda_param=0.01, n_iters=5000)
my_svc.fit(X_train_np, y_train_np)

# --- STEP 2: GET PROBABILITIES ---
# We need these numbers for the AUC and the Optimal Cutoff graph later
custom_svc_probs = my_svc.predict_proba(X_test_np)
custom_svc_preds = my_svc.predict(X_test_np)

# --- STEP 3: THE EVALUATION (The "Alper-Level" Proof) ---
print(f"--- CUSTOM SVC PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test_np, custom_svc_preds):.4f}")
print(f"AUC Score: {roc_auc_score(y_test_np, custom_svc_probs):.4f}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- Calculation for SVC Optimal Cutoff ---
thresholds = np.arange(0.0, 1.0, 0.01)
sens_svm, spec_svm = [], []

for t in thresholds:
    y_pred_t = (custom_svc_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred_t).ravel()
    sens_svm.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    spec_svm.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

# Find the optimal threshold where Sensitivity and Specificity are closest
opt_cutoff_svm = thresholds[np.argmin(np.abs(np.array(sens_svm) - np.array(spec_svm)))]

# ==========================================
# SVC OPTIMAL CUTOFF DASHBOARD
# ==========================================
plt.figure(figsize=(10, 6))

# Plotting the metrics
plt.plot(thresholds, sens_svm, label='Sensitivity (Catch Churners)', color='red', lw=2)
plt.plot(thresholds, spec_svm, label='Specificity (Identify Stayers)', color='green', lw=2)

# Marking the optimal point
plt.axvline(x=opt_cutoff_svm, color='black', linestyle='--',
            label=f'Optimal Cutoff = {opt_cutoff_svm:.2f}')

plt.title('Custom SVC: Optimal Probability Threshold', fontsize=14, fontweight='bold')
plt.xlabel('Probability Cutoff')
plt.ylabel('Metric Score')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.show()

print(f"SVC Optimal Cutoff is confirmed at: {opt_cutoff_svm:.2f}")

# Apply the "Optimal Cutoff" found earlier
final_preds = (custom_svc_probs >= opt_cutoff_svm).astype(int)

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test_np, final_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
plt.title(f"Custom SVC Confusion Matrix (Cutoff: {opt_cutoff_svm:.2f})")
plt.show()

"""### **Advanced SVM Evaluation: Calibration & Hyperparameter Tuning**
We will check if our SVM probabilities are well-calibrated and how the `lambda_param` affects the performance.
"""

from sklearn.calibration import calibration_curve

# 1. SVM Calibration Curve
prob_true_svm, prob_pred_svm = calibration_curve(y_test_np, custom_svc_probs, n_bins=10)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(prob_pred_svm, prob_true_svm, marker='s', label='Custom SVM', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('SVM Calibration (Reliability Diagram)')
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Ratio')
plt.grid(True, alpha=0.3)

# 2. Hyperparameter Test: Lambda
lambdas = [0.0001, 0.001, 0.01, 0.1]
svm_scores = []
for l in lambdas:
    t_svm = CustomSVC(learning_rate=0.0001, lambda_param=l, n_iters=1000)
    t_svm.fit(X_train_np, y_train_np)
    svm_scores.append(roc_auc_score(y_test_np, t_svm.predict_proba(X_test_np)))

plt.subplot(1, 2, 2)
plt.plot(lambdas, svm_scores, marker='o', color='green')
plt.xscale('log')
plt.title('Effect of Lambda on SVM AUC')
plt.xlabel('Lambda')
plt.ylabel('AUC Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class CustomDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # SAFETY CHECK 1: If there is no data, return a dummy leaf
        if n_samples == 0:
            return Node(value=0)

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # SAFETY CHECK 2: If no valid split was found (best_feat is None)
        if best_feat is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

class CustomRandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = CustomDecisionTree(max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      n_features=self.n_features)
            # Create a bootstrap sample
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(preds).most_common(1)[0][0] for preds in tree_preds]
        return np.array(y_pred)

    def predict_proba(self, X):
        # Probability is the fraction of trees that voted "1"
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Average of the "1" votes
        return np.mean(tree_preds, axis=1)

print("[SUCCESS] CustomRandomForest committee is ready.")

# --- 1. TRAIN THE COMMITTEE ---
print("[INFO] Growing your custom forest (this may take a minute)...")
np.random.seed(42)
my_forest = CustomRandomForest(n_trees=20, max_depth=10)
my_forest.fit(X_train_np, y_train_np)

# --- 2. GET PROBABILITIES ---
my_forest_probs = my_forest.predict_proba(X_test_np)
my_forest_preds = my_forest.predict(X_test_np)

# --- 3. THE GRADE ---
from sklearn.metrics import accuracy_score, roc_auc_score

print(f"--- CUSTOM RANDOM FOREST PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test_np, my_forest_preds):.4f}")
print(f"AUC Score: {roc_auc_score(y_test_np, my_forest_probs):.4f}")

# ==========================================
# RANDOM FOREST OPTIMAL CUTOFF DASHBOARD
# ==========================================
thresholds = np.arange(0.0, 1.0, 0.01)
accuracies, sensitivities, specificities = [], [], []

for t in thresholds:
    y_pred_t = (my_forest_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred_t).ravel()
    accuracies.append((tp + tn) / (tp + tn + fp + fn))
    sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

# Find intersection point
opt_cutoff_rf = thresholds[np.argmin(np.abs(np.array(sensitivities) - np.array(specificities)))]

plt.figure(figsize=(10, 5))
plt.plot(thresholds, sensitivities, label='Sensitivity (Catch Churn)', color='red', lw=2.5)
plt.plot(thresholds, specificities, label='Specificity (Identify Stay)', color='green', lw=2.5)
plt.axvline(x=opt_cutoff_rf, color='black', linestyle='--', label=f'Optimal Cutoff = {opt_cutoff_rf:.2f}')
plt.title('Random Forest Optimization Dashboard', fontsize=14, fontweight='bold')
plt.xlabel('Probability Threshold')
plt.ylabel('Metric Score')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 1. Apply the Forest's cutoff
rf_final_preds = (my_forest_probs >= opt_cutoff_rf).astype(int)

# 2. Plot the Final "Committee" Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm_rf = confusion_matrix(y_test_np, rf_final_preds)

plt.figure(figsize=(5,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
plt.title(f"Custom Random Forest Confusion Matrix (Cutoff: {opt_cutoff_rf:.2f})")
plt.show()

"""### **Advanced Random Forest Evaluation: Precision-Recall & Tree Count**
Comparing the Forest's ability to handle class imbalance and determining if more trees would improve the result.
"""

from sklearn.metrics import precision_recall_curve, average_precision_score

# 1. Precision-Recall Curve for Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test_np, my_forest_probs)
avg_p_rf = average_precision_score(y_test_np, my_forest_probs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(recall_rf, precision_rf, color='orange', lw=2, label=f'AP={avg_p_rf:.2f}')
plt.fill_between(recall_rf, precision_rf, alpha=0.2, color='orange')
plt.title('Random Forest Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# 2. Sensitivity to Forest Size (n_trees)
tree_counts = [5, 10, 20, 30]
forest_scores = []
for n in tree_counts:
    t_rf = CustomRandomForest(n_trees=n, max_depth=5)
    t_rf.fit(X_train_np, y_train_np)
    forest_scores.append(accuracy_score(y_test_np, t_rf.predict(X_test_np)))

plt.subplot(1, 2, 2)
plt.plot(tree_counts, forest_scores, marker='^', color='orange')
plt.title('Effect of n_trees on Forest Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np

class CustomANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.train_loss_history = []
        self.val_loss_history = []

        # He Initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def _compute_loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        y_train = y_train.reshape(-1, 1)
        if y_val is not None: y_val = y_val.reshape(-1, 1)

        for epoch in range(self.epochs):
            z1 = np.dot(X_train, self.W1) + self.b1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._sigmoid(z2)
            self.train_loss_history.append(self._compute_loss(y_train, a2))

            if X_val is not None:
                val_z1 = np.dot(X_val, self.W1) + self.b1
                val_a1 = self._relu(val_z1)
                val_z2 = np.dot(val_a1, self.W2) + self.b2
                val_a2 = self._sigmoid(val_z2)
                self.val_loss_history.append(self._compute_loss(y_val, val_a2))

            m = y_train.shape[0]
            dz2 = a2 - y_train
            dW2 = (1 / m) * np.dot(a1.T, dz2)
            db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)
            dz1 = np.dot(dz2, self.W2.T) * self._relu_derivative(z1)
            dW1 = (1 / m) * np.dot(X_train.T, dz1)
            db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict_proba(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self._sigmoid(z2).flatten()

print("[SUCCESS] ANN updated with predict_proba and Loss tracking.")

# Re-train with validation data
ann_model = CustomANN(input_size=X_train_np.shape[1], hidden_size=16, output_size=1, learning_rate=0.05, epochs=3000)
ann_model.fit(X_train_np, y_train_np, X_val=X_test_np, y_val=y_test_np)

# Visualize Overfitting
plt.figure(figsize=(10, 6))
plt.plot(ann_model.train_loss_history, label='Train Loss', color='blue')
plt.plot(ann_model.val_loss_history, label='Validation Loss', color='orange', linestyle='--')
plt.title('Train vs Validation Loss (Overfitting Check)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Fix: Define ann_probs before use
if 'ann_model' in globals():
    ann_probs = ann_model.predict_proba(X_test_np)

    # Calculate and visualize optimal threshold for ANN
    thresholds = np.arange(0.0, 1.0, 0.01)
    sens_ann, spec_ann = [], []

    for t in thresholds:
        y_p = (ann_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_np, y_p).ravel()
        sens_ann.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec_ann.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    opt_cutoff_ann = thresholds[np.argmin(np.abs(np.array(sens_ann) - np.array(spec_ann)))]

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, sens_ann, label='Sensitivity (Recall)', color='red')
    plt.plot(thresholds, spec_ann, label='Specificity', color='green')
    plt.axvline(x=opt_cutoff_ann, color='black', linestyle='--', label=f'Optimal Cutoff: {opt_cutoff_ann:.2f}')
    plt.title('ANN Optimization Dashboard (Optimal Threshold)')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Final ANN Confusion Matrix using optimal threshold
    ann_final_preds = (ann_probs >= opt_cutoff_ann).astype(int)
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test_np, ann_final_preds), annot=True, fmt='d', cmap='Purples',
                xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
    plt.title(f"ANN Final Matrix (Cutoff: {opt_cutoff_ann:.2f})")
    plt.show()
else:
    print("[ERROR] ann_model not found. Please train the model first.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Checking if the model is available
if 'ann_model' in globals():
    # Get probabilities for ANN
    ann_probs = ann_model.predict_proba(X_test_np)

    # Use the optimal cutoff (0.30)
    ann_cutoff = 0.30
    ann_preds = (ann_probs >= ann_cutoff).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test_np, ann_preds)
    precision = precision_score(y_test_np, ann_preds)
    recall = recall_score(y_test_np, ann_preds)
    f1 = f1_score(y_test_np, ann_preds)

    print(f"--- ANN Performance Metrics (Cutoff: {ann_cutoff}) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_np, ann_preds))
else:
    print("[ERROR] 'ann_model' not found. Please run the definition cell (ee55d1ca) and training cell (db7cc652) first.")

"""### **ANN Hyper-parameter Optimization**
In this step, we test different `learning_rate` and `epochs` values for the model to learn faster and more stably. Our goal is to find the point where the validation loss is minimized.
"""

import matplotlib.pyplot as plt

# Parameter sets to try
learning_rates = [0.001, 0.01, 0.05, 0.1]
max_epochs = 2000
results = {}

print(f"[INFO] Optimization is starting. A total of {len(learning_rates)} different scenarios will be tested...")

plt.figure(figsize=(15, 10))

for i, lr in enumerate(learning_rates):
    print(f"--- Testing: Learning Rate = {lr} ---")

    # Initializing the model from scratch each time
    temp_ann = CustomANN(input_size=X_train_np.shape[1], hidden_size=16, output_size=1, learning_rate=lr, epochs=max_epochs)
    temp_ann.fit(X_train_np, y_train_np, X_val=X_test_np, y_val=y_test_np)

    # Store the results
    results[lr] = {
        'train_loss': temp_ann.train_loss_history,
        'val_loss': temp_ann.val_loss_history,
        'final_val_loss': temp_ann.val_loss_history[-1]
    }

    # Visualization
    plt.subplot(2, 2, i+1)
    plt.plot(temp_ann.train_loss_history, label='Train Loss')
    plt.plot(temp_ann.val_loss_history, label='Validation Loss', linestyle='--')
    plt.title(f'LR: {lr} | Final Val Loss: {temp_ann.val_loss_history[-1]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Finding the best learning rate
best_lr = min(results, key=lambda x: results[x]['final_val_loss'])
print(f"\n[RESULT] The learning rate that provides the lowest validation loss: {best_lr}")

"""### **ANN Gelişmiş Değerlendirme Metrikleri**
Sadece doğruluk (accuracy) yeterli değildir. Modelin güvenilirliğini ölçmek için Precision-Recall ve Kalibrasyon analizleri yapıyoruz.
"""

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test_np, ann_probs)
auc_score = roc_auc_score(y_test_np, ann_probs)


# 1. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test_np, ann_probs)
avg_precision = average_precision_score(y_test_np, ann_probs)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(recall, precision, color='purple', lw=2, label=f'AP={avg_precision:.2f}')
plt.fill_between(recall, precision, alpha=0.2, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ANN Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Calibration Curve (Reliability Diagram)
prob_true, prob_pred = calibration_curve(y_test_np, ann_probs, n_bins=10)

plt.subplot(1, 2, 2)
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='ANN', color='darkblue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('ANN Calibration Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. Search for the best neuron count including 16
neuron_candidates = [4, 6, 8, 10, 12, 16]
neuron_results = []

print("[INFO] Finding the best neuron count (3000 epochs) including 16... ")

for n in neuron_candidates:
    # Initialize and train from scratch for each neuron count
    temp_ann = CustomANN(input_size=X_train_np.shape[1],
                         hidden_size=n,
                         output_size=1,
                         learning_rate=0.05,
                         epochs=3000)
    temp_ann.fit(X_train_np, y_train_np, X_val=X_test_np, y_val=y_test_np)

    # Evaluate
    probs = temp_ann.predict_proba(X_test_np)
    current_auc = roc_auc_score(y_test_np, probs)
    neuron_results.append(current_auc)
    print(f"Neurons: {n} | AUC: {current_auc:.4f}")

# Create DataFrame for plotting
df_neurons = pd.DataFrame({
    'Neurons': neuron_candidates,
    'AUC Score': neuron_results
})

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Neurons', y='AUC Score', data=df_neurons, palette='viridis')

# Highlighting the best result
best_auc_val = max(neuron_results)
plt.axhline(y=best_auc_val, color='red', linestyle='--', alpha=0.6, label=f"Best AUC: {best_auc_val:.4f}")

plt.title('Effect of Neuron Count on ANN Performance (3000 Epochs)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Neurons in Hidden Layer', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)
plt.ylim(0.80, 0.84) # Scale adjusted for 16 neurons
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

print(f"The best performance was achieved with {df_neurons.loc[df_neurons['AUC Score'].idxmax(), 'Neurons']} neurons.")

"""### **ANN Hyper-parameter Optimization: Neuron Count Comparison**
In this section, we compare the performance (AUC Score) of our Custom ANN using different numbers of neurons in the hidden layer (4, 6, 8, 10, 12) with a fixed 3000 epochs.
"""