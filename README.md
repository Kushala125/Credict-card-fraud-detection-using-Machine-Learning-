## The Digital Guardian: A Story of Fraud Detection
1. The Hook: The Silent War
In the world of modern finance, a transaction happens in the blink of an eye. But within that millisecond, a silent war is fought between convenience and crime. Fraudsters are constantly evolving, finding new ways to slip through the cracks. This project is about building a "Digital Guardian"—a machine learning system trained to spot the subtle patterns of deception across half a million transactions.

2. The Foundation (Exploratory Data Analysis)
Before we could build our guardian, we had to understand the battlefield. We analyzed a dataset of 568,630 transactions, each described by 31 different numerical signals.

## Key Insights from the Data:

The Perfect Balance: Our dataset was perfectly split (50/50) between legitimate and fraudulent transactions. While rare in the real world, this allowed our model to learn the "DNA" of a fraudster just as clearly as the habits of an honest user.

## “Do fraud transactions behave differently in terms of amount?”
<img width="649" height="419" alt="chart2" src="https://github.com/user-attachments/assets/5b7a29cd-fac2-4871-83f1-7951c3f73d6d" />

EDA - Transaction Amount
Do fraud and normal transactions have similar distributions? Yes, both distributions are almost identical and highly overlapping.
Do you see fraud more in high or low amounts? No clear pattern. Fraud transactions are spread across all amount ranges.
Is 'Amount' a useful feature for detecting fraud? Not very useful alone, as it does not clearly distinguish between fraud and normal transactions.

## “Which features actually influence fraud?”


<img width="675" height="523" alt="chart3" src="https://github.com/user-attachments/assets/ea00ae74-a3b1-4339-9cd9-6c61f16c6b3d" />

Correlation analysis showed that only a few features have strong relationships with fraud, indicating that fraud detection relies on complex interactions between multiple variables.”
## “Which features are MOST related to fraud?”

V4     0.735981
V11    0.724278
V2     0.491878
V19    0.244081
V27    0.214002
V20    0.179851
V8     0.144294
V21    0.109640
V28    0.102024
V26    0.071052
Name: Class, dtype: float64


“Feature correlation analysis showed that variables like V4 and V11 have strong relationships with fraud. However, since the data is PCA-transformed, individual feature interpretation is limited, and fraud detection depends on combined feature interactions.”

## “How do important features behave for fraud vs normal?”

<img width="699" height="368" alt="chart4" src="https://github.com/user-attachments/assets/4f795bc2-3e65-4774-a1c1-2b04911e5b3d" />

“Boxplot analysis of V4 showed clear separation between fraud and normal transactions, indicating it is a strong predictive feature.”

—---------
Analyze V11

<img width="702" height="387" alt="chart5" src="https://github.com/user-attachments/assets/178ed639-6614-44b5-9069-8c0b63952bfd" />

“Boxplot analysis of V11 showed strong separation between fraud and normal transactions, indicating it is one of the most important predictive features.”

## Hypothesis Testing (V4 Feature)
Null Hypothesis (H0): There is NO difference in V4 between fraud and normal transactions.
Alternative Hypothesis (H1): There IS a difference in V4 between fraud and normal transactions.
from scipy.stats import ttest_ind 
fraud = df[df['Class'] == 1]['V4']
normal = df[df['Class'] == 0]['V4']

t_stat, p_value = ttest_ind(fraud, normal)

print("T-Statistic:", t_stat)
print("P-Value:", p_value)
T-Statistic: 819.767061037011
P-Value: 0.0

## V11
fraud = df[df['Class'] == 1]['V11']
normal = df[df['Class'] == 0]['V11']

t_stat, p_value = ttest_ind(fraud, normal)

print("T-Statistic:", t_stat)
print("P-Value:", p_value)

T-Statistic: 792.1004478661719
P-Value: 0.0


“I performed hypothesis testing on key features like V4 and V11 and found p-values close to zero, confirming statistically significant differences between fraud and normal transactions.”


## 3. The Evolution of the Model
We didn't just pick one algorithm; we evolved our defense through three distinct stages:

Stage 1: The Fast Responder (Logistic Regression)
Goal: Establish a quick baseline.

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

[[55493  1257]
 [ 2699 54277]]
              precision    recall  f1-score   support

           0       0.95      0.98      0.97     56750
           1       0.98      0.95      0.96     56976

    accuracy                           0.97    113726
   macro avg       0.97      0.97      0.97    113726
weighted avg       0.97      0.97      0.97    113726


## # Logistic Regression - Model Visualization

This section visualizes:
1. Sigmoid Function (concept)
2. Probability Distribution
3. Probability vs Actual
4. Feature Importance

<img width="622" height="709" alt="chart9" src="https://github.com/user-attachments/assets/40736b53-a2a7-402a-a627-1ed9e0967930" />

<img width="678" height="661" alt="chart10" src="https://github.com/user-attachments/assets/31b1e7ef-ec90-4f42-be62-549de52c3137" />



Result: It was fast and 96.5% accurate. However, it still let over 2,600 fraudulent transactions slip through. In the world of banking, "almost" isn't good enough.

## Stage 2: The Expert (Random Forest)
Goal: Use a "forest" of decision trees to catch complex patterns.
 Random Forest Model
We train a more powerful model to improve performance.
Model:
Random Forest (Ensemble method)
Why?
Captures complex patterns
Handles non-linearity better than Logistic Regression
from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=20,   # reduce trees
    max_depth=10,      # limit depth
    random_state=42,
    n_jobs=-1          # use all CPU cores
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_model.fit(X_train, y_train)   # Train

y_pred_rf = rf_model.predict(X_test)   # Predict

print(confusion_matrix(y_test, y_pred_rf))   # Evaluate
[[56672    78]
 [ 1838 55138]]

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
[[56672    78]
 [ 1838 55138]]
              precision    recall  f1-score   support

           0       0.97      1.00      0.98     56750
           1       1.00      0.97      0.98     56976

    accuracy                           0.98    113726
   macro avg       0.98      0.98      0.98    113726
weighted avg       0.98      0.98      0.98    113726


<img width="715" height="369" alt="chart15" src="https://github.com/user-attachments/assets/5650c420-fc1c-4fc5-9a92-539d304f8eec" />


“I visualized an individual tree from the Random Forest to understand decision logic. Each node represents a split based on a feature, and the model learns hierarchical rules to classify transactions. While a single tree is interpretable, Random Forest combines multiple trees to improve accuracy and reduce overfitting.”

## Random Forest Model Insights

The Random Forest model outperformed Logistic Regression, achieving 98% accuracy.

Precision for fraud detection reached 1.00, indicating that nearly all predicted fraud cases were correct.

Recall improved to 0.97, meaning most fraud cases were successfully detected, although some were still missed.

The model significantly reduced false negatives compared to Logistic Regression, making it more suitable for real-world fraud detection.

Overall, Random Forest provides a better balance between precision and recall, making it a stronger model for this task.

## Stage 3: The Optimized Shield (Threshold Tuning)
The Breakthrough: We realized that being "neutral" (a 50% threshold) wasn't safe. By lowering our decision threshold to 0.3, we told the model: "If you are even slightly suspicious, flag it."

Final Result: This boosted our Recall to 98.8%, catching nearly every single fraudster while maintaining a near-perfect balance.

We will change threshold to catch more fraud.
# Get probabilities instead of 0/1
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Change threshold
threshold = 0.3   # try 0.3 instead of 0.5

y_pred_new = (y_prob_rf >= threshold).astype(int)

# Evaluate again
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred_new))
print(classification_report(y_test, y_pred_new))
[[56069   681]
 [  680 56296]]
              precision    recall  f1-score   support

           0       0.99      0.99      0.99     56750
           1       0.99      0.99      0.99     56976

    accuracy                           0.99    113726
   macro avg       0.99      0.99      0.99    113726
weighted avg       0.99      0.99      0.99    113726



After applying threshold tuning, the model's recall improved significantly from 0.97 to 0.99, reducing false negatives from 1838 to 680. This means the model is now able to detect a much higher number of fraud cases. Although false positives increased, this trade-off is acceptable in fraud detection because missing fraudulent transactions leads to financial loss, whereas false alerts only require manual verification. Therefore, lowering the threshold improved the model’s effectiveness in real-world scenarios.

## MODEL COMPARISON
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Logistic Regression metrics
lr_acc = accuracy_score(y_test, y_pred)
lr_prec = precision_score(y_test, y_pred)
lr_rec = recall_score(y_test, y_pred)
lr_f1 = f1_score(y_test, y_pred)

# Random Forest metrics
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf)
rf_rec = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

# Threshold Tuned RF metrics
tuned_acc = accuracy_score(y_test, y_pred_new)
tuned_prec = precision_score(y_test, y_pred_new)
tuned_rec = recall_score(y_test, y_pred_new)
tuned_f1 = f1_score(y_test, y_pred_new)

# Create comparison table
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Threshold Tuned RF"],
    "Accuracy": [lr_acc, rf_acc, tuned_acc],
    "Precision": [lr_prec, rf_prec, tuned_prec],
    "Recall": [lr_rec, rf_rec, tuned_rec],
    "F1 Score": [lr_f1, rf_f1, tuned_f1]
})

print(comparison)
comparison.set_index("Model").plot(kind="bar", figsize=(10,6))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()


                Model  Accuracy  Precision    Recall  F1 Score
0  Logistic Regression  0.965215   0.977365  0.952629  0.964839
1        Random Forest  0.983152   0.998587  0.967741  0.982922
2   Threshold Tuned RF  0.988033   0.988048  0.988065  0.988056





<img width="718" height="463" alt="chart14" src="https://github.com/user-attachments/assets/e93efe9f-b6ee-44d0-9aaa-a44123cbdc17" />

Logistic Regression provided a solid baseline with good overall performance, but its recall was slightly lower, meaning it missed some fraud cases.

Random Forest significantly improved performance by capturing complex patterns, achieving higher precision and recall.

After applying threshold tuning, the model further improved recall, reducing false negatives and detecting more fraudulent transactions.

Although precision slightly decreased, this trade-off is acceptable in fraud detection because catching fraud is more critical than avoiding false alerts.

Therefore, the threshold-tuned Random Forest model is the best choice, as it prioritizes fraud detection while maintaining strong overall performance.


## 4. Final Performance Insights
Model	Accuracy	Precision	Recall (Detection Rate)
Logistic Regression	96.5%	97.7%	95.2%
Random Forest	98.3%	99.8%	96.7%
Tuned Random Forest	98.8%	98.8%	98.8%
Graph Insight: When looking at the Comparison Bar Chart, you will notice that the "Threshold Tuned RF" provides the most stable and highest scores across all metrics, making it the most reliable choice for a bank.

EE 5. Challenges Faced
The Privacy Wall: Because the features (V1-V28) were transformed for privacy, we had to work with mathematical patterns rather than human-readable data (like location or name).

The "False Alarm" Balance: The biggest challenge was catching more fraud without accidentally blocking too many legitimate customers. Our final model found the "Sweet Spot."

## 6. Conclusion
The project proves that data-driven defense is our best weapon against financial crime. By moving from simple logic to a Threshold-Tuned Random Forest, we created a system that doesn't just look at the numbers—it understands the behavior of fraud.

The Result: A 98.8% success rate in protecting digital assets, ensuring that for every 1,000 fraud attempts, we stop 988 of them in their tracks.

