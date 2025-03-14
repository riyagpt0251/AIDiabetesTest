# 🩺 Diabetes Prediction using Logistic Regression  

![Diabetes Prediction](https://media.giphy.com/media/QWvra259h4LCvdJnxP/giphy.gif)  

## 📌 Project Overview  
This project builds a **Logistic Regression** model to predict diabetes using the **Pima Indians Diabetes Dataset**. The dataset consists of medical attributes such as glucose levels, blood pressure, and BMI to determine whether a patient is diabetic.  

## 📊 Dataset Information  
- 📥 **Source:** [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)  
- 📌 **Features:** 8 attributes including `Pregnancies`, `Glucose`, `BloodPressure`, `BMI`, etc.  
- 🎯 **Target:** `Outcome` (1 = Diabetic, 0 = Non-Diabetic)  

---

## 🚀 Installation  

Ensure you have Python installed, then install the required dependencies:  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📂 Project Structure  
```
📁 diabetes-prediction
│── 📄 diabetes_prediction.py   # Main script
│── 📄 README.md                # Project documentation
│── 📊 diabetes_dataset.csv      # Dataset (if downloaded locally)
```

---

## 🛠️ Steps Involved  

### 1️⃣ Import Libraries  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### 2️⃣ Load Dataset  
```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)
df.head()
```

---

### 📊 Exploratory Data Analysis  
**Handling Missing Values:**  
```python
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col].replace(0, np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)
```

---

### ✂️ Splitting Data  
```python
X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 📏 Feature Scaling  
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 🔥 Training the Model  
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

### 🔍 Making Predictions  
```python
y_pred = model.predict(X_test)
```

---

### 📈 Model Evaluation  
```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

📌 **Sample Confusion Matrix:**  
![Confusion Matrix](https://i.imgur.com/ZwTHyFt.png)

---

### 🏥 Predicting Diabetes for a New Patient  
```python
def predict_diabetes(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)  
    prediction = model.predict(input_data)
    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

# Example Usage:
new_patient = [2, 120, 70, 23, 120, 28.5, 0.45, 25]  
print("\nNew Patient Prediction:", predict_diabetes(new_patient))
```

---

## 🎯 Key Findings  
✅ **Data preprocessing (handling missing values) improved model accuracy.**  
✅ **Feature scaling was essential for better model performance.**  
✅ **The model provides a simple yet effective way to predict diabetes.**  

---

## 📌 Future Improvements  
🔹 Try different **machine learning models** (e.g., Random Forest, XGBoost).  
🔹 Improve feature selection and data engineering.  
🔹 Deploy the model using Flask or Streamlit for a web-based interface.  

---

## 💡 Contributors  
👤 **Your Name** | [GitHub Profile](https://github.com/yourprofile)  

⭐ If you like this project, don't forget to give it a **star** on GitHub! ⭐  

---

🚀 **Happy Coding & Stay Healthy!** 🏥
