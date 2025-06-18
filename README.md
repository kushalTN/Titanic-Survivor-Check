# Titanic-Survivor-Check
This project is a machine learning mini-project based on the Titanic dataset, focused on predicting passenger survival using key features like age, class, fare, and gender. It also includes an interactive feature to enter a ticket number and instantly check whether the corresponding passenger survived or not.

# 🚢 Titanic Survival Prediction & Passenger Lookup

This project focuses on predicting the survival of passengers aboard the Titanic using machine learning techniques. It leverages the Random Forest Classifier to analyze key features such as age, fare, passenger class, and more. The project also includes insightful visualizations and an interactive feature that allows users to enter a ticket number and check whether a specific passenger survived. Built using Python, pandas, scikit-learn, seaborn, and matplotlib, this project demonstrates both analytical and practical application of data science in historical event prediction.

---

## 📂 Dataset

The dataset used is the **Titanic - Machine Learning from Disaster** dataset available on [Kaggle](https://www.kaggle.com/c/titanic).

---

## 🔍 Features

- Data cleaning and preprocessing (handling missing values, encoding)
- Exploratory Data Analysis (EDA) with visualizations
- Model training using Random Forest Classifier
- Classification report and accuracy evaluation
- Ticket number lookup: check if a passenger survived

---

## 🛠️ Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 🚀 How to Run

1. Clone this repository or download the code.
2. Ensure you have the required packages installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
Place the Titanic CSV file in the same directory (name it titanic.csv).

Run the script:

bash
Copy
Edit
python titanic_analysis.py
When prompted, enter a ticket number to check survival status.

📸 Sample Output
text
Copy
Edit
Model Accuracy: 0.97

Enter the ticket number to check survival status:
330911

Passenger: Kelly, Mr. James
Ticket: 330911
Status: Did NOT Survive 🔴
