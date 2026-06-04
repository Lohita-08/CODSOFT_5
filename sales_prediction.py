import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("advertising.csv")

print("\n===== FIRST 5 RECORDS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# ==========================
# CHECK MISSING VALUES
# ==========================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ==========================
# CORRELATION HEATMAP
# ==========================

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# ==========================
# TV VS SALES GRAPH
# ==========================

plt.figure(figsize=(8, 6))
sns.scatterplot(x="TV", y="Sales", data=df)
plt.title("TV Advertising vs Sales")
plt.show()

# ==========================
# FEATURES & TARGET
# ==========================

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# MODEL ACCURACY
# ==========================

r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("R2 Score:", r2)

# ==========================
# ACTUAL VS PREDICTED GRAPH
# ==========================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# ==========================
# USER INPUT PREDICTION
# ==========================

print("\n===== SALES PREDICTION =====")

tv = float(input("Enter TV budget: "))
radio = float(input("Enter Radio budget: "))
news = float(input("Enter Newspaper budget: "))

new_data = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [news]
})

prediction = model.predict(new_data)

print("\nPredicted Sales:", round(prediction[0], 2))