import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title = "Titanic Survival Prediction", page_icon = ":ship:")

st.title("Titanic Survival Prediction :ship:")

# Load the dataset
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    return df

df = load_data()

# Display the dataset
st.header("Titanic Dataset")
st.write(df.head())

# User input features
st.header("User Input Features")

# Options
Pclass_1 = st.selectbox("Passenger Class", ["1st", "2nd", "3rd"])
Sex_1 = st.selectbox("Sex", ["Male", "Female"])
Age = st.slider("Age", 0, 100, 18)
SibSp = st.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.slider("Number of Parents/Children Aboard", 0, 10, 0)
Fare = st.slider("Fare", 0, 300, 50)
model_type = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Decision Tree"])
cr = st.checkbox("View Classification Report")

# Preprocess user input
def preprocess_input(Pclass_1, Sex_1, Age, SibSp, Parch, Fare):
    if Pclass_1 == "1st":
        Pclass = 1
    elif Pclass_1 == "2nd":
        Pclass = 2
    else:
        Pclass = 3

    if Sex_1 == "male":
        Sex = 0
    else:
        Sex = 1

    return Pclass, Sex, Age, SibSp, Parch, Fare

# Predict function
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare):
    # Load data
    df = load_data()
    
    # Preprocess input
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    if model_type == "Random Forest":
        model = RandomForestClassifier()
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        return None

    model.fit(X_train, y_train)

    # Make prediction
    prediction = model.predict([[Pclass, Sex, Age, SibSp, Parch, Fare]])
    prediction = model.predict(X_test)
    
    if cr = "True":
        report = classification_report(y_test, prediction)
        st.subheader("Classification Report:")
        st.write(report)
    
# Display prediction
if st.button("Predict"):
    Pclass, Sex, Age, SibSp, Parch, Fare = preprocess_input(Pclass_1, Sex_1, Age, SibSp, Parch, Fare)
    prediction = predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare)
    st.subheader("Prediction:")
    st.write("Survived" if prediction[0] == 1 else "Not Survived")
