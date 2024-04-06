# Define the Titanic dataset
titanic_data = [
    # (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Survived)
    (3, 'male', 22.0, 1, 0, 7.25, 'S', 0),
    (1, 'female', 38.0, 1, 0, 71.2833, 'C', 1),
    (3, 'female', 26.0, 0, 0, 7.925, 'S', 1),
    (1, 'female', 35.0, 1, 0, 53.1, 'S', 1),
    (3, 'male', 35.0, 0, 0, 8.05, 'S', 0),
    # Add more data here...
]

# Function to preprocess data
def preprocess_data(data):
    processed_data = []
    for row in data:
        pclass, sex, age, sibsp, parch, fare, embarked, survived = row
        # Convert sex to numerical value (male: 0, female: 1)
        sex = 1 if sex == 'female' else 0
        # Convert embarked to numerical value ('S': 0, 'C': 1, 'Q': 2)
        embarked = {'S': 0, 'C': 1, 'Q': 2}.get(embarked, 0)
        processed_data.append((pclass, sex, age, sibsp, parch, fare, embarked, survived))
    return processed_data

# Function to train the model
def train_model(data):
    # Simple rule-based model:
    # Women (sex=1) and children (age < 18) from first and second class are predicted to survive
    model = lambda row: 1 if (row[1] == 1 or row[2] < 18) and row[0] in [1, 2] else 0
    return model

# Function to test the model
def test_model(model, data):
    correct_predictions = 0
    total_predictions = len(data)
    for row in data:
        features, label = row[:-1], row[-1]
        prediction = model(features)
        if prediction == label:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy

# Preprocess the Titanic dataset
processed_titanic_data = preprocess_data(titanic_data)

# Train the model
model = train_model(processed_titanic_data)

# Test the model
accuracy = test_model(model, processed_titanic_data)
print("Model accuracy:", accuracy)
