# Ex.No: 13 Mini Project
### DATE: 4.11.2024                                                                 
### REGISTER NUMBER : 212222040034

### AIM:
To write a program to train a classifier for bus delay prediction using randomforest classifier.

### Algorithm:
1. Import necessary libraries 
2. Load the dataset of images representing bus delay
3. Define the EfficientNet-B0 model and add a classification head to it
4. Create data loaders for training and validation sets
5. Train the model using the Adam optimizer and cross-entropy loss function
6. Monitor the model's performance on the validation set during training


### Program:
```import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('chennai_bus_delay_prediction.csv')

# Encode categorical variables if needed
label_encoder = LabelEncoder()
data['Day of Week'] = label_encoder.fit_transform(data['Day of Week'])
data['Weather Condition'] = label_encoder.fit_transform(data['Weather Condition'])
data['Traffic Condition'] = label_encoder.fit_transform(data['Traffic Condition'])

# Features: Select relevant columns based on your dataset
X = data[['Delay (in Minutes)', 'Traffic Condition', 'Weather Condition', 'Day of Week']]
y = data['Incident Detected']

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to search for bus data by bus number
def search_bus_data(bus_number):
    # Filter the data for the specific bus number
    bus_data = data[data['Bus Number'] == bus_number]

    if bus_data.empty:
        print(f"No data found for Bus Number: {bus_number}")
    else:
        # Display only relevant information for the specified bus number
        for index, row in bus_data.iterrows():
            print(f"Bus Number: {row['Bus Number']}, Route ID: {row['Route ID']}, "
                  f"Starting Point: {row['Starting Point']}, Ending Point: {row['Ending Point']}, "
                  f"Predicted Arrival Time: {row['Predicted Arrival Time']}, "
                  f"Delay (in Minutes): {row['Delay (in Minutes)']}, "
                  )

# Main program to prompt user for input
if __name__ == "__main__":
    bus_number = input("Enter the bus number: ")  # Prompt user to enter a bus number
    search_bus_data(bus_number)  # Search and display the bus data

```


### Output:

![image](https://github.com/user-attachments/assets/4ac5c902-2826-4749-838e-f02cf1549cda)

The Model reached an accuracy of 97.5% after 10 epochs against the test dataset.


### Result:
Thus the system was trained successfully and the prediction was carried out.
