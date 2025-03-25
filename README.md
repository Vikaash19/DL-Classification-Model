# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Load the dataset
Load the Iris dataset using a suitable library.

### STEP 2: Preprocess the data
Preprocess the data by handling missing values and normalizing features.

### STEP 3: Split into training and testing sets
Split the dataset into training and testing sets.

### STEP 4: Train the model
Train a classification model using the training data.


### STEP 5:  Evaluate the model
Evaluate the model on the test data and calculate accuracy.


### STEP 6: Display results
Display the test accuracy, confusion matrix, and classification report.



## PROGRAM

### Name:

### Register Number:

```python
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        #Include your code here

    def forward(self, x):
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here

```

### Dataset Information
Include screenshot of the dataset.

### OUTPUT

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Include your result here
