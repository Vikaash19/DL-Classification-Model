# Developing a Neural Network Classification Model

## AIM:
To develop a neural network classification model for the given dataset.

## THEORY:
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model:
![alt text](<nn model exp2.png>)

## DESIGN STEPS:
### STEP 1: Load the dataset
Load the Iris dataset using a suitable library.

### STEP 2: Preprocess the data
Preprocess the data by handling missing values and normalizing features.

### STEP 3: Split the dataset
Split the dataset into training and testing sets.

### STEP 4: Train the model
Train a classification model using the training data.

### STEP 5:  Evaluate the model
Evaluate the model on the test data and calculate accuracy.

### STEP 6: Display results
Display the test accuracy, confusion matrix, and classification report.


## PROGRAM:

### Name:Vikaash K S

### Register Number:212223240179

```python
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset,DataLoader
from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target
df=pd.DataFrame(x,columns=iris.feature_names)
df['target']=y

print("First 5 rows of dataset:",df.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=t.tensor(x_train,dtype=t.float32)
x_test=t.tensor(x_test,dtype=t.float32)
y_train=t.tensor(y_train,dtype=t.long)
y_test=t.tensor(y_test,dtype=t.long)

train_dataset=TensorDataset(x_train,y_train)
test_dataset=TensorDataset(x_test,y_test)
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=16)

class IrisClassifier(nn.Module):
  def __init__(self,input_size):
    super(IrisClassifier,self).__init__()
    self.fc1=nn.Linear(input_size,16)
    self.fc2=nn.Linear(16,8)
    self.fc3=nn.Linear(8,3)

  def forward(self,x):
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    return self.fc3(x)

def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for x_batch,y_batch in train_loader:
      optimizer.zero_grad()
      output=model(x_batch)
      loss=criterion(output,y_batch)
      loss.backward()
      optimizer.step()
    if(epoch+1)%10==0:
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
model=IrisClassifier(input_size=x_train.shape[1])
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
train_model(model,train_loader,criterion,optimizer,epochs=100)

model.eval()
predictions,actuals=[],[]
with t.no_grad():
  for x_batch,y_batch in test_loader:
    outputs=model(x_batch)
    _,predicted=t.max(outputs,1)
    predictions.extend(predicted.numpy())
    actuals.extend(y_batch.numpy())

accuracy = accuracy_score (actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=iris.target_names)
print("\nName:Vikaash K S")
print("Register No:212223240179")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix: \n", conf_matrix)
print("Classification Report:\n", class_report)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

sample_input = x_test[5].unsqueeze (0)
with t.no_grad():
  output = model(sample_input)
  predicted_class_index = t.argmax(output[0]).item()
  predicted_class_label = iris.target_names[predicted_class_index]

print("\nName:Vikaash K S")
print("Register No:212223240179")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {iris.target_names[y_test[5].item()]}')

```

### Dataset Information:
![alt text](<Dataset Information.png>)

### OUTPUT:

## Confusion Matrix:
![alt text](<Confusion matrix.png>)

## Classification Report:
![alt text](<Classification report.png>)

### New Sample Data Prediction:
![alt text](<Data prediction.png>)

## RESULT:
Thus, a neural network classification model was successfully developed and trained using PyTorch.
