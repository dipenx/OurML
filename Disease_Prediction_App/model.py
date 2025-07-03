# stress_app/train_combined_models.py
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os

# === STRESS DETECTION MODEL ===
print("\n📦 Loading stress detection dataset...")
stress_df = pd.read_csv('dataset/stress_detection_dataset.csv')

stress_features = [
    'Feeling_Overwhelmed', 'Sleep_Quality', 'Mood_Swings', 'Daily_Work_Hours',
    'Difficulty_Concentrating', 'Fatigue_Headache', 'Anxiety_Frequency',
    'Social_Isolation', 'Appetite_Changes', 'Physical_Activity'
]

stress_df['Stress_Level'] = stress_df['Stress_Level'].map({
    'Low': 0,
    'Moderate': 1,
    'High': 2
})

X_stress = stress_df[stress_features]
y_stress = stress_df['Stress_Level']

X_train_stress, X_test_stress, y_train_stress, y_test_stress = train_test_split(X_stress, y_stress, test_size=0.2, random_state=42)

print("🌲 Training Random Forest for Stress Detection...")
stress_model = RandomForestClassifier(n_estimators=100, random_state=42)
stress_model.fit(X_train_stress, y_train_stress)

joblib.dump(stress_model, 'stress_model.pkl')
print("✅ Stress model saved as stress_model.pkl")

# === DISEASE PREDICTION MODEL ===
print("\n📦 Loading disease prediction dataset...")
disease_df = pd.read_csv("dataset/Training.csv")
disease_df.fillna(0, inplace=True)

X_disease = disease_df.drop(columns=["prognosis"])
y_disease = disease_df["prognosis"]

print("🔤 Encoding disease labels...")
disease_label_encoder = LabelEncoder()
y_disease_encoded = disease_label_encoder.fit_transform(y_disease)
joblib.dump(disease_label_encoder, "label_encoder.pkl")

print("📊 Scaling disease features...")
disease_scaler = StandardScaler()
X_disease_scaled = disease_scaler.fit_transform(X_disease)
joblib.dump(disease_scaler, "scaler.pkl")

X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(
    X_disease_scaled, y_disease_encoded, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train_disease, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_disease, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_disease, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_disease, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

class DiseasePredictionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DiseasePredictionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

input_size = X_train_disease.shape[1]
output_size = len(np.unique(y_disease_encoded))
disease_model = DiseasePredictionModel(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(disease_model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
disease_model.to(device)

print("🚀 Training disease prediction DNN...")
num_epochs = 100
for epoch in range(num_epochs):
    disease_model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = disease_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

print("🧪 Evaluating disease model...")
disease_model.eval()
y_preds, y_true = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = disease_model(batch_X)
        y_pred = torch.argmax(outputs, axis=1).cpu().numpy()
        y_preds.extend(y_pred)
        y_true.extend(batch_y.numpy())

accuracy = accuracy_score(y_true, y_preds)
print(f"✅ Disease Model Accuracy: {accuracy * 100:.2f}%")

MODEL_PATH = "disease_model_dnn.pth"
torch.save(disease_model.state_dict(), MODEL_PATH)
print(f"✅ Disease model saved at {MODEL_PATH}")