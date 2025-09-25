# BMW Car Sales Classification

A machine learning project that predicts BMW car sales performance classification using XGBoost. This model analyzes various car features and market factors to classify sales as either "High" or "Low".

## 🚗 Project Overview

This project uses machine learning to predict BMW car sales classifications based on various vehicle characteristics and market conditions. The model is trained on historical BMW sales data and can help predict whether a particular BMW model configuration will have high or low sales performance.

## 📊 Features

The model considers the following features for prediction:
- **Model**: BMW car model (e.g., 3 Series, 5 Series, X3, i8)
- **Year**: Manufacturing year
- **Region**: Sales region (Asia, North America, Middle East, Europe)
- **Color**: Vehicle color
- **Fuel Type**: Petrol, Hybrid, Electric, Diesel
- **Transmission**: Manual or Automatic
- **Engine Size**: Engine displacement in liters
- **Mileage**: Vehicle mileage in kilometers
- **Price**: Price in USD
- **Sales Volume**: Historical sales volume

## 🎯 Target Variable

- **Sales_Classification**: Binary classification (High/Low)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd BMW_car_sales
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.7+
- pandas >= 2.0.0
- xgboost >= 2.0.0
- scikit-learn >= 1.3.0
- joblib >= 1.3.0
- questionary >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0

## 🚀 Usage

### Training the Model

To train the model from scratch:

```bash
python model.py
```

This will:
- Load the BMW sales dataset
- Preprocess the data using ordinal encoding
- Split data into training and testing sets
- Train an XGBoost classifier
- Evaluate model accuracy
- Save the trained model and encoders

### Making Predictions

To make predictions on new data:

```bash
python infer.py
```

The inference script will:
- Load the pre-trained model and encoders
- Prompt you to select vehicle characteristics through an interactive menu
- Ask for numerical inputs (mileage, price, sales volume)
- Output the predicted sales classification

### Example Prediction Flow

```
Select Model: [Interactive menu with BMW models]
Select Year: [Available years]
Select Region: [Asia, North America, Middle East, Europe]
Select Color: [Available colors]
Select Fuel_Type: [Petrol, Hybrid, Electric, Diesel]
Select Transmission: [Manual, Automatic]
Enter Mileage: 50000
Enter Price_USD: 85000
Enter Sales_Volume: 5000

The Sales Classification is: High
```

## 📁 Project Structure

```
BMW_car_sales/
├── BMW_Car_Sales_Classification.csv  # Dataset
├── model.py                          # Training script
├── infer.py                          # Inference script
├── requirements.txt                  # Dependencies
├── model.json                        # Trained XGBoost model
├── encoder.pkl                       # Feature encoder
├── encoder_predict.pkl               # Target encoder
├── venv/                             # Virtual environment
├── .gitignore                        # Git ignore file
└── README.md                         # This file
```

## 🔧 Model Details

- **Algorithm**: XGBoost Classifier
- **Hyperparameters**:
  - n_estimators: 50
  - num_parallel_tree: 2
  - max_depth: 5
- **Preprocessing**: Ordinal encoding for categorical features
- **Train-Test Split**: 80-20 split with random_state=42

## 📈 Model Performance

The model's performance is evaluated using accuracy score on the test set. Results are displayed during training.

## 🔄 Data Pipeline

1. **Data Loading**: Load BMW sales data from CSV
2. **Feature Encoding**: Apply ordinal encoding to categorical features
3. **Target Encoding**: Encode sales classification labels
4. **Train-Test Split**: Split data for training and validation
5. **Model Training**: Train XGBoost classifier
6. **Model Evaluation**: Calculate accuracy on test set
7. **Model Persistence**: Save model and encoders for future use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure all dependencies are installed
2. **File Not Found**: Make sure you're running scripts from the project root directory
3. **Encoding Errors**: Ensure the CSV file is properly formatted and accessible

### Support

If you encounter any issues or have questions, please open an issue in the repository.

---

**Note**: This project is for educational and research purposes. The model's predictions should be validated against actual market data before making business decisions.