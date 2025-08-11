# Spam Mail Prediction Using Machine Learning With Python

A machine learning project that automatically classifies emails as spam or legitimate (ham) using Natural Language Processing and Logistic Regression algorithm.

## 🎯 Project Overview

This project implements a spam email detection system using Python and machine learning techniques. The model analyzes email content and predicts whether an email is spam or legitimate with high accuracy using TF-IDF vectorization and Logistic Regression.

## 🚀 Features

- **Text Preprocessing**: Automatic cleaning and preparation of email data
- **TF-IDF Vectorization**: Converts email text into numerical features
- **Machine Learning Classification**: Uses Logistic Regression for prediction
- **Performance Metrics**: Displays training and testing accuracy scores
- **Real-time Prediction**: Test the model with custom email content

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning library
  - TfidfVectorizer for feature extraction
  - LogisticRegression for classification
  - train_test_split for data splitting
  - accuracy_score for model evaluation

## 📁 Project Structure

```
spam-mail-prediction/
│
├── spam_detection.py          # Main Python script
├── email_origin.csv          # Dataset (spam/ham emails)
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SauravDnj/Spam-Mail-Prediction-Using-Machine-Learning-With-Python.git
cd Spam-Mail-Prediction-Using-Machine-Learning-With-Python
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Project
```bash
python spam_detection.py
```

## 📊 Dataset

The project uses `email_origin.csv` containing:
- **Message**: Email content/text
- **Category**: Classification label
  - `spam` (0): Unwanted/junk emails
  - `ham` (1): Legitimate emails

## 🧠 How It Works

1. **Data Loading**: Reads email dataset from CSV file
2. **Data Preprocessing**: 
   - Handles missing values
   - Converts categorical labels to numeric (spam=0, ham=1)
3. **Feature Extraction**: 
   - Uses TF-IDF vectorization to convert text to numerical features
   - Removes English stop words and converts to lowercase
4. **Model Training**:
   - Splits data into 80% training, 20% testing
   - Trains Logistic Regression model
5. **Evaluation**: 
   - Calculates accuracy on both training and test sets
6. **Prediction**: Tests model with new email content

## 📈 Model Performance

The model achieves high accuracy in distinguishing between spam and legitimate emails. Performance metrics are displayed during execution:

- Training Accuracy: ~XX%
- Testing Accuracy: ~XX%

## 🎯 Usage Example

```python
# Example of testing with custom email
input_mail = ["Congratulations! You've won $1000. Click here to claim your prize!"]
prediction = model.predict(feature_extraction.transform(input_mail))

if prediction[0] == 1:
    print("Legitimate Email")
else:
    print("Spam Email")
```

## 📝 Key Features of the Code

- **Data Validation**: Handles missing values in dataset
- **Feature Engineering**: TF-IDF vectorization for text analysis
- **Model Evaluation**: Comprehensive accuracy testing
- **Scalable Design**: Easy to retrain with new data

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Future Enhancements

- [ ] Add more ML algorithms (SVM, Random Forest, Naive Bayes)
- [ ] Implement cross-validation
- [ ] Create a web interface using Flask/Django
- [ ] Add email preprocessing improvements
- [ ] Include confusion matrix and classification report
- [ ] Deploy model using cloud services

## 🔗 Requirements

Create a `requirements.txt` file with:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## 👨‍💻 Author

**Saurav Danej**
- GitHub: [@SauravDnj](https://github.com/SauravDnj)
- LinkedIn: [@sauravdnj](https://www.linkedin.com/in/sauravdnj)
