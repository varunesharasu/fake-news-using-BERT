# 🚨 Fake News Detection System

A comprehensive machine learning project that detects fake NEWS using multiple algorithms including traditional ML models, LSTM, and BERT transformer models. This system is trained on the WELFake dataset and provides comparison between different approaches to fake news classification.

## 📊 Project Overview

This project implements and compares six different machine learning approaches for fake news detection:

- **Traditional ML Models**: Logistic Regression, Naive Bayes, Decision Tree, SVM
- **Deep Learning**: LSTM (Long Short-Term Memory)
- **Transformer Model**: BERT (Bidirectional Encoder Representations from Transformers)

## 🎯 Features

- **Multi-Model Comparison**: Compare performance across different algorithms
- **Text Preprocessing**: Advanced text cleaning and preprocessing pipeline
- **TF-IDF Vectorization**: Feature extraction for traditional ML models
- **Deep Learning**: LSTM implementation with embedding layers
- **BERT Integration**: State-of-the-art transformer model using Hugging Face
- **Visualization**: Performance comparison charts
- **Real-time Prediction**: Function to predict new text samples
- **No Virtual Environment Required**: Direct installation approach

## 📁 Project Structure

```
fake-news-using-BERT/
├── Model/
│   ├── fake_news_detection.py    # Main implementation file
│   └── WELFake_Dataset.csv       # Dataset file (to be downloaded)
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
```

## 🔧 Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn tensorflow transformers
```

### NLTK Data
The script automatically downloads required NLTK data:
- `wordnet` (for lemmatization)
- `stopwords` (for text preprocessing)

## 📥 Dataset Setup

1. **Download the WELFake Dataset**:
   - Visit: [WELFake Dataset on Zenodo](https://zenodo.org/record/4561253/files/WELFake_Dataset.csv?download=1)
   - Download `WELFake_Dataset.csv`
   - Place the file in the `Model/` directory

2. **Dataset Information**:
   - **Size**: ~72,000 news articles
   - **Labels**: 0 = Fake News, 1 = Real News
   - **Features**: Title, Text, Subject, Date
   - **Source**: Combines multiple datasets (Kaggle, McIntire, Reuters, BuzzFeed Political)

## 🚀 Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/varunesharasu/fake-news-using-BERT.git
cd fake-news-using-BERT
```

2. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn tensorflow transformers
```

3. **Download the dataset**:
   - Download `WELFake_Dataset.csv` and place it in the `Model/` folder

4. **Run the system**:
```bash
cd Model
python fake_news_detection.py
```

## 🔍 How It Works

### 1. Data Preprocessing
- **Text Cleaning**: Removes non-alphabetic characters, converts to lowercase
- **Stop Words Removal**: Filters out common English stop words
- **Lemmatization**: Reduces words to their root form
- **Feature Engineering**: Combines title and text for comprehensive analysis

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts text to numerical features with n-grams (1,2)
- **Word Embeddings**: For LSTM model using Keras tokenizer
- **BERT Tokenization**: Special tokenization for transformer model

### 3. Model Training

#### Traditional ML Models
- **Logistic Regression**: Linear classifier with regularization
- **Naive Bayes**: Probabilistic classifier (MultinomialNB)
- **Decision Tree**: Tree-based classifier with max_depth=10
- **SVM**: Support Vector Machine with RBF kernel

#### LSTM Model
- **Architecture**: Embedding → SpatialDropout1D → LSTM → Dense → Output
- **Parameters**: 
  - Embedding dimension: 300
  - LSTM units: 128
  - Sequence length: 256
  - Dropout rates: 0.2-0.5

#### BERT Model
- **Pre-trained Model**: `bert-base-uncased`
- **Fine-tuning**: Custom classification head
- **Sequence length**: 60 tokens
- **Learning rate**: 2e-5

### 4. Evaluation Metrics
- **Accuracy Score**: Primary evaluation metric
- **Classification Report**: Precision, Recall, F1-score
- **Visual Comparison**: Bar chart of model accuracies

## 📈 Expected Performance

Based on typical performance on the WELFake dataset:

| Model | Expected Accuracy |
|-------|------------------|
| BERT | 98-99% |
| LSTM | 94-96% |
| SVM | 92-94% |
| Logistic Regression | 91-93% |
| Naive Bayes | 89-91% |
| Decision Tree | 85-88% |

## 🎮 Usage Examples

### Predicting New Text
```python
# Example usage after running the script
sample_text = "Breaking: Scientists discover cure for all diseases!"
predict_news(sample_text)
```

### Sample Predictions
The system provides predictions from all trained models:
```
🔍 Predicting: Breaking: Scientists discover cure for all diseases!...
  Logistic Regression: ❌ Fake
  SVM               : ❌ Fake
  Decision Tree     : ❌ Fake
  LSTM              : ❌ Fake
  BERT              : ❌ Fake
```

## 🛠️ Customization Options

### Adjust Model Parameters
```python
# LSTM Model
model_lstm = Sequential([
    Embedding(input_dim=10000, output_dim=300, input_length=256),
    SpatialDropout1D(0.4),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),  # Adjust LSTM units
    Dense(64, activation='relu'),                     # Adjust dense layer size
    Dropout(0.5),                                    # Adjust dropout rate
    Dense(1, activation='sigmoid')
])
```

### Modify Text Preprocessing
```python
def preprocess_text(text):
    # Add custom preprocessing steps
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Modify regex pattern
    # Add more preprocessing as needed
    return text
```

### Change TF-IDF Parameters
```python
vectorizer = TfidfVectorizer(
    max_features=10000,     # Adjust vocabulary size
    ngram_range=(1,2),      # Modify n-gram range
    min_df=2,              # Add minimum document frequency
    max_df=0.95            # Add maximum document frequency
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Not Found**:
   ```
   ❌ Dataset not found! Please download WELFake_Dataset.csv
   ```
   **Solution**: Download the dataset and place it in the `Model/` directory

2. **NLTK Data Missing**:
   ```
   LookupError: Resource wordnet not found
   ```
   **Solution**: The script auto-downloads NLTK data, but you can manually run:
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

3. **Memory Issues with BERT**:
   ```
   ResourceExhaustedError: OOM when allocating tensor
   ```
   **Solution**: Reduce batch size or sequence length:
   ```python
   # Reduce batch size
   batch_size=16  # instead of 32
   
   # Reduce sequence length
   max_len=30     # instead of 60
   ```

4. **Transformers Library Missing**:
   ```
   ModuleNotFoundError: No module named 'transformers'
   ```
   **Solution**: Install transformers:
   ```bash
   pip install transformers
   ```

### Performance Optimization

1. **For Faster Training**:
   - Reduce dataset size for testing
   - Use smaller BERT model (`distilbert-base-uncased`)
   - Reduce number of epochs

2. **For Better Accuracy**:
   - Increase training epochs
   - Use larger sequence lengths
   - Implement cross-validation
   - Add data augmentation

## 📚 Technical Details

### Data Flow
```
Raw Text → Preprocessing → Feature Extraction → Model Training → Prediction
```

### Model Architecture Details

#### LSTM Architecture
```
Input (256 tokens) → Embedding (300d) → SpatialDropout → LSTM (128) → Dense (64) → Output (1)
```

#### BERT Architecture
```
Input Tokens → BERT Encoder → CLS Token → Dense Layer → Sigmoid → Output
```

### Preprocessing Pipeline
```
Raw Text → Lowercase → Remove Special Chars → Tokenize → Remove Stopwords → Lemmatize → Clean Text
```

## 🔬 Research Background

### Fake News Detection Approaches
1. **Feature-based**: TF-IDF, N-grams, Linguistic features
2. **Deep Learning**: RNN, LSTM, CNN
3. **Transformer Models**: BERT, RoBERTa, GPT
4. **Ensemble Methods**: Combining multiple models

### Dataset Characteristics
- **WELFake Dataset**: Comprehensive collection of labeled news
- **Class Distribution**: Balanced dataset with equal fake/real samples
- **Text Length**: Variable length articles (50-5000+ words)
- **Domains**: Politics, Entertainment, Sports, Technology

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Contribution Ideas
- Add more preprocessing techniques
- Implement additional models (XGBoost, Random Forest)
- Add cross-validation
- Implement ensemble methods
- Add web interface
- Create API endpoints

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

- **Author**: Varuna Esharasu
- **GitHub**: [@varunesharasu](https://github.com/varunesharasu)
- **Project**: [fake-news-using-BERT](https://github.com/varunesharasu/fake-news-using-BERT)

## 🙏 Acknowledgments

- **WELFake Dataset**: [Verma et al., 2021](https://zenodo.org/record/4561253)
- **Hugging Face**: For the transformers library and pre-trained models
- **TensorFlow/Keras**: For deep learning framework
- **Scikit-learn**: For traditional machine learning algorithms
- **NLTK**: For natural language processing tools

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@misc{fake-news-bert-2025,
  title={Fake News Detection using BERT and Traditional ML Models},
  author={Varuna Esharasu},
  year={2025},
  url={https://github.com/varunesharasu/fake-news-using-BERT}
}
```

---

**⭐ If you found this project helpful, please give it a star!**
