# =============================================
# FAKE NEWS DETECTION SYSTEM
# Using WELFake Dataset | No Virtual Env
# Models: Logistic, NB, DT, SVM, LSTM, BERT
# =============================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# 1. DOWNLOAD NLTK DATA (Only first time)
# -------------------------------
try:
    nltk.data.find('tokenizers/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# 2. LOAD DATASET (Local CSV)
# -------------------------------
DATASET_PATH = "WELFake_Dataset.csv"  # Place file in same folder

if not os.path.exists(DATASET_PATH):
    print("❌ Dataset not found! Please download WELFake_Dataset.csv from:")
    print("https://zenodo.org/record/4561253/files/WELFake_Dataset.csv?download=1")
    exit()

df = pd.read_csv(DATASET_PATH)
print(f"✅ Loaded dataset with {len(df)} rows")

# Check label mapping: 0 = fake, 1 = real
print("Label distribution:")
print(df['label'].value_counts())

# Drop missing text
df.dropna(subset=['text'], inplace=True)
df.reset_index(drop=True, inplace=True)

# -------------------------------
# 3. TEXT PREPROCESSING
# -------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Lowercase
    text = str(text).lower()
    # Remove non-alphabetic chars
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    words = text.split()
    # Remove short words and stopwords, then lemmatize
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words and len(word) > 2]
    return ' '.join(words)

print("🔍 Preprocessing text...")
df['cleaned_text'] = df['title'].fillna('') + " " + df['text']
df['cleaned_text'] = df['cleaned_text'].apply(preprocess_text)
df = df[df['cleaned_text'].str.strip() != ""]
df.reset_index(drop=True, inplace=True)
print("✅ Text preprocessing done!")

# -------------------------------
# 4. TF-IDF FOR ML MODELS
# -------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print(f"TF-IDF Shape: {X_train.shape}")

# -------------------------------
# 5. TRAIN MACHINE LEARNING MODELS
# -------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "SVM": SVC(kernel='rbf', C=1.0)
}

results = {}

print("\n🚀 Training ML Models...\n")
for name, model in models.items():
    print(f"👉 Training {name}...")
    try:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = acc
        print(f"✅ {name} Accuracy: {acc:.4f}\n")
    except Exception as e:
        print(f"❌ {name} failed: {e}\n")

# -------------------------------
# 6. LSTM MODEL
# -------------------------------
print("🚀 Training LSTM...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize
tokenizer_lstm = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer_lstm.fit_on_texts(df['cleaned_text'])

sequences = tokenizer_lstm.texts_to_sequences(df['cleaned_text'])
X_pad = pad_sequences(sequences, maxlen=256, padding='post', truncating='post')

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_pad, y, test_size=0.2, random_state=42, stratify=y
)

# Build model
model_lstm = Sequential([
    Embedding(input_dim=10000, output_dim=300, input_length=256),
    SpatialDropout1D(0.4),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.summary()

# Train
try:
    history = model_lstm.fit(
        X_train_lstm, y_train_lstm,
        epochs=3,
        batch_size=128,
        validation_data=(X_test_lstm, y_test_lstm),
        verbose=1
    )
    y_pred_lstm = (model_lstm.predict(X_test_lstm) > 0.5).astype(int)
    acc_lstm = accuracy_score(y_test_lstm, y_pred_lstm)
    results["LSTM"] = acc_lstm
    print(f"\n✅ LSTM Accuracy: {acc_lstm:.4f}")
except Exception as e:
    print(f"❌ LSTM failed: {e}")

# -------------------------------
# 7. BERT MODEL (Using Hugging Face)
# -------------------------------
print("\n🚀 Loading BERT Model...")

try:
    from transformers import BertTokenizer, TFBertModel
    import tensorflow as tf

    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    def bert_encode(texts, max_len=60):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = tokenizer_bert.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='tf'
            )
            input_ids.append(encoded['input_ids'][0])
            attention_masks.append(encoded['attention_mask'][0])
        return np.array(input_ids), np.array(attention_masks)

    # Prepare data
    input_ids, masks = bert_encode(df['cleaned_text'].tolist())
    y_bert = df['label'].values

    X_train_b, X_test_b, mask_train, mask_test, y_train_b, y_test_b = train_test_split(
        input_ids, masks, y_bert, test_size=0.2, random_state=42, stratify=y_bert
    )

    # Build classifier
    input_id_layer = tf.keras.Input(shape=(60,), dtype=tf.int32, name='input_ids')
    mask_layer = tf.keras.Input(shape=(60,), dtype=tf.int32, name='attention_mask')

    bert_output = bert_model(input_id_layer, attention_mask=mask_layer)
    cls_output = bert_output.last_hidden_state[:, 0, :]  # CLS token
    output = tf.keras.layers.Dense(1, activation='sigmoid')(cls_output)

    model_bert = tf.keras.Model(inputs=[input_id_layer, mask_layer], outputs=output)
    model_bert.compile(optimizer=tf.keras.optimizers.Adam(2e-5), 
                       loss='binary_crossentropy', metrics=['accuracy'])

    print("✅ BERT Model Built")

    # Train briefly
    history_bert = model_bert.fit(
        [X_train_b, mask_train], y_train_b,
        epochs=2,
        batch_size=32,
        validation_data=([X_test_b, mask_test], y_test_b),
        verbose=1
    )

    y_pred_bert = (model_bert.predict([X_test_b, mask_test]) > 0.5).astype(int)
    acc_bert = accuracy_score(y_test_b, y_pred_bert)
    results["BERT"] = acc_bert
    print(f"\n✅ BERT Accuracy: {acc_bert:.4f}")

except Exception as e:
    print(f"❌ BERT failed: {e}")
    print("💡 Tip: Run 'pip install transformers tensorflow' if missing")

# -------------------------------
# 8. FINAL RESULTS COMPARISON
# -------------------------------
print("\n" + "="*50)
print("📊 FINAL MODEL ACCURACIES")
print("="*50)

for model, acc in results.items():
    print(f"{model:20} : {acc:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
models_list = list(results.keys())
acc_list = [results[m] for m in models_list]
sns.barplot(x=acc_list, y=models_list, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy")
plt.xlim(0.5, 1.0)
for i, v in enumerate(acc_list):
    plt.text(v + 0.005, i, f"{v:.3f}", va='center')
plt.tight_layout()
plt.show()

# ----------------------------------
# 9. PREDICTION FUNCTION
# ----------------------------------
def predict_news(text):
    print(f"\n🔍 Predicting: {text[:100]}...")
    cleaned = preprocess_text(text)
    
    # TF-IDF prediction
    vec = vectorizer.transform([cleaned])
    for name in ["Logistic Regression", "SVM", "Decision Tree"]:
        if name in models:
            pred = models[name].predict(vec)[0]
            print(f"  {name:18}: {'✅ Real' if pred == 1 else '❌ Fake'}")
    
    # LSTM
    try:
        seq = tokenizer_lstm.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=256, padding='post')
        pred = (model_lstm.predict(pad) > 0.5).astype(int)[0][0]
        print(f"  LSTM               : {'✅ Real' if pred == 1 else '❌ Fake'}")
    except:
        pass

    # BERT
    try:
        input_ids, masks = bert_encode([cleaned], max_len=60)
        pred = (model_bert.predict([input_ids, masks]) > 0.5).astype(int)[0][0]
        print(f"  BERT               : {'✅ Real' if pred == 1 else '❌ Fake'}")
    except:
        pass

# Example prediction
sample_text = "The Earth is flat and NASA is hiding the truth."
predict_news(sample_text)