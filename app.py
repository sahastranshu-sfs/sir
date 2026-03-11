import streamlit as st
st.set_page_config(page_title="HS-CBDS", page_icon="🛡️", layout="wide")
import pandas as pd
import re
import nltk
import os
import joblib

# Set NLTK data path to local directory to avoid permission issues
nltk_data_dir = os.path.abspath("./nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from scipy.sparse import hstack, csr_matrix
import numpy as np
import gc
from textblob import TextBlob
from datetime import datetime

# Sample Bag of Words collected from survey (copied from model.py)
cyberbully_words = set(["idiot", "loser", "hate", "ugly","bc", "mc", "stupid", "fuck", "idiot", "go to hell", "bsdk", "bitch", "threats", "fuckoff", "fake", "kutta", "busturd", "threat", "mother fucker", "fuck you", "nudity", "disgusting", "asshole", "fraud", "vulgarness", "sala", "fuckyou", "fake ads", "abuse", "evil", "mms", "fake calls", "pissoff", "chutiya", "sex", "sexy", "bullshit", "bloodyhell", "basturd", "fuck off", "dumb", "fake id", "nigga", "non sense", "nude", "rascal", "fucker", "low life", "nasty", "hate", "lame", "hell", "nudes", "account hack", "duffer", "fake comment", "dogshit", "useless", "bhenchod", "nonsense", "make money", "fake account", "dawg", "racist", "mf", "give money", "kutti", "hot", "naked", "flaming", "fake links", "awful", "abuse language", "fake ids", "racism", "worried", "mental", "harrasment", "dowry", "gambling", "frustrate", "life threat", "kussa", "dog", "motherfucker", "photos", "sexual", "slemia", "job scam", "abusive language", "asking for money", "hacking links", "action again", "fake comments", "kanjar", "mofo", "post", "crip", "naughty", "lode", "get lost", "humilation", "rash language", "low-life", "gandu", "ass hole", "jerk", "halal", "exclusion", "bcdk", "racsal", "salla", "fake call", "saala", "buy product", "kamina", "bullying", "links", "kbc", "crypto", "earn money online", "oh ! no", "leaked video", "ediot", "don't call me ever", "slave", "my pic used", "hake id", "bank fake calls", "an affective", "dumbass", "fuck up", "size of a tea", "morchod", "sadness", "scambag", "retard", "image", "negative comment", "otp", "chappa", "reload", "vigas", "bander", "elvishbhai", "tatakae", "pedopile", "black", "porn", "abuse word", "chusli", "bamb", "suicide", "sexual contant", "khalistani", "fake id message", "fake photo", "blocked", "send pic", "wanker", "bollocks", "poor life", "i will kill you", "lotery", "pagal", "emailspan", "killyou", "theft", "brother", "affection", "abusing", "send nudes", "deaf theart", "cheater", "problem", "without any help", "leg pulling", "backdog", "bulling",  "loan without interest", "free voucher", "you win a reward", "wanting", "report them", "vedio call", "please reply", "i don't know how they get my number or my email .", "body shaming etc", "earn money", "brain was", "embrassing", "leak", "looser", "cyber stalking", "dancing", "ideat", "blockno", "dum", "vulgariness", "abuseing", "fools", "job scams", "embrasing comment", "weight salon", "difficulties", "depressed", "spam contact", "fuck of gay", "whose", "fuggat", "hateful", "bloody hill", "wrong no", "chapi", "something wrong", "dump", "khalisthani", "support lgbtq", "fraud calling", "harmfull video", "leaked", "worthless", "rasist", "fu", "calls", "tota", "work from home", "depression", "oye papa ji", "sexual haresment", "fake video", "yelled", "video chat", "bloody hell", "humiliation", "gendu", "idiots", "madarchod", "low live", "law life", "i know your ip address", "download", "fucking", "piss of", "lowlife", "online payment", "police", "cbbsdk", "harrasing", "explore our life", "blackmailing", "chutia", "videos", "patience", "stalker", "bad comment", "abused", "cash back offer", "obsession", "videocall", "video call", "spam", "fudu", "jihaad", "photo leaked", "skinny", "mad", "in approprate", "fakecall", "soicla", "rmosur", "cyber staking", "dick", "gays", "using photos", "kill you", "go to hell", "go hell","fake massage", "nude posts", "commenting a cloths", "fake posts", "riacism", "she cried a river", "nude post", "bustard", "landnudes", "hoe", "murdered", "btdk", "fake post","chupa mar", "curry", "abuse words", "senstive contact", "topyfake", "bakchadi", "bakchodi", "gold digger", "chod", "chomu", "purchase my assest", "shit eating indian", "fake offers", "i kill you", "chatiya", "cash", "fatso", "send me ransom", "click on link", "you are worthfull", "low energy", "ladla",  "chat with me", "i kill myself", "body shame", "baby shaming", "impact", "mcar", "fool", "run you", "faggot", "sexuality", "bandar", "try to understand", "bad comments", "nude pics", "free easy job with good package", "harassment", "hatred", "you have won a lottery", "trading", "khalistan","mula", "others", "dark", "fakechat", "reject", "abusive word", "egergavis", "fake people dm", "vulgar message", "Vulgar ness", "bahen chood", "teasing", "discrimination", "religion", "it is killing me", "fucuer", "booharmil", "being absent in class", "staging office", "job scam links", "lgbt", "abusive words", "shitoff", "mkc", "terriost", "bupendra jogi", "fraud message", "hardcore", "lorny", "penchod", "moye moye", "chek my ass", "brown road shitter", "fake contact", "hack", "shabal gang", "yeah buddy", "bhenke lode", "bc mc", "i got your personal data", "fatass", "body shaming", "raid", "scams", "durgs", "fraud call", "boby shaming", "shame", "laodayia", "lawder", "loude", "gmm", "sali", "ghada", "vulgur", "stupid", "dumb", "fat", "kill", "shut up", "worthless"])

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def add_features(df, text_column):
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['text_length'] = df[text_column].apply(len)
    return df

def check_bag_of_words(text):
    tokens = set(word_tokenize(text.lower()))
    lex = get_lexicon()
    matched_words = tokens.intersection(lex)
    return matched_words

def get_lexicon():
    base = cyberbully_words
    custom = set(st.session_state.get("custom_words", []))
    return base.union(custom)

def get_explicit_words():
    base = {"fuck", "bitch", "asshole", "motherfucker", "fucker", "bastard", "slut", "whore", "bhenchod", "chutiya"}
    custom = set(st.session_state.get("explicit_words", []))
    return base.union(custom)

def is_explicit_abuse(text):
    if not st.session_state.get("explicit_override_enabled", True):
        return False
    t = text.lower()
    for w in get_explicit_words():
        p = r"\b" + re.escape(w) + r"\w*\b"
        if re.search(p, t):
            return True
    return False
@st.cache_resource
def load_or_train_model():
    model_file = "cyberbullying_model.pkl"
    
    if os.path.exists(model_file):
        print(f"Loading model from {model_file}...")
        return joblib.load(model_file)
        
    print("Model file not found. Training new model...")
    # Use relative path or the absolute path provided by user
    try:
        df = pd.read_csv("anti-bully.csv", encoding="latin-1")
    except FileNotFoundError:
        df = pd.read_csv(r"c:\Users\PC\Desktop\sir\anti-bully.csv", encoding="latin-1")
        
    df['clean_text'] = df['text'].apply(preprocess_text)
    df = add_features(df, 'clean_text')

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, max_df=0.95)
    X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

    scaler = StandardScaler()
    numerical_features = df[['sentiment', 'text_length']].fillna(0)
    scaled_features = scaler.fit_transform(numerical_features)
    X_combined = hstack([X_tfidf, csr_matrix(scaled_features)])

    X_train, X_test, y_train, y_test = train_test_split(X_combined, df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    classifiers = {
        'lsvc': LinearSVC(random_state=42),
        'lr': LogisticRegression(random_state=42),
        'rf': RandomForestClassifier(random_state=42, n_jobs=-1),
        'dt': DecisionTreeClassifier(random_state=42),
        'ann': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    best_models = {}

    X_train_gs = X_train
    y_train_gs = y_train
    max_gs_samples = 20000
    if X_train.shape[0] > max_gs_samples:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_gs_samples, random_state=42)
        for idx, _ in splitter.split(X_train, y_train):
            X_train_gs = X_train[idx]
            if hasattr(y_train, "iloc"):
                y_train_gs = y_train.iloc[idx]
            else:
                y_train_gs = y_train[idx]
    total_classifiers = len(classifiers)
    for i, (name, clf) in enumerate(classifiers.items()):
        print(f"Tuning {name}...")
        param_grid = {}
        if name == 'lsvc':
            param_grid = {'C': [0.1, 1.0, 10.0]}
        elif name == 'lr':
            param_grid = {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'saga']}
        elif name == 'rf':
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
        elif name == 'dt':
            param_grid = {'max_depth': [10, 20, 30], 'min_samples_split': [2, 5]}
        elif name == 'ann':
            param_grid = {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'max_iter': [300]}

        grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train_gs, y_train_gs)
        best_params = grid_search.best_params_
        best_clf = clf.__class__(**best_params)
        best_clf.fit(X_train, y_train)
        best_models[name] = best_clf
        print(f"Finished {name}")

    print("Training complete! Saving model...")
    del X_combined
    gc.collect()

    weights = {'lsvc': 0.2, 'lr': 0.2, 'rf': 0.2, 'dt': 0.2, 'ann': 0.2}
    predictions = []

    for name, model in best_models.items():
        pred = model.predict(X_test)
        predictions.append(pred * weights[name])

    final_prediction = np.round(np.sum(predictions, axis=0)).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, final_prediction),
        'precision': precision_score(y_test, final_prediction, average='weighted'),
        'recall': recall_score(y_test, final_prediction, average='weighted'),
        'f1': f1_score(y_test, final_prediction, average='weighted'),
        'roc_auc': roc_auc_score(y_test, final_prediction)
    }
    cm = confusion_matrix(y_test, final_prediction)
    report = classification_report(y_test, final_prediction, output_dict=True)
    artifacts = {
        'confusion_matrix': cm,
        'classification_report': report,
        'y_test': y_test,
        'final_prediction': final_prediction,
        'trained_at': datetime.now().isoformat()
    }
    result = (best_models, weights, tfidf_vectorizer, scaler, metrics, artifacts)
    joblib.dump(result, model_file)
    print(f"Model saved to {model_file}")
    
    return result

def train_quick_model():
    try:
        df = pd.read_csv("anti-bully.csv", encoding="latin-1")
    except FileNotFoundError:
        df = pd.read_csv(r"c:\Users\PC\Desktop\sir\anti-bully.csv", encoding="latin-1")
    df['clean_text'] = df['text'].apply(preprocess_text)
    df = add_features(df, 'clean_text')
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, max_df=0.95)
    X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])
    scaler = StandardScaler()
    numerical_features = df[['sentiment', 'text_length']].fillna(0)
    scaled_features = scaler.fit_transform(numerical_features)
    X_combined = hstack([X_tfidf, csr_matrix(scaled_features)])
    X_train, X_test, y_train, y_test = train_test_split(X_combined, df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    best_models = {
        'lsvc': LinearSVC(random_state=42),
        'lr': LogisticRegression(random_state=42),
        'rf': RandomForestClassifier(random_state=42, n_jobs=-1),
        'dt': DecisionTreeClassifier(random_state=42),
        'ann': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    for name, clf in best_models.items():
        clf.fit(X_train, y_train)
    weights = {'lsvc': 0.2, 'lr': 0.2, 'rf': 0.2, 'dt': 0.2, 'ann': 0.2}
    predictions = []
    for name, model in best_models.items():
        pred = model.predict(X_test)
        predictions.append(pred * weights[name])
    final_prediction = np.round(np.sum(predictions, axis=0)).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_test, final_prediction),
        'precision': precision_score(y_test, final_prediction, average='weighted'),
        'recall': recall_score(y_test, final_prediction, average='weighted'),
        'f1': f1_score(y_test, final_prediction, average='weighted'),
        'roc_auc': roc_auc_score(y_test, final_prediction)
    }
    cm = confusion_matrix(y_test, final_prediction)
    report = classification_report(y_test, final_prediction, output_dict=True)
    artifacts = {
        'confusion_matrix': cm,
        'classification_report': report,
        'y_test': y_test,
        'final_prediction': final_prediction,
        'trained_at': datetime.now().isoformat()
    }
    result = (best_models, weights, tfidf_vectorizer, scaler, metrics, artifacts)
    joblib.dump(result, "cyberbullying_model.pkl")
    return result
def main():
    st.title("🛡️ Hybrid Supervised Cyber-bully Detection System")

    st.sidebar.header("Model")
    uploaded_model = st.sidebar.file_uploader("Upload pre-trained model (.pkl)", type=["pkl"])
    if uploaded_model is not None:
        try:
            data_bytes = uploaded_model.read()
            with open("cyberbullying_model.pkl", "wb") as f:
                f.write(data_bytes)
            st.sidebar.success("Model uploaded and saved.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Upload failed: {e}")
    retrain_clicked = st.sidebar.button("Retrain Model")
    if retrain_clicked:
        try:
            if os.path.exists("cyberbullying_model.pkl"):
                os.remove("cyberbullying_model.pkl")
            st.cache_resource.clear()
            st.sidebar.info("Cache cleared. Starting fresh training…")
        except Exception as e:
            st.sidebar.error(f"Could not clear cache: {e}")
        st.rerun()
    data = None
    if os.path.exists("cyberbullying_model.pkl"):
        try:
            data = joblib.load("cyberbullying_model.pkl")
        except Exception as e:
            st.sidebar.error(f"Could not load saved model: {e}")
    if data is None:
        st.warning("No saved model found.")
        c1, c2 = st.columns(2)
        if c1.button("Train Full Model"):
            with st.spinner("Training full model..."):
                try:
                    data = load_or_train_model()
                except Exception as e:
                    st.error(f"Training failed: {e}")
        if c2.button("Train Quick Model"):
            with st.spinner("Training quick model..."):
                try:
                    data = train_quick_model()
                except Exception as e:
                    st.error(f"Quick training failed: {e}")
    if data is None:
        return
    if isinstance(data, tuple) and len(data) == 6:
        best_models, weights, tfidf_vectorizer, scaler, metrics, artifacts = data
    else:
        best_models, weights, tfidf_vectorizer, scaler, metrics = data
        artifacts = {}
    trained_at = artifacts.get('trained_at', None)
    if trained_at:
        st.sidebar.success(f"Loaded • {trained_at}")
    else:
        st.sidebar.info("Loaded model")
    tabs = st.tabs(["Overview", "Test Message", "Lexicon", "Analytics", "About"])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        c2.metric("Precision", f"{metrics['precision']:.4f}")
        c3.metric("Recall", f"{metrics['recall']:.4f}")
        c4, c5 = st.columns(2)
        c4.metric("F1 Score", f"{metrics['f1']:.4f}")
        c5.metric("AUC-ROC", f"{metrics['roc_auc']:.4f}")
        cm_ov = artifacts.get('confusion_matrix', None)
        if cm_ov is not None:
            tn, fp, fn, tp = cm_ov.ravel()
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("TP", f"{tp}")
            d2.metric("TN", f"{tn}")
            d3.metric("FP", f"{fp}")
            d4.metric("FN", f"{fn}")
    with tabs[1]:
        st.subheader("Test the Model")
        if "history" not in st.session_state:
            st.session_state["history"] = []
        col_text, col_clear = st.columns([4, 1])
        with col_text:
            text = st.text_area("Enter a message")
        with col_clear:
            if st.button("Clear History"):
                st.session_state["history"] = []
                st.rerun()
        if st.button("Check"):
            if text:
                if is_explicit_abuse(text):
                    matched_bow = check_bag_of_words(text)
                    st.error("The message is predicted to be CYBERBULLYING.")
                    if matched_bow:
                        st.warning(", ".join(matched_bow))
                    st.session_state["history"].append({"text": text[:80], "prediction": "Cyberbullying"})
                else:
                    cleaned = preprocess_text(text)
                    sentiment = TextBlob(cleaned).sentiment.polarity
                    text_length = len(cleaned)
                    matched_bow = check_bag_of_words(text)
                    input_tfidf = tfidf_vectorizer.transform([cleaned])
                    input_df = pd.DataFrame([[sentiment, text_length]], columns=['sentiment', 'text_length'])
                    input_scaled = scaler.transform(input_df)
                    input_combined = hstack([input_tfidf, csr_matrix(input_scaled)])
                    preds = []
                    for name, model in best_models.items():
                        pred = model.predict(input_combined)
                        preds.append(pred * weights[name])
                    final_pred = np.round(np.sum(preds, axis=0)).astype(int)[0]
                    label = "Cyberbullying" if final_pred == 1 else "Not Cyberbullying"
                    if final_pred == 1:
                        st.error("The message is predicted to be CYBERBULLYING.")
                    else:
                        st.success("The message is NOT cyberbullying.")
                    if matched_bow:
                        st.warning(", ".join(matched_bow))
                    st.session_state["history"].append({"text": text[:80], "prediction": label})
        if st.session_state.get("history"):
            st.subheader("Recent Checks")
            st.dataframe(pd.DataFrame(st.session_state["history"][-10:]))
    with tabs[2]:
        st.subheader("Lexicon")
        if "custom_words" not in st.session_state:
            st.session_state["custom_words"] = []
        if "explicit_words" not in st.session_state:
            st.session_state["explicit_words"] = []
        if "explicit_override_enabled" not in st.session_state:
            st.session_state["explicit_override_enabled"] = True
        st.write(f"Bag of Words")
        if st.button("Clear Custom Words"):
            st.session_state["custom_words"] = []
            st.success("Cleared custom words.")
        if st.session_state["custom_words"]:
            st.write("Custom words")
            st.dataframe(pd.DataFrame({"word": st.session_state["custom_words"]}))
        st.checkbox("Enable explicit override", key="explicit_override_enabled")
        add_explicit = st.text_area("Add explicit override words (comma or newline separated)")
        e1, e2 = st.columns(2)
        if e1.button("Add Override Words"):
            new_items = []
            for part in re.split(r"[,\n]", add_explicit):
                w = part.strip().lower()
                if w:
                    new_items.append(w)
            st.session_state["explicit_words"] = list(sorted(set(st.session_state["explicit_words"]).union(new_items)))
            st.success(f"Added {len(new_items)} override word(s).")
        if e2.button("Clear Override Words"):
            st.session_state["explicit_words"] = []
            st.success("Cleared override words.")
        if st.session_state["explicit_words"]:
            st.write("Explicit override words")
            st.dataframe(pd.DataFrame({"word": st.session_state["explicit_words"]}))
    with tabs[3]:
        st.subheader("Analytics")
        cm = artifacts.get('confusion_matrix', None)
        report = artifacts.get('classification_report', None)
        if cm is not None:
            st.write("Confusion Matrix")
            st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
            tn, fp, fn, tp = cm.ravel()
            st.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
            total = tp + tn + fp + fn
            acc_v = (tp + tn) / total if total else 0
            prec_v = tp / (tp + fp) if (tp + fp) else 0
            rec_v = tp / (tp + fn) if (tp + fn) else 0
            f1_v = 2 * prec_v * rec_v / (prec_v + rec_v) if (prec_v + rec_v) else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc_v:.4f}")
            c2.metric("Precision", f"{prec_v:.4f}")
            c3.metric("Recall", f"{rec_v:.4f}")
            c4.metric("F1 Score", f"{f1_v:.4f}")
        if report is not None:
            st.write("Classification Report")
            rep_df = pd.DataFrame(report).transpose()
            st.dataframe(rep_df)
        if cm is None and report is None:
            st.info("Analytics will be available after next training.")
    with tabs[4]:
        st.markdown("NOVEL METHOD TO DETECT CYBER-BULLY MESSAGES ON SOCIAL MEDIA USING SUPERVISED MACHINE LEARNING TECHNIQUE")
   

if __name__ == "__main__":
    main()
