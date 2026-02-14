import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix
import numpy as np
import gc
from textblob import TextBlob

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Sample Bag of Words collected from survey
cyberbully_words = set(["idiot", "loser", "hate", "ugly","bc", "mc", "stupid", "fuck", "idiot", "go to hell", "bsdk", "bitch", "threats", "fuckoff", "fake", "kutta", "busturd", "threat", "mother fucker", "fuck you", "nudity", "disgusting", "asshole", "fraud", "vulgarness", "sala", "fuckyou", "fake ads", "abuse", "evil", "mms", "fake calls", "pissoff", "chutiya", "sex", "sexy", "bullshit", "bloodyhell", "basturd", "fuck off", "dumb", "fake id", "nigga", "non sense", "nude", "rascal", "fucker", "low life", "nasty", "hate", "lame", "hell", "nudes", "account hack", "duffer", "fake comment", "dogshit", "useless", "bhenchod", "nonsense", "make money", "investment", "fake account", "dawg", "racist", "mf", "money", "kutti", "hot", "naked", "flaming", "fake links", "awful", "abuse language", "fake ids", "racism", "worried", "mental", "harrasment", "dowry", "gambling", "frustrate", "life threat", "kussa", "dog", "motherfucker", "photos", "sexual", "slemia", "job scam", "abusive language", "asking for money", "hacking links", "action again", "fake comments", "kanjar", "mofo", "post", "crip", "naughty", "lode", "get lost", "humilation", "rash language", "low-life", "gandu", "ass hole", "jerk", "hello", "halal", "exclusion", "bcdk", "racsal", "salla", "fake call", "saala", "buy product", "kamina", "bullying", "links", "kbc", "crypto", "earn money online", "oh ! no", "leaked video", "ediot", "don't call me ever", "heribaov", "slave", "my pic used", "hake id", "bank fake calls", "an affective", "dumbass", "fuck up", "size of a tea", "morchod", "sadness", "scambag", "retard", "image", "negative comment", "otp", "chappa", "reload", "vigas", "bander", "elvishbhai", "tatakae", "pedopile", "black", "porm", "abuse word", "chusli", "video", "bamb", "suicide", "sexual contant", "khalistani", "fake id message", "fake photo", "blocked", "send pic", "wanker", "bollocks", "poor life", "i will kill you", "sorry", "lotery", "pagal", "emailspan", "killyou", "theft", "brother", "affection", "abusing", "send nudes", "deaf theart", "cheater", "problem", "without any help", "leg pulling", "backdog", "bulling", "product many more", "loan without interest", "free voucher", "you win a reward", "wanting", "report them", "vedio call", "please reply", "i don't know how they get my number or my email .", "body shaming etc", "earn money", "brain was", "embrassing", "leak", "looser", "cyber stalking", "dancing", "ideat", "blockno", "dum", "vulgariness", "abuseing", "fools", "job scams", "embrasing comment", "weight salon", "difficulties", "depressed", "spam contact", "fuck of gay", "whose", "fuggat", "hateful", "bloody hill", "wrong no", "chapi", "something wrong", "dump", "khalisthani", "support lgbtq", "messages", "fraud calling", "message", "harmfull video", "leaked", "worthless", "rasist", "fu", "calls", "tota", "work from home", "depression", "oye papa ji", "sexual haresment", "fake video", "yelled", "video chat", "bloody hell", "humiliation", "gendu", "idiots", "madarchod", "low live", "law life", "i know your ip address", "download", "fucking", "piss of", "lowlife", "online payment", "police", "cbbsdk", "harrasing", "explore our life", "blackmailing", "chutia", "videos", "patience", "stalker", "bad comment", "abused", "cash back offer", "obsession", "videocall", "video call", "spam", "fudu", "jihaad", "photo leaked", "skinny", "mad", "in approprate", "fakecall", "soicla", "rmosur", "cyber staking", "dick", "gays", "using photos", "kill you", "go o hell", "fake massage", "nude posts", "commenting a cloths", "fake message", "fake posts", "riacism", "she cried a river", "nude post", "bustard", "landnudes", "hoe", "murdered", "btdk", "fake post", "curry", "abuse words", "senstive contact", "topyfake", "bakchadi", "bakchodi", "gold digger", "chod", "chomu", "purchase my assest", "shit eating indian", "fake offers", "i kill you", "chatiya", "cash", "fatso", "send me ransom", "click on link", "you are worthfull", "low energy", "ladla", "online banking", "chat with me", "i kill myself", "body shame", "baby shaming", "impact", "mcar", "fool", "run you", "faggot", "sexuality", "bandar", "try to understand", "bad comments", "nude pics", "free easy job with good package", "harassment", "hatred", "you have won a lottery", "trading", "khalistan", "others", "dark", "fakechat", "reject", "abusive word", "egergavis", "fake people dm", "vulgar message", "Vulgar ness", "bahen chood", "teasing", "discrimination", "religion", "it is killing me", "fucuer", "booharmil", "being absent in class", "staging office", "job scam links", "lgbt", "abusive words", "shitoff", "mkc", "terriost", "bupendra jogi", "fraud message", "hardcore", "lorny", "penchod", "moye moye", "chek my ass", "brown road shitter", "fake contact", "hack", "shabal gang", "yeah buddy", "bhenke lode", "bc mc", "i got your personal data", "hack account", "fatass", "body shaming", "raid", "scams", "durgs", "fraud call", "boby shaming", "shame", "laodayia", "lawder", "loude", "gmm", "sali", "ghada", "vulgur", "kusa", "stupid", "dumb", "fat", "kill", "shut up", "worthless"])

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
    matched_words = tokens.intersection(cyberbully_words)
    return matched_words

def main():
    df = pd.read_csv("/content/anti-bully.csv", encoding="latin-1")
    df['clean_text'] = df['text'].apply(preprocess_text)
    df = add_features(df, 'clean_text')

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2, max_df=0.95)
    X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

    scaler = StandardScaler()
    numerical_features = df[['sentiment', 'text_length']].fillna(0)
    scaled_features = scaler.fit_transform(numerical_features)
    X_combined = hstack([X_tfidf, csr_matrix(scaled_features)])

    X_train, X_test, y_train, y_test = train_test_split(X_combined, df['label'], test_size=0.2, random_state=42)

    classifiers = {
        'lsvc': LinearSVC(random_state=42),
        'lr': LogisticRegression(random_state=42, n_jobs=-1),
        'rf': RandomForestClassifier(random_state=42, n_jobs=-1),
        'dt': DecisionTreeClassifier(random_state=42),
        'ann': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    best_models = {}


    for name, clf in classifiers.items():
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
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")

    del X_combined
    gc.collect()

    weights = {'lsvc': 0.2, 'lr': 0.2, 'rf': 0.2, 'dt': 0.2, 'ann': 0.2}
    predictions = []

    for name, model in best_models.items():
        pred = model.predict(X_test)
        predictions.append(pred * weights[name])

    final_prediction = np.round(np.sum(predictions, axis=0)).astype(int)

    accuracy = accuracy_score(y_test, final_prediction)
    precision = precision_score(y_test, final_prediction, average='weighted')
    recall = recall_score(y_test, final_prediction, average='weighted')
    f1 = f1_score(y_test, final_prediction, average='weighted')
    roc_auc = roc_auc_score(y_test, final_prediction)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

    user_input = input("Enter a message to check if it's cyberbullying or not: ")
    cleaned_input = preprocess_text(user_input)
    sentiment = TextBlob(cleaned_input).sentiment.polarity
    text_length = len(cleaned_input)

    matched_bow = check_bag_of_words(user_input)
    if matched_bow:
        print(f"⚠️ The message contains potential cyberbullying words: {', '.join(matched_bow)}")

    input_tfidf = tfidf_vectorizer.transform([cleaned_input])
    input_df = pd.DataFrame([[sentiment, text_length]], columns=['sentiment', 'text_length'])
    input_scaled_features = scaler.transform(input_df)
    input_combined = hstack([input_tfidf, csr_matrix(input_scaled_features)])

    user_predictions = []
    for name, model in best_models.items():
        pred = model.predict(input_combined)
        user_predictions.append(pred * weights[name])
    final_user_prediction = np.round(np.sum(user_predictions, axis=0)).astype(int)

    if final_user_prediction[0] == 1:
        print("⚠️ The message is NOT cyberbullying.")
    else:
        print("✅ The message is predicted to be CYBERBULLYING. ")

if __name__ == "__main__":
    main()