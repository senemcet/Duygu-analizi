import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Tkinter kullanımını devre dışı bırak
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score

#veri setini oku
file_path = "magaza_yorumlari.csv"
df = pd.read_csv(file_path, encoding="utf-16")

#İlk 5 satırı görüntüle
print("İlk 5 Satır:")
print(df.head())


# Etiketleri düzenle (Olumlu - 1, Olumsuz - 0)
df["Durum"] = df["Durum"].map({"Olumlu": 1, "Olumsuz": 0})

# Yorumları küçük harfe çevir
df["Görüş"] = df["Görüş"].str.lower().fillna("")

#Örnek 5 yorum
print("\nÖrnek 5 Yorum:")
print(df[['Görüş', 'Durum']].sample(5))

#Duygu etiketlerinin dağılımı
print("\nDuygu Etiketi Dağılımı:")
print(df['Durum'].value_counts())

#Yüzdelik dağılım
print("\n Duygu Etiketi Yüzdelik Dağılımı:")
print(df['Durum'].value_counts(normalize=True) * 100)

# Türkçe stopword temizleme
stop_words = [
    "mı", "ancak", "ve", "veya", "bir", "bu", "şu", "çok", "daha", "gibi", "için",
    "ama", "ben", "sen", "biz", "siz", "o", "mu", "mü", "mı", "de", "da",
    "ile", "ne", "neden", "çünkü", "kadar", "yani", "ise", "bile", "artık"
]

for word in stop_words:
    df["Görüş"] = df["Görüş"].str.replace(f" {word} ", " ", regex=True)


# TF-IDF ile özellik çıkarma
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["Görüş"].values.astype("U"))
y = df["Durum"]

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Modeli eğit (Lojistik Regresyon)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Modelin tahmin yapması
y_pred = clf.predict(X_test)

# Modelin performansını değerlendir
print("\nModel Başarı Raporu:\n", classification_report(y_test, y_pred))

# Doğruluk skoru
score = clf.score(X_test, y_test)
print(f"\nModel Doğruluk Skoru: {score:.2f}")

# Confusion Matrix çizdir
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=0.1, square=True, cmap="Blues")
plt.ylabel("Gerçek Değer")
plt.xlabel("Tahmin Edilen Değer")
plt.title(f"Confusion Matrix (Accuracy: {score:.2f})")
plt.show()
plt.savefig("confusion_matrix.png")  # Confusion Matrix grafiğini kaydet
plt.savefig("precision_recall_curve.png")  # Precision-Recall grafiğini kaydet


# Precision-Recall Curve çizdir
y_pred_prob = clf.predict_proba(X_test)[:, 1]
prec, recall, _ = precision_recall_curve(y_test, y_pred_prob)
avg_prec = average_precision_score(y_test, y_pred_prob)

plt.figure()
plt.plot(recall, prec, color="blue", lw=2, label="Precision-Recall curve (AP = %0.2f)" % avg_prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()

# Kullanıcıdan yorum al ve tahmin yap
def tahmin_et(yorum):
    yorum = yorum.lower()
    for word in stop_words:
        yorum = yorum.replace(f" {word} ", " ")
    
    yorum_tfidf = vectorizer.transform([yorum])
    tahmin = clf.predict(yorum_tfidf)[0]
    return "Olumlu" if tahmin == 1 else "Olumsuz"

while True:
    giris = input("\nBir yorum girin (Çıkmak için 't' tuşuna basın): ")
    if giris.lower() == "t":
        break
    sonuc = tahmin_et(giris)
    print(f"Tahmin: {sonuc}")
