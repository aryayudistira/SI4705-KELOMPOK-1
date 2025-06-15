import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score

st.set_page_config(page_title="Prediksi Dampak Media Sosial", layout="wide")

st.title("📱 Prediksi Dampak Media Sosial terhadap Performa Akademik Mahasiswa")

# Upload file
uploaded_file = st.file_uploader("📂 Upload dataset CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop kolom ID jika ada
    df = df.drop(columns=["Student_ID"], errors='ignore')

    # Encode kategorikal
    categorical_cols = ["Gender", "Academic_Level", "Most_Used_Platform", "Relationship_Status"]
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])

    # Encode target biner
    if "Affects_Academic_Performance" in df.columns:
        df["Affects_Academic_Performance"] = df["Affects_Academic_Performance"].map({"Yes": 1, "No": 0})

    # ------------------- 🔹 UNSUPERVISED LEARNING (CLUSTERING) -------------------
    st.subheader("🔹 Unsupervised Learning - Klasterisasi Mahasiswa")

    cluster_features = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Conflicts_Over_Social_Media",
        "Addicted_Score"
    ]

    st.subheader("🔹 Unsupervised Learning - Klasterisasi Mahasiswa Cluster 3")
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster_Label"] = kmeans.fit_predict(X_cluster)

    # Visualisasi klaster
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Addicted_Score", y="Avg_Daily_Usage_Hours", hue="Cluster_Label", data=df, palette="Set2", ax=ax1)
    ax1.set_title("Visualisasi Klaster Mahasiswa")
    st.pyplot(fig1)

    # --- Statistik Karakteristik Tiap Cluster ---
    st.markdown("### 📊 Statistik Karakteristik per Klaster")
    cluster_summary = df.groupby("Cluster_Label")[cluster_features].mean()
    cluster_counts = df["Cluster_Label"].value_counts().sort_index()
    cluster_summary["Jumlah Anggota"] = cluster_counts
    st.dataframe(cluster_summary.style.format("{:.2f}"))

    # --- Visualisasi Boxplot Fitur per Cluster ---
    st.markdown("### 🎨 Distribusi Fitur per Klaster")
    fig_box, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(cluster_features):
        sns.boxplot(x="Cluster_Label", y=feature, data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f"Distribusi {feature} per Klaster")
    st.pyplot(fig_box)

    # --- Evaluasi Kualitas Klaster ---
    sil_score = silhouette_score(X_cluster, df["Cluster_Label"])
    st.markdown(f"### 📈 Evaluasi Klasterisasi\n*Silhouette Score:* {sil_score:.3f}")

    # --- Analisis Korelasi Fitur ---
    st.markdown("### 🔗 Korelasi Antar Fitur")
    corr = df[cluster_features + ["Affects_Academic_Performance"]].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Matriks Korelasi Fitur & Target")
    st.pyplot(fig_corr)

    st.subheader("🔹 Unsupervised Learning - Klasterisasi Mahasiswa Cluster 4")
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["Cluster_Label"] = kmeans.fit_predict(X_cluster)

    # Visualisasi klaster
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Addicted_Score", y="Avg_Daily_Usage_Hours", hue="Cluster_Label", data=df, palette="Set2", ax=ax1)
    ax1.set_title("Visualisasi Klaster Mahasiswa")
    st.pyplot(fig1)

    # --- Statistik Karakteristik Tiap Cluster ---
    st.markdown("### 📊 Statistik Karakteristik per Klaster")
    cluster_summary = df.groupby("Cluster_Label")[cluster_features].mean()
    cluster_counts = df["Cluster_Label"].value_counts().sort_index()
    cluster_summary["Jumlah Anggota"] = cluster_counts
    st.dataframe(cluster_summary.style.format("{:.2f}"))

    # --- Visualisasi Boxplot Fitur per Cluster ---
    st.markdown("### 🎨 Distribusi Fitur per Klaster")
    fig_box, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(cluster_features):
        sns.boxplot(x="Cluster_Label", y=feature, data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f"Distribusi {feature} per Klaster")
    st.pyplot(fig_box)

    # --- Evaluasi Kualitas Klaster ---
    sil_score = silhouette_score(X_cluster, df["Cluster_Label"])
    st.markdown(f"### 📈 Evaluasi Klasterisasi\n*Silhouette Score:* {sil_score:.3f}")

    # --- Analisis Korelasi Fitur ---
    st.markdown("### 🔗 Korelasi Antar Fitur")
    corr = df[cluster_features + ["Affects_Academic_Performance"]].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Matriks Korelasi Fitur & Target")
    st.pyplot(fig_corr)

    st.subheader("🔹 Unsupervised Learning - Klasterisasi Mahasiswa Cluster 5")
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["Cluster_Label"] = kmeans.fit_predict(X_cluster)

    # Visualisasi klaster
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Addicted_Score", y="Avg_Daily_Usage_Hours", hue="Cluster_Label", data=df, palette="Set2", ax=ax1)
    ax1.set_title("Visualisasi Klaster Mahasiswa")
    st.pyplot(fig1)

    # --- Statistik Karakteristik Tiap Cluster ---
    st.markdown("### 📊 Statistik Karakteristik per Klaster")
    cluster_summary = df.groupby("Cluster_Label")[cluster_features].mean()
    cluster_counts = df["Cluster_Label"].value_counts().sort_index()
    cluster_summary["Jumlah Anggota"] = cluster_counts
    st.dataframe(cluster_summary.style.format("{:.2f}"))

    # --- Visualisasi Boxplot Fitur per Cluster ---
    st.markdown("### 🎨 Distribusi Fitur per Klaster")
    fig_box, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(cluster_features):
        sns.boxplot(x="Cluster_Label", y=feature, data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f"Distribusi {feature} per Klaster")
    st.pyplot(fig_box)

    # --- Evaluasi Kualitas Klaster ---
    sil_score = silhouette_score(X_cluster, df["Cluster_Label"])
    st.markdown(f"### 📈 Evaluasi Klasterisasi\n*Silhouette Score:* {sil_score:.3f}")

    # --- Analisis Korelasi Fitur ---
    st.markdown("### 🔗 Korelasi Antar Fitur")
    corr = df[cluster_features + ["Affects_Academic_Performance"]].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Matriks Korelasi Fitur & Target")
    st.pyplot(fig_corr)

    st.subheader("🔹 Unsupervised Learning - Klasterisasi Mahasiswa Cluster 6")
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=6, random_state=42)
    df["Cluster_Label"] = kmeans.fit_predict(X_cluster)

    # Visualisasi klaster
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x="Addicted_Score", y="Avg_Daily_Usage_Hours", hue="Cluster_Label", data=df, palette="Set2", ax=ax1)
    ax1.set_title("Visualisasi Klaster Mahasiswa")
    st.pyplot(fig1)

    # --- Statistik Karakteristik Tiap Cluster ---
    st.markdown("### 📊 Statistik Karakteristik per Klaster")
    cluster_summary = df.groupby("Cluster_Label")[cluster_features].mean()
    cluster_counts = df["Cluster_Label"].value_counts().sort_index()
    cluster_summary["Jumlah Anggota"] = cluster_counts
    st.dataframe(cluster_summary.style.format("{:.2f}"))

    # --- Visualisasi Boxplot Fitur per Cluster ---
    st.markdown("### 🎨 Distribusi Fitur per Klaster")
    fig_box, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(cluster_features):
        sns.boxplot(x="Cluster_Label", y=feature, data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f"Distribusi {feature} per Klaster")
    st.pyplot(fig_box)

    # --- Evaluasi Kualitas Klaster ---
    sil_score = silhouette_score(X_cluster, df["Cluster_Label"])
    st.markdown(f"### 📈 Evaluasi Klasterisasi\n*Silhouette Score:* {sil_score:.3f}")

    # --- Analisis Korelasi Fitur ---
    st.markdown("### 🔗 Korelasi Antar Fitur")
    corr = df[cluster_features + ["Affects_Academic_Performance"]].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Matriks Korelasi Fitur & Target")
    st.pyplot(fig_corr)

    # ------------------- 🔸 SUPERVISED LEARNING (LOGISTIC REGRESSION) -------------------
    split_ratios = {
    "Anggota 1 (70:30)": 0.3,
    "Anggota 2 (80:20)": 0.2,
    "Anggota 3 (65:35)": 0.35,
    "Anggota 4 (75:25)": 0.25
}
    for anggota, test_ratio in split_ratios.items():
        st.subheader(f"🔸 Supervised Learning - {anggota}")

        # Train model excluding derived features
        X = df.drop(columns=["Affects_Academic_Performance", "Cluster_Label", "Addicted_Score"])
        y = df["Affects_Academic_Performance"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Classification report
        st.markdown("📌 Classification Report:")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)

        # Visualisasi Classification Report - Bar Chart
        st.markdown("📊 Bar Chart Classification Report (Kelas 0 & 1):")
        report_vis = report_df.loc[["0", "1"], ["precision", "recall", "f1-score"]]
        fig2, ax2 = plt.subplots()
        report_vis.plot(kind='bar', ax=ax2)
        ax2.set_title("Performance per Class")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.1)
        st.pyplot(fig2)

        # Visualisasi Classification Report - Heatmap
        st.markdown("🌡 Heatmap Classification Report:")
        fig3, ax3 = plt.subplots()
        sns.heatmap(report_vis, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax3)
        ax3.set_title("Classification Metrics Heatmap")
        st.pyplot(fig3)

        # Confusion Matrix
        st.markdown("🧮 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig4, ax4 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak", "Ya"], yticklabels=["Tidak", "Ya"])
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
        ax4.set_title("Confusion Matrix")
        st.pyplot(fig4)

        # Visualisasi Hubungan Fitur vs Dampak Akademik
        st.markdown("### 📉 Hubungan Fitur Utama dengan Dampak Akademik")
        fig_rel, axes_rel = plt.subplots(1, 2, figsize=(12, 5))

        sns.boxplot(x="Affects_Academic_Performance", y="Avg_Daily_Usage_Hours", data=df, ax=axes_rel[0], palette="Set3")
        axes_rel[0].set_title("Jam Penggunaan Harian vs Dampak Akademik")
        axes_rel[0].set_xticklabels(["Tidak Terpengaruh", "Terpengaruh"])

        sns.boxplot(x="Affects_Academic_Performance", y="Sleep_Hours_Per_Night", data=df, ax=axes_rel[1], palette="Set3")
        axes_rel[1].set_title("Jam Tidur per Malam vs Dampak Akademik")
        axes_rel[1].set_xticklabels(["Tidak Terpengaruh", "Terpengaruh"])

        st.pyplot(fig_rel)

    # Validation Form
    st.subheader("📋 Kepribadian dan Gaya Hidup")
    with st.form(key="validation_form"):
        age = st.number_input("Umur", min_value=18, max_value=24, value=19)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        academic_level = st.selectbox("Tingkat Akademik", ["High School", "Undergraduate", "Graduate"])
        avg_daily_usage_hours = st.slider("Rata-rata Penggunaan Harian (jam)", 1.0, 8.0, 5.0)
        most_used_platform = st.selectbox("Platform Paling Sering Digunakan", ["Instagram", "Twitter", "TikTok", "YouTube", "Facebook", "Snapchat", "LinkedIn", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat"])
        sleep_hours_per_night = st.slider("Jam Tidur per Malam", 4.0, 9.0, 6.0)
        relationship_status = st.selectbox("Status Hubungan", ["Single", "In Relationship", "Complicated"])
        conflicts_over_social_media = st.slider("Konflik akibat Media Sosial", 0, 5, 2)

        submit_button = st.form_submit_button(label="Prediksi Dampak")
            
    # Prediction based on form input
    if submit_button:
        st.subheader("🎓 Prediksi Dampak Akademik")
        form_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Academic_Level": [academic_level],
            "Avg_Daily_Usage_Hours": [avg_daily_usage_hours],
            "Most_Used_Platform": [most_used_platform],
            "Sleep_Hours_Per_Night": [sleep_hours_per_night],
            "Relationship_Status": [relationship_status],
            "Conflicts_Over_Social_Media": [conflicts_over_social_media]
        })

        # Encode form data
        categorical_cols = ["Gender", "Academic_Level", "Most_Used_Platform", "Relationship_Status"]
        form_data_encoded = pd.get_dummies(form_data, columns=categorical_cols)
        # Align with training data columns
        missing_cols = set(X.columns) - set(form_data_encoded.columns)
        for col in missing_cols:
            form_data_encoded[col] = 0
        form_data_encoded = form_data_encoded[X.columns]

        # Predict
        prediction = model.predict(form_data_encoded)
        impact = "Terpengaruh" if prediction[0] == 1 else "Tidak Terpengaruh"

        # Detailed reasoning
        if impact == "Terpengaruh":
            reason = (
                f"Berdasarkan input Anda, dampak media sosial terhadap performa akademik diprediksi {impact} karena: "
                f"- Rata-rata penggunaan harian ({avg_daily_usage_hours} jam) melebihi batas wajar yang dapat mengurangi waktu belajar."
                f"- Jam tidur per malam ({sleep_hours_per_night} jam) mungkin tidak cukup untuk pemulihan optimal."
                f"- Terdapat {conflicts_over_social_media} konflik akibat media sosial, yang dapat memengaruhi konsentrasi."
            )
        else:
            reason = (
                f"Berdasarkan input Anda, dampak media sosial terhadap performa akademik diprediksi {impact} karena: "
                f"- Rata-rata penggunaan harian ({avg_daily_usage_hours} jam) masih dalam batas yang dapat dikelola."
                f"- Jam tidur per malam ({sleep_hours_per_night} jam) cukup untuk mendukung produktivitas."
                f"- Hanya {conflicts_over_social_media} konflik akibat media sosial, menunjukkan dampak minimal."
            )
        st.write(f"Hasil Prediksi: {impact}")
        st.write(reason)

else:
    st.info("Silakan upload file CSV untuk melanjutkan.")