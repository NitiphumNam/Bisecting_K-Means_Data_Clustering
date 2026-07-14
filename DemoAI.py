import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# ==========================================
# 1. โหลดและเตรียมข้อมูลเบื้องต้น
# ==========================================
df = pd.read_csv('Dataset\\telecom_dataset.csv')

# แปลงข้อมูล Categorical เป็นตัวเลข
categorical_columns = ['State', 'International plan', 'Voice mail plan']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# ==========================================
# 2. แบ่งข้อมูลเป็นชุด Train และ Test (ข้อมูลใหม่)
# ==========================================
# สุ่มแบ่งข้อมูล 80% สำหรับเทรน และ 20% ไว้สำหรับทดสอบ (จำลองเป็นข้อมูลที่ไม่เคยเห็น)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

print(f"จำนวนข้อมูลชุด Train (ใช้สร้างโมเดล): {train_data.shape[0]} แถว")
print(f"จำนวนข้อมูลชุด Test (จำลองเป็นข้อมูลใหม่): {test_data.shape[0]} แถว\n")

# ==========================================
# 3. Data Preprocessing (Scaling)
# ==========================================
scaler = StandardScaler()
# กฎเหล็ก: Fit scaler ด้วยชุด Train เท่านั้น ป้องกันข้อมูล Leak ไปหาชุด Test
X_train_scaled = scaler.fit_transform(train_data)
# แปลงชุด Test ด้วยสเกลเดิมที่เรียนรู้จาก Train
X_test_scaled = scaler.transform(test_data)

# ==========================================
# 4. สร้างโมเดลและประเมินผล (Demo & Evaluation)
# ==========================================
n_clusters = 3
bisect_kmeans = BisectingKMeans(n_clusters=n_clusters, n_init=5, random_state=42)

# --- ส่วนที่ 1: Train Model ---
# ให้โมเดลเรียนรู้และจัดกลุ่มข้อมูลชุด Train
train_labels = bisect_kmeans.fit_predict(X_train_scaled)

# ประเมินผลชุด Train ด้วย Silhouette Score
train_silhouette = silhouette_score(X_train_scaled, train_labels)
print("--- ผลการประเมินชุด Train ---")
print(f"Silhouette Score (Train): {train_silhouette:.4f}\n")

# --- ส่วนที่ 2: ประยุกต์กับข้อมูลที่ไม่ใช่ชุด Train (Test Set) ---
# ใช้คำสั่ง predict() เพื่อจัดกลุ่มข้อมูลใหม่ โดยไม่ได้ fit ใหม่
test_labels = bisect_kmeans.predict(X_test_scaled)

# ประเมินผลชุด Test
test_silhouette = silhouette_score(X_test_scaled, test_labels)
print("--- ผลการประเมินชุด Test (Unseen Data) ---")
print(f"Silhouette Score (Test): {test_silhouette:.4f}")
print("(ถ้าค่าของ Test ใกล้เคียงกับ Train แสดงว่าโมเดลสามารถนำไปใช้กับข้อมูลจริงได้ดี)\n")

# นำผลลัพธ์ไปใส่กลับในตารางเพื่อให้เห็นผลลัพธ์เชิงธุรกิจ (Demo)
demo_result = test_data.copy()
demo_result['Predicted_Cluster'] = test_labels
print("--- ตัวอย่างผลลัพธ์การจัดกลุ่มลูกค้าใหม่ (5 แถวแรก) ---")
print(demo_result[['Total day minutes', 'Total intl calls', 'Customer service calls', 'Predicted_Cluster']].head())

# ==========================================
# 5. Visualization (แสดงกราฟผลลัพธ์จริง)
# ==========================================
# ใช้ PCA ลดมิติเหลือ 2 มิติเพื่อวาดกราฟ (Fit ด้วย Train เท่านั้นเช่นกัน)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
centroids_pca = pca.transform(bisect_kmeans.cluster_centers_)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

# กราฟฝั่งซ้าย: ข้อมูล Train
axes[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_labels, cmap='viridis', alpha=0.5, label='Train Data')
axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')
axes[0].set_title(f'Train Data Clusters (Score: {train_silhouette:.2f})')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend()

# กราฟฝั่งขวา: ข้อมูล Test (ข้อมูลที่ไม่ใช่ชุด Train)
axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=test_labels, cmap='viridis', alpha=0.8, marker='s', label='Test Data')
axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')
axes[1].set_title(f'Test Data Predictions (Score: {test_silhouette:.2f})')
axes[1].set_xlabel('Principal Component 1')
axes[1].legend()

plt.suptitle('Bisecting K-Means: Applying Model to Unseen Data', fontsize=16)
plt.tight_layout()
plt.show()