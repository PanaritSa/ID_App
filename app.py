import streamlit as st
st.set_page_config(page_title="Image Property Recommender", layout="wide")

import os
import io
import pickle
import sqlite3
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------- CONFIG ----------
image_size = (224, 224)
top_k = 5

# ---------- LOAD MODEL AND FEATURES ----------
@st.cache_resource
def load_model_and_data():
    model = load_model("efficientnet_feature_extractor.h5")
    features = np.load("features.npy")
    with open("image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)
    return model, features, image_paths

model, features, image_paths = load_model_and_data()

# ---------- FUNCTION: GET INFO FROM SQLITE ----------
def get_property_info(image_path):
    # เปิด connection ใหม่ทุกครั้ง (เพื่อแก้ปัญหา thread-safe)
    with sqlite3.connect("property_full_data_clickable.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT title, price, bedrooms, bathrooms, area 
            FROM properties 
            WHERE image_path = ?
            LIMIT 1
        """, (image_path,))
        row = cursor.fetchone()

    if row:
        return {
            "title": row[0],
            "price": row[1],
            "bedrooms": row[2],
            "bathrooms": row[3],
            "area": row[4]
        }
    return None

# ---------- PREPROCESS ----------
def preprocess_image(img):
    img = img.convert("RGB").resize(image_size)
    x = np.expand_dims(np.array(img), axis=0)
    return preprocess_input(x)

# ---------- SEARCH FUNCTION ----------
def search_similar_images(query_img):
    query_feature = model.predict(preprocess_image(query_img), verbose=0)[0]
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    nn.fit(features)
    distances, indices = nn.kneighbors([query_feature])

    results = []
    for i, idx in enumerate(indices[0]):
        img_path = image_paths[idx]
        sim = 1 - distances[0][i]
        sim_percent = round(sim * 100, 2)
        info = get_property_info(img_path)
        results.append({
            "img_path": img_path,
            "similarity": sim_percent,
            "info": info
        })
    return results

# ---------- STREAMLIT UI ----------
st.title("🏠 ระบบแนะนำอสังหาริมทรัพย์จากภาพ")

uploaded_file = st.file_uploader("📤 อัปโหลดภาพห้อง/บ้าน", type=["jpg", "jpeg", "png"])

if uploaded_file:
    query_img = Image.open(uploaded_file)
    st.image(query_img, caption="📷 ภาพที่อัปโหลด", use_column_width=False, width=300)

    st.subheader("🔍 ผลการค้นหาที่คล้ายกัน:")
    results = search_similar_images(query_img)

    for result in results:
        st.markdown("---")
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(result["img_path"], use_column_width=True)
        with cols[1]:
            info = result["info"]
            if info:
                st.markdown(f"### {info['title']}")
                st.markdown(f"💰 **{info['price']}**")
                st.markdown(f"🛏️ {info['bedrooms']} ห้องนอน | 🛁 {info['bathrooms']} ห้องน้ำ")
                st.markdown(f"📐 พื้นที่ใช้สอย: {info['area']}")
            else:
                st.markdown("_ไม่พบข้อมูลในฐานข้อมูล_")
            st.markdown(f"🎯 **ความคล้าย: {result['similarity']}%**")
