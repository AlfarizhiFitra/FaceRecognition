import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime

# --- KONEKSI KE FUNGSI ABSENSI CSV ---
def catat_kehadiran_streamlit(nama_orang):
    file_csv = "kehadiran.csv"
    sekarang = datetime.now()
    tanggal_hari_ini = sekarang.strftime("%Y-%m-%d")
    jam_sekarang = sekarang.strftime("%H:%M:%S")
    
    if not os.path.exists(file_csv):
        with open(file_csv, "w") as f:
            f.write("Nama,Tanggal,Jam Hadir\n")
            
    sudah_absen = False
    with open(file_csv, "r") as f:
        baris_data = f.readlines()
        for baris in baris_data:
            data = baris.strip().split(",")
            if len(data) >= 2 and data[0] == nama_orang and data[1] == tanggal_hari_ini:
                sudah_absen = True
                break
                
    if not sudah_absen:
        with open(file_csv, "a") as f:
            f.write(f"{nama_orang},{tanggal_hari_ini},{jam_sekarang}\n")
        return f"✅ ABSEN BERHASIL: {nama_orang} dicatat hadir pada {tanggal_hari_ini} - {jam_sekarang}"
    else:
        return f"⚠️ INFORMASI: {nama_orang} sudah melakukan absensi hari ini."

# --- LOAD DATABASE WAJAH SAAT APLIKASI DIKUNJUNGI ---
@st.cache_data
def load_database():
    known_face_encodings = []
    known_face_names = []
    DATABASE_DIR = "database_wajah"
    
    if os.path.exists(DATABASE_DIR):
        for file_name in os.listdir(DATABASE_DIR):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                name_identity = os.path.splitext(file_name)[0].replace('_', ' ')
                img_path = os.path.join(DATABASE_DIR, file_name)
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(name_identity)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_database()

# --- CONFIGURATION INTERFACE STREAMLIT ---
st.set_page_config(page_title="AI Attendance System", layout="wide")

# Navigasi Menu di Sidebar (Sesuai Prinsip UI/UX)
st.sidebar.title("🧭 Menu Navigasi")
halaman = st.sidebar.radio("Pilih Halaman:", ["Halaman Utama", "Halaman Prediksi", "Dashboard Statistik"])

# 1. HALAMAN UTAMA
if halaman == "Halaman Utama":
    st.title("🎯 AI Attendance System")
    st.subheader("Proyek Praktikum Image Processing")
    
    col_id, col_desc = st.columns([1, 2])
    with col_id:
        st.markdown("### 🧑‍🎓 Identitas Mahasiswa")
        st.info("**Nama:** Alfarizhi Fitra \n\n**NIM:** 2311533014 \n\n**Jurusan:** Sistem Informasi / FKG")
    with col_desc:
        st.markdown("### 📋 Deskripsi Proyek")
        st.write("""
        Sistem Absensi Berbasis AI ini menggunakan teknologi *Face Recognition* dengan pre-trained model berbasis *Deep Residual Network* (ResNet-34).
        Aplikasi ini memfasilitasi pencatatan kehadiran presensi secara digital, instan, dan nirkontak (*contactless*) guna meminimalkan celah manipulasi data atau kecurangan absensi konvensional.
        """)

# 2. HALAMAN PREDIKSI
elif halaman == "Halaman Prediksi":
    st.title("📸 Monitor Kamera Presensi")
    st.write("Silakan isi nama Anda dan posisikan wajah menghadap kamera dengan jelas.")
    
    nama_input = st.text_input("Masukkan Nama Lengkap Anda:", placeholder="Contoh: Alfarizhi Fitra")
    foto_kamera = st.camera_input("Aktivasi Kamera")
    
    if foto_kamera is not None:
        if not nama_input.strip():
            st.error("Peringatan: Kolom Nama Lengkap tidak boleh kosong!")
        else:
            start_time = time.time()
            
            # Konversi file gambar kamera ke format OpenCV & RGB
            bytes_data = foto_kamera.getvalue()
            cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Deteksi & Ekstrak Wajah
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings_uji = face_recognition.face_encodings(rgb_image, face_locations)
            
            waktu_inferensi = time.time() - start_time
            
            st.markdown("### 📊 Hasil Analisis AI")
            col_img, col_metrics = st.columns([2, 1])
            
            with col_img:
                if len(face_locations) > 0:
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 4)
                st.image(rgb_image, caption="Gambar Hasil Deteksi Kamera", use_container_width=True)
            
            with col_metrics:
                if len(face_locations) == 0:
                    st.error("Wajah Tidak Terdeteksi")
                    st.metric(label="Confidence Score", value="0 %")
                else:
                    wajah_cocok = False
                    confidence_pct = 0.0
                    
                    for face_encoding in face_encodings_uji:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:
                            best_match = np.argmin(face_distances)
                            if matches[best_match]:
                                wajah_cocok = True
                                # Konversi nilai jarak euclidean menjadi persentase kedekatan akurasi
                                confidence_pct = (1.0 - face_distances[best_match]) * 100
                                break
                    
                    if wajah_cocok:
                        msg_log = catat_kehadiran_streamlit(nama_input.strip())
                        st.success("Wajah Terverifikasi")
                        st.info(msg_log)
                        st.metric(label="Confidence Score", value=f"{confidence_pct:.1f} %")
                    else:
                        st.error("Wajah Tidak Cocok dengan Database")
                        st.metric(label="Confidence Score", value="0 %")
                        
                st.metric(label="Waktu Inferensi", value=f"{waktu_inferensi:.3f} detik")

# 3. DASHBOARD STATISTIK
elif halaman == "Dashboard Statistik":
    st.title("📊 Dasbor Analitik Kehadiran")
    
    file_csv = "kehadiran.csv"
    total_data = 0
    
    if os.path.exists(file_csv):
        with open(file_csv, "r") as f:
            total_data = len(f.readlines()) - 1
            if total_data < 0: total_data = 0
            
    # Tampilan kartu metrik responsif
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Jumlah Data Diuji (Log)", value=f"{total_data} Orang")
    m2.metric(label="Akurasi Model Baseline", value="99.3 %")
    m3.metric(label="Rata-rata Target Inferensi", value="~ 0.4 Detik")
    
    st.markdown("### 📋 Isi Berkas Riwayat Kehadiran (kehadiran.csv)")
    if total_data > 0:
        import pandas as pd
        df = pd.read_csv(file_csv)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Belum ada data absensi yang tercatat dalam log berkas.")
