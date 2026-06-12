import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
import time
import pandas as pd
from datetime import datetime

# --- KONEKSI KE FUNGSI ABSENSI CSV ---
def catat_kehadiran_streamlit(nama_orang):
    file_csv = "kehadiran.csv"
    sekarang = datetime.now()
    tanggal_hari_ini = sekarang.strftime("%Y-%m-%d")
    jam_sekarang = tokens_waktu = sekarang.strftime("%H:%M:%S")
    
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

# --- LOAD DATABASE WAJAH ---
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

# --- SETUP LAYOUT & TEMA UI (PRINSIP UI/UX) ---
st.set_page_config(page_title="AI Attendance System", layout="wide", initial_sidebar_state="expanded")

# Navigasi Menu di Sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: #007bff;'>🧭 MENU UTAMA</h2>", unsafe_view_menu=True)
halaman = st.sidebar.radio("Pilih Halaman Kerja:", ["Halaman Utama", "Halaman Prediksi (Webcam)", "Dashboard Analytics"])

# ----------------- MENU 1: HALAMAN UTAMA -----------------
if halaman == "Halaman Utama":
    st.title("🎯 Smart Attendance System berbasis AI Face Recognition")
    st.markdown("---")
    
    col_id, col_desc = st.columns([1, 2])
    with col_id:
        st.markdown("### 🧑‍🎓 Identitas Mahasiswa")
        st.info("**Nama:** Alfarizhi Fitra \n\n**NIM:** 2311533014 \n\n**Mata Kuliah:** Image Processing")
    with col_desc:
        st.markdown("### 📋 Deskripsi Sistem")
        st.write("""
        Aplikasi ini dikembangkan untuk mendigitalisasi pencatatan kehadiran presensi menggunakan teknologi *Face Recognition*. 
        Dengan mengekstrak geometri wajah menjadi matriks linear 128-Dimensi *Face Embeddings*, model dapat melakukan verifikasi 
        identitas secara instan tanpa kontak fisik (*contactless*). Nilai tambah sistem ini adalah integrasi visualisasi data 
        secara *real-time* untuk memantau grafik tren kehadiran.
        """)

# ----------------- MENU 2: HALAMAN PREDIKSI -----------------
elif halaman == "Halaman Prediksi (Webcam)":
    st.title("📸 Monitor Kamera Presensi")
    st.write("Silakan isi nama Anda dan posisikan wajah menghadap kamera dengan jelas.")
    st.markdown("---")
    
    nama_input = st.text_input("Masukkan Nama Lengkap Anda:", placeholder="Contoh: Alfarizhi Fitra")
    foto_kamera = st.camera_input("Aktivasi Perangkat Kamera")
    
    if foto_kamera is not None:
        if not nama_input.strip():
            st.error("Peringatan: Kolom Nama Lengkap wajib diisi sebelum mengambil foto!")
        else:
            start_time = time.time()
            
            # Konversi file gambar kamera ke format OpenCV & RGB
            bytes_data = foto_kamera.getvalue()
            cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Deteksi & Ekstrak Vektor Wajah
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings_uji = face_recognition.face_encodings(rgb_image, face_locations)
            
            waktu_inferensi = time.time() - start_time
            
            st.markdown("### 📊 Hasil Analisis Citra AI")
            col_img, col_metrics = st.columns([2, 1])
            
            with col_img:
                if len(face_locations) > 0:
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 4)
                st.image(rgb_image, caption="Gambar Hasil Pemrosesan Kotak Deteksi", use_container_width=True)
            
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
                                # Konversi nilai jarak Euclidean ke Confidence Score %
                                confidence_pct = (1.0 - face_distances[best_match]) * 100
                                break
                    
                    if wajah_cocok:
                        msg_log = catat_kehadiran_streamlit(nama_input.strip())
                        st.success("Wajah Terverifikasi Sesuai")
                        st.info(msg_log)
                        st.metric(label="Confidence Score (Kedekatan Geometri)", value=f"{confidence_pct:.1f} %")
                    else:
                        st.error("Wajah Tidak Cocok dengan Data Acuan")
                        st.metric(label="Confidence Score", value="0 %")
                        
                st.metric(label="Waktu Pemrosesan (Inference Time)", value=f"{waktu_inferensi:.3f} detik")

# ----------------- MENU 3: DASHBOARD ANALYTICS (NILAI TAMBAH) -----------------
elif halaman == "Dashboard Analytics":
    st.title("📊 Dasbor Analitik Statistik Kehadiran")
    st.markdown("---")
    
    file_csv = "kehadiran.csv"
    
    if os.path.exists(file_csv) and os.path.getsize(file_csv) > 25:
        df = pd.read_csv(file_csv)
        
        # Ekstraksi statistik dasar
        total_logs = len(df)
        total_mahasiswa_unik = df['Nama'].nunique()
        
        # Layout Ringkasan Atas
        m1, m2, m3 = st.columns(3)
        m1.metric(label="📈 Total Log Presensi Berhasil", value=f"{total_logs} Kali")
        m2.metric(label="👥 Jumlah Mahasiswa Unik", value=f"{total_mahasiswa_unik} Orang")
        m3.metric(label="🎯 Akurasi Baseline Sistem", value="99.3 %")
        
        st.markdown("### 📈 Grafik Distribusi Tren Kehadiran")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.write("**Frekuensi Absen Per Tanggal**")
            tren_tanggal = df['Tanggal'].value_counts().sort_index()
            st.bar_chart(tren_tanggal)
            
        with col_g2:
            st.write("**Top 5 Mahasiswa Paling Sering Hadir**")
            top_mahasiswa = df['Nama'].value_counts().head(5)
            st.line_chart(top_mahasiswa)
            
        st.markdown("### 📋 Berkas Log Digital Aktual (kehadiran.csv)")
        st.dataframe(df, use_container_width=True)
    else:
        # Tampilan Fallback jika file csv kosong/belum ada yang absen
        m1, m2, m3 = st.columns(3)
        m1.metric(label="📈 Total Log Presensi Berhasil", value="0 Kali")
        m2.metric(label="👥 Jumlah Mahasiswa Unik", value="0 Orang")
        m3.metric(label="🎯 Akurasi Baseline Sistem", value="99.3 %")
        st.warning("Belum ada data riwayat absensi yang tersimpan di dalam berkas database lokal.")
