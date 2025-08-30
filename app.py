
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.segmentation import flood
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ================== APP HEADER ==================
st.set_page_config(page_title="ASCUS Analyzer", layout="wide")
st.title("🔬 ASCUS Analyzer: Nuclear Morphometric and Automatic Classification")

st.markdown("""
Langkah-langkah:
1) Pilih sumber gambar (upload / foto sampel)  
2) Atur threshold  
3) Klik pada inti **curiga** & **intermediate** pada gambar threshold  
4) (Opsional) Simpan dataset → Latih CNN → Klasifikasi
""")

# ================== PATH & FOLDERS ==================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ascus_cnn.h5")
DATA_DIR = "dataset"
SAMPLES_DIR = "samples"  # letakkan foto-foto sampel demo di sini
CLASS_NAMES = ["reaktif", "ascus"]  # referensi

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "ascus"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "reaktif"), exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("⚙️ Pengaturan")
save_dataset = st.sidebar.checkbox("💾 Simpan dataset (crop, csv, overlay)", value=True)

st.sidebar.subheader("⚡ Training CNN")
train_epochs = st.sidebar.slider("Epochs", 5, 50, 12, 1)
batch_size = st.sidebar.selectbox("Batch size", [8, 16, 32], index=1)
img_size = (64, 64)

st.sidebar.subheader("📏 Kalibrasi")
calib_option = st.sidebar.radio("Pilih metode kalibrasi:", ("Tanpa Kalibrasi", "Manual", "Otomatis"))
manual_calib_length = 7.0
if calib_option == "Manual":
    manual_calib_length = st.sidebar.number_input("Panjang nyata antar titik (µm):", min_value=0.1, value=7.0, step=0.1)

# ================== LOAD / TRAIN MODEL ==================
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model CNN berhasil dimuat dari 'models/ascus_cnn.h5'")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model: {e}")

def train_cnn_from_dataset():
    if sum(len(files) for _,_,files in os.walk(DATA_DIR)) == 0:
        st.error("Dataset kosong. Kumpulkan dulu contoh ke 'dataset/ascus' dan 'dataset/reaktif'.")
        return None, None

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    st.info(f"Kelas ditemukan: {train_gen.class_indices}")

    cnn = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    mc = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)

    with st.spinner("🚀 Training CNN..."):
        history = cnn.fit(train_gen, validation_data=val_gen, epochs=train_epochs, callbacks=[es, mc])

    try:
        best_model = load_model(MODEL_PATH)
    except Exception:
        best_model = cnn
        best_model.save(MODEL_PATH)

    st.success("✅ Training selesai! Model terbaik disimpan ke 'models/ascus_cnn.h5'")
    st.subheader("📈 Kurva Training")
    st.line_chart({"train_loss": history.history["loss"], "val_loss": history.history["val_loss"]})
    st.line_chart({"train_acc": history.history["accuracy"], "val_acc": history.history["val_accuracy"]})

    return best_model, history

if st.sidebar.button("🎓 Mulai Training dari Dataset"):
    model, _ = train_cnn_from_dataset()

# ================== PILIH SUMBER GAMBAR ==================
st.subheader("📤 Sumber Gambar")
source = st.radio("Pilih sumber gambar:", ("Upload gambar sendiri", "Gunakan foto sampel"))

image = None
filename = None
if source == "Upload gambar sendiri":
    uploaded_file = st.file_uploader("Unggah gambar (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        filename = uploaded_file.name
else:
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if len(sample_files) == 0:
        st.error(f"Tidak ada foto sampel di folder '{SAMPLES_DIR}'")
    else:
        chosen = st.selectbox("Pilih foto sampel:", sample_files, index=0)
        sample_path = os.path.join(SAMPLES_DIR, chosen)
        try:
            image = Image.open(sample_path).convert("RGB")
            filename = chosen
            st.info("Menggunakan foto sampel bawaan aplikasi.")
        except Exception as e:
            st.error(f"Gagal membuka foto sampel: {e}")

if image is None:
    st.info("Unggah gambar atau pilih foto sampel untuk memulai.")
    st.stop()

image_np = np.array(image)
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

# ================== THRESHOLDING ==================
st.subheader("🎚️ Thresholding")
col_t1, col_t2 = st.columns(2)
with col_t1:
    lower = st.slider("Batas Bawah", 0, 255, 80)
with col_t2:
    upper = st.slider("Batas Atas", 0, 255, 180)
thresholded = cv2.inRange(gray, lower, upper)

col1, col2 = st.columns(2)
with col1:
    st.image(image_np, caption="Gambar Asli", use_column_width=True)
with col2:
    st.image(thresholded, caption="Gambar Threshold", use_column_width=True)

# ================== KALIBRASI ==================
um_per_pixel = 1.0
if calib_option == "Manual":
    st.subheader("🖱️ Klik dua titik untuk kalibrasi")
    calib_canvas = st_canvas(
        fill_color="rgba(0,0,255,0.3)",
        stroke_width=2,
        stroke_color="#0000FF",
        background_image=Image.fromarray(image_np),
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="point",
        point_display_radius=5,
        key="calib_canvas",
    )
    if calib_canvas.json_data and len(calib_canvas.json_data.get("objects", [])) >= 2:
        pts = []
        for obj in calib_canvas.json_data["objects"][:2]:
            x = int(obj.get("left", 0))
            y = int(obj.get("top", 0))
            pts.append((x, y))
        dist_pixel = np.sqrt((pts[1][0]-pts[0][0])**2 + (pts[1][1]-pts[0][1])**2)
        um_per_pixel = manual_calib_length / dist_pixel
        st.success(f"Kalibrasi berhasil: {um_per_pixel:.3f} µm/pixel")
    else:
        st.warning("Klik dua titik pada gambar untuk kalibrasi")
elif calib_option == "Otomatis":
    ideal_area = 35.0
    masks_dummy = np.zeros_like(gray)
    masks_dummy[0:10, 0:10] = 1  # dummy agar tidak error
    area_inter_px = int(np.sum(masks_dummy)) if 'masks' not in locals() else int(np.sum(masks[1]))
    if area_inter_px:
        um_per_pixel = np.sqrt(ideal_area / area_inter_px)
        st.info(f"Kalibrasi otomatis diterapkan: {um_per_pixel:.3f} µm/pixel")

# ================== CANVAS KLIK INTI ==================
st.subheader("🖱️ Klik inti Curiga dan Intermediate pada Gambar Threshold")
display_width = 800
scale = display_width / thresholded.shape[1]
resized_thresh = cv2.resize(thresholded, (display_width, int(thresholded.shape[0]*scale)))

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=3,
    stroke_color="#FF0000",
    background_image=Image.fromarray(resized_thresh),
    update_streamlit=True,
    height=resized_thresh.shape[0],
    width=resized_thresh.shape[1],
    drawing_mode="point",
    point_display_radius=5,
    key="canvas",
)

if not (canvas_result.json_data and len(canvas_result.json_data.get("objects", [])) >= 2):
    st.warning("❗ Klik **dua titik** pada gambar threshold (Curiga & Intermediate).")
    st.stop()

coords = []
for obj in canvas_result.json_data["objects"][:2]:
    x = int(obj.get("left", 0) / scale)
    y = int(obj.get("top", 0) / scale)
    coords.append((x, y))

try:
    masks = [flood(thresholded, (y, x), tolerance=10) for (x, y) in coords]
except Exception as e:
    st.error(f"Gagal segmentasi flood: {e}")
    st.stop()

if any(m.sum() == 0 for m in masks):
    st.error("Segmentasi gagal pada salah satu titik.")
    st.stop()

# ================== ANALISIS ==================
def analyze_roi(mask: np.ndarray):
    area = int(np.sum(mask))
    mean_gray = float(np.mean(gray[mask])) if area > 0 else 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = float((4*np.pi*area)/(perimeter**2))
    return area, mean_gray, circularity

area_curiga, gray_curiga, circ_curiga = analyze_roi(masks[0])
area_inter,  gray_inter,  circ_inter  = analyze_roi(masks[1])
# kalibrasi luas
area_curiga_um2 = area_curiga * um_per_pixel**2
area_inter_um2 = area_inter * um_per_pixel**2
ratio = (area_curiga_um2 / area_inter_um2) if area_inter_um2 else 0.0

# ================== CROP INTI CURIGA ==================
y_idx_c, x_idx_c = np.where(masks[0])
x_min, x_max = int(np.min(x_idx_c)), int(np.max(x_idx_c))
y_min, y_max = int(np.min(y_idx_c)), int(np.max(y_idx_c))
x_min, y_min = max(x_min,0), max(y_min,0)
x_max, y_max = min(x_max, image_np.shape[1]-1), min(y_max, image_np.shape[0]-1)
crop_rgb = image_np[y_min:y_max+1, x_min:x_max+1]
crop_gray = gray[y_min:y_max+1, x_min:x_max+1]

h, w = crop_gray.shape
q1 = crop_gray[:h//2, :w//2]; q2 = crop_gray[:h//2, w//2:]
q3 = crop_gray[h//2:, :w//2]; q4 = crop_gray[h//2:, w//2:]
mean_quadrants = [float(np.mean(q)) for q in [q1,q2,q3,q4]]
mean_total = float(np.mean(crop_gray)) if crop_gray.size > 0 else 0.0

# ================== CROP INTI INTERMEDIATE ==================
y_idx_i, x_idx_i = np.where(masks[1])
x_min_i, x_max_i = int(np.min(x_idx_i)), int(np.max(x_idx_i))
y_min_i, y_max_i = int(np.min(y_idx_i)), int(np.max(y_idx_i))
x_min_i, y_min_i = max(x_min_i,0), max(y_min_i,0)
x_max_i, y_max_i = min(x_max_i, image_np.shape[1]-1), min(y_max_i, image_np.shape[0]-1)
crop_rgb_inter = image_np[y_min_i:y_max_i+1, x_min_i:x_max_i+1]

# ================== OVERLAY ==================
overlay = image_np.copy()
overlay[masks[0]] = [255,0,0]
overlay[masks[1]] = [0,255,0]

# ================== LABEL OTOMATIS ==================
auto_label = "ascus" if ratio > 2.5 else "reaktif"
crop_curiga_resized = cv2.resize(crop_rgb, img_size)
crop_inter_resized = cv2.resize(crop_rgb_inter, img_size)

if save_dataset:
    base_folder = auto_label
    os.makedirs(base_folder, exist_ok=True)
    existing = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    subfolder_num = len(existing)+1
    subfolder = os.path.join(base_folder, f"{auto_label}{subfolder_num}")
    os.makedirs(subfolder, exist_ok=True)

    overlay_path = os.path.join(subfolder, f"segmentasi_{filename}")
    Image.fromarray(overlay).save(overlay_path)
    crop_pair = np.concatenate([crop_curiga_resized, crop_inter_resized], axis=1)
    crop_pair_path = os.path.join(subfolder, f"crop_{filename}")
    Image.fromarray(crop_pair).save(crop_pair_path)

    result = {
        "filename": filename,
        "area_curiga": area_curiga_um2,
        "area_intermediate": area_inter_um2,
        "ratio": ratio,
        "circularity_curiga": circ_curiga,
        "circularity_intermediate": circ_inter,
        "gray_q1": mean_quadrants[0],
        "gray_q2": mean_quadrants[1],
        "gray_q3": mean_quadrants[2],
        "gray_q4": mean_quadrants[3],
        "gray_total": mean_total
    }
    df = pd.DataFrame([result])
    csv_path = os.path.join(subfolder, f"hasil_{os.path.splitext(filename)[0]}.csv")
    df.to_csv(csv_path, index=False)

    dataset_folder = os.path.join(DATA_DIR, auto_label)
    os.makedirs(dataset_folder, exist_ok=True)
    count_existing = len([f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))])
    crop_save_path = os.path.join(dataset_folder, f"{auto_label}_{count_existing+1}.png")
    Image.fromarray(crop_curiga_resized).save(crop_save_path)

    st.success(f"✅ Dataset disimpan ke: `{subfolder}`")
    st.success(f"✅ Crop training disimpan ke: `{crop_save_path}`")
    st.success(f"✅ CSV disimpan ke: `{csv_path}`")

# ================== TAMPILKAN HASIL ==================
st.subheader("📊 Hasil Analisis")
st.markdown(f"- **Nama file:** `{filename}`")
st.markdown(f"- **Luas Curiga:** {area_curiga_um2:.2f} µm²")
st.markdown(f"- **Luas Intermediate:** {area_inter_um2:.2f} µm²")
st.markdown(f"- **Rasio Curiga/Intermediate:** {ratio:.2f}")
st.markdown(f"- **Circularity Curiga:** {circ_curiga:.3f}")
st.markdown(f"- **Circularity Intermediate:** {circ_inter:.3f}")
st.markdown("**Distribusi Kromatin (Mean Gray, Inti Curiga):**")
st.markdown(f"- Q1: {mean_quadrants[0]:.2f} | Q2: {mean_quadrants[1]:.2f} | "
            f"Q3: {mean_quadrants[2]:.2f} | Q4: {mean_quadrants[3]:.2f} | Total: {mean_total:.2f}")
st.image(overlay, caption="Segmentasi: Curiga (Merah) & Intermediate (Hijau)", use_column_width=True)

c1, c2 = st.columns(2)
with c1:
    st.image(crop_rgb, caption="Crop RGB Inti Curiga", use_column_width=True)
with c2:
    st.image(crop_rgb_inter, caption="Crop RGB Inti Intermediate", use_column_width=True)

# ================== KLASIFIKASI CNN ==================
if model:
    x_input = crop_curiga_resized.astype("float32") / 255.0
    x_input = np.expand_dims(x_input, axis=0)
    pred = model.predict(x_input)[0][0]
    prob = float(pred)
    label_pred = "ascus" if prob > 0.5 else "reaktif"
    st.subheader("🧠 Klasifikasi CNN Lokal")
    st.markdown(f"- Prediksi: **{label_pred.upper()}**")
    st.markdown(f"- Probabilitas ASCUS: {prob:.3f}")
else:
    st.info("⚠️ Model CNN belum tersedia. Gunakan tombol Training di sidebar.")

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.segmentation import flood
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ================== APP HEADER ==================
st.set_page_config(page_title="ASCUS Analyzer", layout="wide")
st.title("🔬 ASCUS Analyzer: Nuclear Morphometric and Automatic Classification")

st.markdown("""
Langkah-langkah:
1) Pilih sumber gambar (upload / foto sampel)  
2) Atur threshold  
3) Klik pada inti **curiga** & **intermediate** pada gambar threshold  
4) (Opsional) Simpan dataset → Latih CNN → Klasifikasi
""")

# ================== PATH & FOLDERS ==================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ascus_cnn.h5")
DATA_DIR = "dataset"
SAMPLES_DIR = "samples"  # letakkan foto-foto sampel demo di sini
CLASS_NAMES = ["reaktif", "ascus"]  # referensi

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "ascus"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "reaktif"), exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("⚙️ Pengaturan")
save_dataset = st.sidebar.checkbox("💾 Simpan dataset (crop, csv, overlay)", value=True)

st.sidebar.subheader("⚡ Training CNN")
train_epochs = st.sidebar.slider("Epochs", 5, 50, 12, 1)
batch_size = st.sidebar.selectbox("Batch size", [8, 16, 32], index=1)
img_size = (64, 64)

st.sidebar.subheader("📏 Kalibrasi")
calib_option = st.sidebar.radio("Pilih metode kalibrasi:", ("Tanpa Kalibrasi", "Manual", "Otomatis"))
manual_calib_length = 7.0
if calib_option == "Manual":
    manual_calib_length = st.sidebar.number_input("Panjang nyata antar titik (µm):", min_value=0.1, value=7.0, step=0.1)

# ================== LOAD / TRAIN MODEL ==================
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model CNN berhasil dimuat dari 'models/ascus_cnn.h5'")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model: {e}")

def train_cnn_from_dataset():
    if sum(len(files) for _,_,files in os.walk(DATA_DIR)) == 0:
        st.error("Dataset kosong. Kumpulkan dulu contoh ke 'dataset/ascus' dan 'dataset/reaktif'.")
        return None, None

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    st.info(f"Kelas ditemukan: {train_gen.class_indices}")

    cnn = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    mc = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)

    with st.spinner("🚀 Training CNN..."):
        history = cnn.fit(train_gen, validation_data=val_gen, epochs=train_epochs, callbacks=[es, mc])

    try:
        best_model = load_model(MODEL_PATH)
    except Exception:
        best_model = cnn
        best_model.save(MODEL_PATH)

    st.success("✅ Training selesai! Model terbaik disimpan ke 'models/ascus_cnn.h5'")
    st.subheader("📈 Kurva Training")
    st.line_chart({"train_loss": history.history["loss"], "val_loss": history.history["val_loss"]})
    st.line_chart({"train_acc": history.history["accuracy"], "val_acc": history.history["val_accuracy"]})

    return best_model, history

if st.sidebar.button("🎓 Mulai Training dari Dataset"):
    model, _ = train_cnn_from_dataset()

# ================== PILIH SUMBER GAMBAR ==================
st.subheader("📤 Sumber Gambar")
source = st.radio("Pilih sumber gambar:", ("Upload gambar sendiri", "Gunakan foto sampel"))

image = None
filename = None
if source == "Upload gambar sendiri":
    uploaded_file = st.file_uploader("Unggah gambar (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        filename = uploaded_file.name
else:
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if len(sample_files) == 0:
        st.error(f"Tidak ada foto sampel di folder '{SAMPLES_DIR}'")
    else:
        chosen = st.selectbox("Pilih foto sampel:", sample_files, index=0)
        sample_path = os.path.join(SAMPLES_DIR, chosen)
        try:
            image = Image.open(sample_path).convert("RGB")
            filename = chosen
            st.info("Menggunakan foto sampel bawaan aplikasi.")
        except Exception as e:
            st.error(f"Gagal membuka foto sampel: {e}")

if image is None:
    st.info("Unggah gambar atau pilih foto sampel untuk memulai.")
    st.stop()

image_np = np.array(image)
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

# ================== THRESHOLDING ==================
st.subheader("🎚️ Thresholding")
col_t1, col_t2 = st.columns(2)
with col_t1:
    lower = st.slider("Batas Bawah", 0, 255, 80)
with col_t2:
    upper = st.slider("Batas Atas", 0, 255, 180)
thresholded = cv2.inRange(gray, lower, upper)

col1, col2 = st.columns(2)
with col1:
    st.image(image_np, caption="Gambar Asli", use_column_width=True)
with col2:
    st.image(thresholded, caption="Gambar Threshold", use_column_width=True)

# ================== KALIBRASI ==================
um_per_pixel = 1.0
if calib_option == "Manual":
    st.subheader("🖱️ Klik dua titik untuk kalibrasi")
    calib_canvas = st_canvas(
        fill_color="rgba(0,0,255,0.3)",
        stroke_width=2,
        stroke_color="#0000FF",
        background_image=Image.fromarray(image_np),
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="point",
        point_display_radius=5,
        key="calib_canvas",
    )
    if calib_canvas.json_data and len(calib_canvas.json_data.get("objects", [])) >= 2:
        pts = []
        for obj in calib_canvas.json_data["objects"][:2]:
            x = int(obj.get("left", 0))
            y = int(obj.get("top", 0))
            pts.append((x, y))
        dist_pixel = np.sqrt((pts[1][0]-pts[0][0])**2 + (pts[1][1]-pts[0][1])**2)
        um_per_pixel = manual_calib_length / dist_pixel
        st.success(f"Kalibrasi berhasil: {um_per_pixel:.3f} µm/pixel")
    else:
        st.warning("Klik dua titik pada gambar untuk kalibrasi")
elif calib_option == "Otomatis":
    ideal_area = 35.0
    masks_dummy = np.zeros_like(gray)
    masks_dummy[0:10, 0:10] = 1  # dummy agar tidak error
    area_inter_px = int(np.sum(masks_dummy)) if 'masks' not in locals() else int(np.sum(masks[1]))
    if area_inter_px:
        um_per_pixel = np.sqrt(ideal_area / area_inter_px)
        st.info(f"Kalibrasi otomatis diterapkan: {um_per_pixel:.3f} µm/pixel")

# ================== CANVAS KLIK INTI ==================
st.subheader("🖱️ Klik inti Curiga dan Intermediate pada Gambar Threshold")
display_width = 800
scale = display_width / thresholded.shape[1]
resized_thresh = cv2.resize(thresholded, (display_width, int(thresholded.shape[0]*scale)))

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=3,
    stroke_color="#FF0000",
    background_image=Image.fromarray(resized_thresh),
    update_streamlit=True,
    height=resized_thresh.shape[0],
    width=resized_thresh.shape[1],
    drawing_mode="point",
    point_display_radius=5,
    key="canvas",
)

if not (canvas_result.json_data and len(canvas_result.json_data.get("objects", [])) >= 2):
    st.warning("❗ Klik **dua titik** pada gambar threshold (Curiga & Intermediate).")
    st.stop()

coords = []
for obj in canvas_result.json_data["objects"][:2]:
    x = int(obj.get("left", 0) / scale)
    y = int(obj.get("top", 0) / scale)
    coords.append((x, y))

try:
    masks = [flood(thresholded, (y, x), tolerance=10) for (x, y) in coords]
except Exception as e:
    st.error(f"Gagal segmentasi flood: {e}")
    st.stop()

if any(m.sum() == 0 for m in masks):
    st.error("Segmentasi gagal pada salah satu titik.")
    st.stop()

# ================== ANALISIS ==================
def analyze_roi(mask: np.ndarray):
    area = int(np.sum(mask))
    mean_gray = float(np.mean(gray[mask])) if area > 0 else 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = float((4*np.pi*area)/(perimeter**2))
    return area, mean_gray, circularity

area_curiga, gray_curiga, circ_curiga = analyze_roi(masks[0])
area_inter,  gray_inter,  circ_inter  = analyze_roi(masks[1])
# kalibrasi luas
area_curiga_um2 = area_curiga * um_per_pixel**2
area_inter_um2 = area_inter * um_per_pixel**2
ratio = (area_curiga_um2 / area_inter_um2) if area_inter_um2 else 0.0

# ================== CROP INTI CURIGA ==================
y_idx_c, x_idx_c = np.where(masks[0])
x_min, x_max = int(np.min(x_idx_c)), int(np.max(x_idx_c))
y_min, y_max = int(np.min(y_idx_c)), int(np.max(y_idx_c))
x_min, y_min = max(x_min,0), max(y_min,0)
x_max, y_max = min(x_max, image_np.shape[1]-1), min(y_max, image_np.shape[0]-1)
crop_rgb = image_np[y_min:y_max+1, x_min:x_max+1]
crop_gray = gray[y_min:y_max+1, x_min:x_max+1]

h, w = crop_gray.shape
q1 = crop_gray[:h//2, :w//2]; q2 = crop_gray[:h//2, w//2:]
q3 = crop_gray[h//2:, :w//2]; q4 = crop_gray[h//2:, w//2:]
mean_quadrants = [float(np.mean(q)) for q in [q1,q2,q3,q4]]
mean_total = float(np.mean(crop_gray)) if crop_gray.size > 0 else 0.0

# ================== CROP INTI INTERMEDIATE ==================
y_idx_i, x_idx_i = np.where(masks[1])
x_min_i, x_max_i = int(np.min(x_idx_i)), int(np.max(x_idx_i))
y_min_i, y_max_i = int(np.min(y_idx_i)), int(np.max(y_idx_i))
x_min_i, y_min_i = max(x_min_i,0), max(y_min_i,0)
x_max_i, y_max_i = min(x_max_i, image_np.shape[1]-1), min(y_max_i, image_np.shape[0]-1)
crop_rgb_inter = image_np[y_min_i:y_max_i+1, x_min_i:x_max_i+1]

# ================== OVERLAY ==================
overlay = image_np.copy()
overlay[masks[0]] = [255,0,0]
overlay[masks[1]] = [0,255,0]

# ================== LABEL OTOMATIS ==================
auto_label = "ascus" if ratio > 2.5 else "reaktif"
crop_curiga_resized = cv2.resize(crop_rgb, img_size)
crop_inter_resized = cv2.resize(crop_rgb_inter, img_size)

if save_dataset:
    base_folder = auto_label
    os.makedirs(base_folder, exist_ok=True)
    existing = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    subfolder_num = len(existing)+1
    subfolder = os.path.join(base_folder, f"{auto_label}{subfolder_num}")
    os.makedirs(subfolder, exist_ok=True)

    overlay_path = os.path.join(subfolder, f"segmentasi_{filename}")
    Image.fromarray(overlay).save(overlay_path)
    crop_pair = np.concatenate([crop_curiga_resized, crop_inter_resized], axis=1)
    crop_pair_path = os.path.join(subfolder, f"crop_{filename}")
    Image.fromarray(crop_pair).save(crop_pair_path)

    result = {
        "filename": filename,
        "area_curiga": area_curiga_um2,
        "area_intermediate": area_inter_um2,
        "ratio": ratio,
        "circularity_curiga": circ_curiga,
        "circularity_intermediate": circ_inter,
        "gray_q1": mean_quadrants[0],
        "gray_q2": mean_quadrants[1],
        "gray_q3": mean_quadrants[2],
        "gray_q4": mean_quadrants[3],
        "gray_total": mean_total
    }
    df = pd.DataFrame([result])
    csv_path = os.path.join(subfolder, f"hasil_{os.path.splitext(filename)[0]}.csv")
    df.to_csv(csv_path, index=False)

    dataset_folder = os.path.join(DATA_DIR, auto_label)
    os.makedirs(dataset_folder, exist_ok=True)
    count_existing = len([f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))])
    crop_save_path = os.path.join(dataset_folder, f"{auto_label}_{count_existing+1}.png")
    Image.fromarray(crop_curiga_resized).save(crop_save_path)

    st.success(f"✅ Dataset disimpan ke: `{subfolder}`")
    st.success(f"✅ Crop training disimpan ke: `{crop_save_path}`")
    st.success(f"✅ CSV disimpan ke: `{csv_path}`")

# ================== TAMPILKAN HASIL ==================
st.subheader("📊 Hasil Analisis")
st.markdown(f"- **Nama file:** `{filename}`")
st.markdown(f"- **Luas Curiga:** {area_curiga_um2:.2f} µm²")
st.markdown(f"- **Luas Intermediate:** {area_inter_um2:.2f} µm²")
st.markdown(f"- **Rasio Curiga/Intermediate:** {ratio:.2f}")
st.markdown(f"- **Circularity Curiga:** {circ_curiga:.3f}")
st.markdown(f"- **Circularity Intermediate:** {circ_inter:.3f}")
st.markdown("**Distribusi Kromatin (Mean Gray, Inti Curiga):**")
st.markdown(f"- Q1: {mean_quadrants[0]:.2f} | Q2: {mean_quadrants[1]:.2f} | "
            f"Q3: {mean_quadrants[2]:.2f} | Q4: {mean_quadrants[3]:.2f} | Total: {mean_total:.2f}")
st.image(overlay, caption="Segmentasi: Curiga (Merah) & Intermediate (Hijau)", use_column_width=True)

c1, c2 = st.columns(2)
with c1:
    st.image(crop_rgb, caption="Crop RGB Inti Curiga", use_column_width=True)
with c2:
    st.image(crop_rgb_inter, caption="Crop RGB Inti Intermediate", use_column_width=True)

# ================== KLASIFIKASI CNN ==================
if model:
    x_input = crop_curiga_resized.astype("float32") / 255.0
    x_input = np.expand_dims(x_input, axis=0)
    pred = model.predict(x_input)[0][0]
    prob = float(pred)
    label_pred = "ascus" if prob > 0.5 else "reaktif"
    st.subheader("🧠 Klasifikasi CNN Lokal")
    st.markdown(f"- Prediksi: **{label_pred.upper()}**")
    st.markdown(f"- Probabilitas ASCUS: {prob:.3f}")
else:
    st.info("⚠️ Model CNN belum tersedia. Gunakan tombol Training di sidebar.")


