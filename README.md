# UTS Praktikum Machine Learning — Kelas C2 Kelompok 4

**Mata Kuliah:** Prak Pembelajaran Mesin (1 sks)  
**Kelas:** C2  
**Kelompok:** 4  
**Kode Soal:** D  
**Dosen Pengampu:**
- Dr. Indah Werdiningsih, S.Si., M.Kom
- Barry Nuqoba, S.Si., M.Kom., Ph.D
- Purbandini, S.Si., M.Kom

**Anggota Kelompok:**

| NIM | Nama |
|---|---|
| 4342410... | Sovia Aribi Damayanti |
| 4342410... | Kayla Dicta Pramudya |
| 4342410... | Michael Putra Pratama Otemusu |

---

## Deskripsi

Project ini adalah implementasi **UTS Praktikum Machine Learning** untuk **Soal D**, dengan alur sesuai flowchart:

```
Input
  ↓
Preprocessing
  ↓
Split Data
  ↓
┌───────────────┬───────────────┐
│ Data Testing  │ Data Training │
│      ↓        │       ↓       │
│ Transformation│ Transformation│
│      ↓        │       ↓       │
│    Fitur      │ Ekstraksi Fitur│
│               │       ↓       │
│               │    Fitur      │
└───────────────┴───────────────┘
                  ↓
               Training
                  ↓
               Testing
                  ↓
               Evaluasi
```

---

## Dataset

| Item | Keterangan |
|---|---|
| **Nama** | Adult Census Income |
| **Sumber** | [Kaggle — UCI ML](https://www.kaggle.com/datasets/uciml/adult-census-income) |
| **Jumlah Baris** | 32.561 |
| **Jumlah Kolom** | 15 |
| **Target** | `income` (`<=50K` / `>50K`) |
| **Tipe Fitur** | Kategorikal + Numerik |
| **Karakteristik** | Imbalanced (mayoritas `<=50K`) |

**Deskripsi:**  
Dataset ini digunakan untuk memprediksi apakah seseorang memiliki penghasilan **lebih dari $50K/tahun** berdasarkan data sensus. Fitur mencakup usia, pekerjaan, pendidikan, status pernikahan, jam kerja, dll.

---

## Metode

| Tahap | Metode | Keterangan |
|---|---|---|
| **Preprocessing** | Missing value, duplikasi, outlier, encoding | Deteksi & handling |
| **Transformasi** | Min-Max Scaler | Rentang 0–1 |
| **Split Data** | 80% training, 20% testing | Stratified |
| **Resampling** | ADASYN + Tomek Links | Hanya data training |
| **Seleksi Fitur 1** | SelectFromModel (Decision Tree) | Embedded method |
| **Seleksi Fitur 2** | RFECV (Recursive Feature Elimination with CV) | Wrapper method |
| **Klasifikasi** | Decision Tree | `criterion='gini'` |
| **Evaluasi** | Confusion Matrix | Akurasi, Presisi, Recall, F1-Score |

---

## Struktur Folder

```
UTS-Machine-Learning/
│
├── data/
│   └── adult.csv                  # Dataset (tidak di-push ke GitHub)
│
├── output/
│   ├── confusion_matrix_SelectFromModel.png
│   ├── confusion_matrix_RFECV.png
│   ├── classification_report_SelectFromModel.txt
│   └── classification_report_RFECV.txt
│
├── src/
│   ├── preprocessing.py           # Load, missing value, duplikasi, outlier, encoding
│   ├── transform.py               # Min-Max Scaling
│   ├── resampling.py              # ADASYN + Tomek Links
│   ├── feature_selection.py       # SelectFromModel + RFECV
│   ├── model.py                   # Decision Tree
│   ├── evaluation.py              # Confusion Matrix + metrik
│   └── main.py                    # Full pipeline
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/Mikee01-dev/UTS-Machine-Learning.git
cd UTS-Machine-Learning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download `adult.csv` dari [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income), lalu letakkan di folder `data/`.

### 4. Jalankan Program
```bash
python src/main.py
```

### 5. Lihat Hasil
- Output program tampil di terminal.
- Confusion matrix & classification report tersimpan di folder `output/`.

---

## Hasil Evaluasi

| Metrik | SelectFromModel | RFECV |
|---|---|---|
| **Accuracy** | 0.7707 | 0.7675 |
| **Precision** | 0.7756 | 0.7727 |
| **Recall** | 0.7707 | 0.7675 |
| **F1-Score** | **0.7730** | 0.7699 |
| **Jumlah Fitur** | 7 | 9 |

**Kesimpulan:**  
Model **SelectFromModel** lebih unggul tipis di semua metrik dengan **lebih sedikit fitur** (7 vs 9), sehingga lebih efisien.

---

## Confusion Matrix

### SelectFromModel
```
[[4147  793]
 [ 699  869]]
```

### RFECV
```
[[4134  806]
 [ 707  861]]
```

---

## Analisis Singkat

- **Akurasi ~77%** → model cukup baik untuk prediksi income.
- **Kelas mayoritas (`<=50K`)** → F1-Score 0.85 (bagus).
- **Kelas minoritas (`>50K`)** → F1-Score 0.54 (kurang bagus).
- **Overfitting** → pohon sangat dalam (64 & 62), perlu tuning.
- **SelectFromModel** lebih efisien (7 fitur, F1 0.7730).

---

## Requirements

```txt
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

---

## Lisensi

Project ini dibuat untuk keperluan **Ujian Tengah Semester** mata kuliah **Prak Pembelajaran Mesin**, Program Studi D4 Teknik Informatika, Fakultas Vokasi, Universitas Airlangga.

---

## Referensi

- [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html)
- [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
- [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)