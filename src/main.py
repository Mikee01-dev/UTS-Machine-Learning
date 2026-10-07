import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    load_data,
    ganti_tanda_tanya,
    cek_missing_value,
    handle_missing_value,
    cek_duplikasi,
    cek_outlier,
    handle_outlier,
    encode_kategorikal,
    pisahkan_fitur_target
)
from transform import minmax_scaling
from resampling import adasyn_tomek
from feature_selection import select_from_model, rfecv_selection
from model import train_decision_tree
from evaluation import evaluasi_model
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'adult.csv'
OUTPUT_DIR = BASE_DIR / 'output'
TARGET_COL = 'income'
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    print("=" * 60)
    print("1. INPUT DATA")
    print("=" * 60)
    df = load_data(str(DATA_PATH))
    df = ganti_tanda_tanya(df)

    print("\n" + "=" * 60)
    print("2. PREPROCESSING")
    print("=" * 60)

    df = cek_missing_value(df)
    df = handle_missing_value(df)

    df = cek_duplikasi(df)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    df = cek_outlier(df, numeric_cols)
    df = handle_outlier(df, numeric_cols)

    df = encode_kategorikal(df)

    X, y = pisahkan_fitur_target(df, TARGET_COL)

    print("\n" + "=" * 60)
    print("3. SPLIT DATA (80% training, 20% testing)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\n" + "=" * 60)
    print("4. TRANSFORMASI MIN-MAX")
    print("=" * 60)
    X_train_scaled, X_test_scaled, scaler = minmax_scaling(X_train, X_test)

    print("\n" + "=" * 60)
    print("5. RESAMPLING (ADASYN + TOMEK LINKS)")
    print("=" * 60)
    X_train_res, y_train_res = adasyn_tomek(X_train_scaled, y_train)

    print("\n" + "=" * 60)
    print("6. SELEKSI FITUR")
    print("=" * 60)

    X_train_sfm, X_test_sfm, sfm = select_from_model(
        X_train_res, y_train_res, X_test_scaled
    )

    X_train_rfecv, X_test_rfecv, rfecv = rfecv_selection(
        X_train_res, y_train_res, X_test_scaled
    )

    print("\n" + "=" * 60)
    print("7. KLASIFIKASI DECISION TREE")
    print("=" * 60)

    print("\n--- Model dengan SelectFromModel ---")
    model_sfm = train_decision_tree(X_train_sfm, y_train_res)

    print("\n--- Model dengan RFECV ---")
    model_rfecv = train_decision_tree(X_train_rfecv, y_train_res)

    print("\n" + "=" * 60)
    print("8. EVALUASI")
    print("=" * 60)

    print("\n### Evaluasi Model SelectFromModel ###")
    hasil_sfm = evaluasi_model(
        model_sfm, X_test_sfm, y_test,
        output_dir=str(OUTPUT_DIR),
        nama_model='SelectFromModel'
    )

    print("\n### Evaluasi Model RFECV ###")
    hasil_rfecv = evaluasi_model(
        model_rfecv, X_test_rfecv, y_test,
        output_dir=str(OUTPUT_DIR),
        nama_model='RFECV'
    )

    print("\n" + "=" * 60)
    print("9. RINGKASAN HASIL")
    print("=" * 60)

    print("\nSelectFromModel:")
    print(f"  Accuracy : {hasil_sfm['accuracy']:.4f}")
    print(f"  Precision: {hasil_sfm['precision']:.4f}")
    print(f"  Recall   : {hasil_sfm['recall']:.4f}")
    print(f"  F1-Score : {hasil_sfm['f1_score']:.4f}")

    print("\nRFECV:")
    print(f"  Accuracy : {hasil_rfecv['accuracy']:.4f}")
    print(f"  Precision: {hasil_rfecv['precision']:.4f}")
    print(f"  Recall   : {hasil_rfecv['recall']:.4f}")
    print(f"  F1-Score : {hasil_rfecv['f1_score']:.4f}")

    print("\n" + "=" * 60)
    print("10. KESIMPULAN")
    print("=" * 60)

    if hasil_sfm['f1_score'] > hasil_rfecv['f1_score']:
        print("Model terbaik: SelectFromModel")
        print(f"  F1-Score: {hasil_sfm['f1_score']:.4f}")
    else:
        print("Model terbaik: RFECV")
        print(f"  F1-Score: {hasil_rfecv['f1_score']:.4f}")

    print("\nOutput tersimpan di:", OUTPUT_DIR)
    print("Pipeline selesai.")


if __name__ == '__main__':
    main()