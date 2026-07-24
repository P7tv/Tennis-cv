import argparse
import sys
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Windows console บาง terminal default เป็น cp1252 ไม่รองรับ emoji/ตัวอักษรไทยบางตัว
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="เทรน hit classifier จาก hit_candidates.csv ที่ label แล้ว")
    parser.add_argument("--csv", default="hit_candidates.csv", help="path ของ CSV ที่มีคอลัมน์ is_hit label แล้ว")
    parser.add_argument("--model-out", default="hit_classifier.pkl", help="path ที่จะบันทึกโมเดล")
    args = parser.parse_args()

    csv_file = args.csv
    model_file = args.model_out

    if not os.path.exists(csv_file):
        print(f"❌ ไม่พบไฟล์ {csv_file}")
        print("กรุณารันโปรแกรมและกด 'วิเคราะห์ Hit Events' เพื่อสร้างไฟล์เก็บข้อมูลก่อนครับ")
        return
        
    df = pd.read_csv(csv_file)
    
    # Check if there's any labeled data
    if "is_hit" not in df.columns or df["is_hit"].isnull().all():
        print(f"❌ ยังไม่ได้ใส่ Label (0 หรือ 1) ในคอลัมน์ 'is_hit'")
        print("กรุณาเปิดไฟล์ CSV แล้วใส่เลขบอก AI ว่าอันไหนตีโดนจริง (1) หรือตีมั่ว (0) ก่อนครับ")
        return
        
    # Drop rows without labels
    df = df.dropna(subset=["is_hit"])
    print(f"✅ พบข้อมูลที่ Label แล้วจำนวน {len(df)} แถว")
    
    if len(df) < 10:
        print("⚠️ ข้อมูลน้อยเกินไป แนะนำให้ใส่ Label อย่างน้อย 30-50 จังหวะครับ")
    
    # Feature columns
    feature_cols = [
        "wrist_speed", "ball_dist", "ball_vel_before", "ball_vel_after", "ball_vel_change",
        "ball_angle_change", "racket_dist",
        "speed_pre_mean", "speed_post_mean", "speed_decel_ratio",
        "speed_peak_sharpness", "speed_std_window",
    ]
    
    X = df[feature_cols].fillna(0)
    y = df["is_hit"].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nกำลังเทรนโมเดล Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"ความแม่นยำ (Accuracy): {acc*100:.2f}%")
    print("\nรายงานผลการทดสอบ (random row split — เสี่ยง data leakage ถ้า candidate จาก "
          "คลิปเดียวกันหลุดไปอยู่ทั้ง train และ test ดู grouped CV ด้านล่างสำหรับตัวเลขที่เชื่อถือได้กว่า):")
    print(classification_report(y_test, y_pred))

    # ─── Grouped (Leave-One-Clip-Out) cross-validation — เหตุผลเดียวกับ
    # train_stroke_classifier.py: held-out ทีละคลิปเต็มๆ กัน leakage ข้าม candidate
    # ในคลิปเดียวกัน (โมเดลอาจจำ noise/สภาพแสงของคลิปแทนที่จะเรียนรู้ว่า "ตี" หน้าตายังไง)
    if "clip_id" in df.columns and df["clip_id"].nunique() > 1:
        from sklearn.model_selection import LeaveOneGroupOut
        groups = df["clip_id"]
        logo = LeaveOneGroupOut()
        oof_true, oof_pred = [], []
        for train_idx, test_idx in logo.split(X, y, groups):
            clf_fold = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42,
                                               class_weight="balanced")
            clf_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_pred.extend(clf_fold.predict(X.iloc[test_idx]))
            oof_true.extend(y.iloc[test_idx])
        logo_acc = accuracy_score(oof_true, oof_pred)
        print(f"\n===== Grouped (Leave-One-Clip-Out) cross-validation — {groups.nunique()} คลิป =====")
        print(f"ความแม่นยำ (held-out ทีละคลิปเต็มๆ, ไม่มี leakage): {logo_acc*100:.2f}%")
        print(classification_report(oof_true, oof_pred, zero_division=0))
    else:
        print("\n⚠️ ไม่มีคอลัมน์ clip_id หรือมีคลิปเดียว — ข้าม grouped cross-validation")

    # Feature Importance
    print("\nความสำคัญของฟีเจอร์ (ยิ่งเยอะยิ่งดี):")
    importances = clf.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"- {name}: {imp*100:.1f}%")
        
    # Save model
    with open(model_file, "wb") as f:
        pickle.dump((clf, feature_cols), f)
        
    print(f"\n✅ บันทึกโมเดลเสร็จสิ้น: {model_file}")
    print("ตอนนี้ระบบ Hit Detection พร้อมใช้ AI ช่วยตัดสินใจแล้วครับ!")

if __name__ == "__main__":
    main()
