"""เทรน stroke type classifier (FH/BH/SV/...) จาก dataset/training/stroke_features.csv
(สร้างโดย build_training_data.py) — เลียนแบบ style/pattern เดียวกับ train_hit_classifier.py

รัน: python train_model/train_stroke_classifier.py
"""

import argparse
import os
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Windows console บาง terminal default เป็น cp1252 ไม่รองรับ emoji/ตัวอักษรไทยบางตัว
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEATURE_COLS = [
    "body_shoulder_rotation_at_impact_deg",
    "body_hip_rotation_at_impact_deg",
    "body_shoulder_hip_separation_at_impact_deg",
    "arm_follow_through_angle_deg",
    "arm_backswing_depth_deg",
    "contact_height_cm",
    "contact_distance_from_body_cm",
    "wrist_minus_head_y",
    "wrist_minus_spine_x_dominant_relative",
]

# stroke type ที่มีตัวอย่างเทรนน้อยกว่านี้ -> classify_stroke() จะไม่ให้ ML มีสิทธิ์
# override rule-based เวลา rule-based เองก็ทายว่าเป็น class นั้น (ดู
# loeuf_cv/schema_builder/classifier.py) — ลองแล้วพบว่า confidence threshold
# อย่างเดียวไม่พอกันเคสนี้: บน held-out test ของ set2 ตัวอย่าง BH จริง (dataset
# มีแค่ 3 ตัวอย่าง) ถูก ML ทายเป็น SV ด้วย confidence 0.63 (ผ่าน threshold ปกติ
# สบายๆ) ทั้งที่ rule-based (geometry ล้วน ไม่พึ่งข้อมูลเทรน) ตอบถูกว่า BH —
# โมเดลไม่มีทางรู้ตัวว่า "ไม่มั่นใจ" เพราะไม่เคยเห็น BH พอจะเรียนรู้ decision
# boundary เลย ทางแก้ที่ตรงกว่าคือกันทั้ง class ไว้ล่วงหน้าจากจำนวนตัวอย่างจริง
MIN_CLASS_SAMPLES = 5


def main():
    parser = argparse.ArgumentParser(description="เทรน stroke type classifier จาก stroke_features.csv")
    parser.add_argument("--csv", default="dataset/training/stroke_features.csv")
    parser.add_argument("--model-out", default="stroke_classifier.pkl")
    args = parser.parse_args()

    csv_file = args.csv
    model_file = args.model_out

    if not os.path.exists(csv_file):
        print(f"❌ ไม่พบไฟล์ {csv_file}")
        print("กรุณารัน train_model/build_training_data.py ก่อนเพื่อสร้างไฟล์นี้")
        return

    df = pd.read_csv(csv_file)

    if "stroke_type" not in df.columns or df["stroke_type"].isnull().all():
        print("❌ ไม่มีคอลัมน์ 'stroke_type' หรือไม่มีข้อมูล")
        return

    df = df.dropna(subset=["stroke_type"])
    print(f"✅ พบข้อมูล stroke จำนวน {len(df)} แถว")

    type_counts = df["stroke_type"].value_counts()
    print("\nจำนวนต่อ stroke type:")
    for stype, count in type_counts.items():
        flag = " ⚠️ ตัวอย่างน้อยมาก" if count < 5 else ""
        print(f"- {stype}: {count}{flag}")

    if len(df) < 30:
        print("\n⚠️ ข้อมูลรวมน้อยกว่า 30 ตัวอย่าง — โมเดลนี้เหมาะเป็น prototype เริ่มต้นเท่านั้น "
              "ยังไม่ควรใช้แทน rule-based classifier เต็มรูปแบบ")
    if len(type_counts) < 3:
        print("⚠️ มี stroke type น้อยกว่า 3 แบบในข้อมูล — โมเดลจะทำนายได้แค่ type ที่เจอในเทรนเท่านั้น")

    X = df[FEATURE_COLS].fillna(0)
    y = df["stroke_type"]

    can_stratify = type_counts.min() >= 2 and len(df) >= 10
    split_kwargs = {"test_size": 0.2, "random_state": 42}
    if can_stratify:
        split_kwargs["stratify"] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    print("\nกำลังเทรนโมเดล Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"ความแม่นยำ (Accuracy บน test set เล็กมาก ตีความอย่างระวัง): {acc*100:.2f}%")
    print("\nรายงานผลการทดสอบ:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nความสำคัญของฟีเจอร์ (ยิ่งเยอะยิ่งดี):")
    importances = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True):
        print(f"- {name}: {imp*100:.1f}%")

    ml_supported_classes = set(type_counts[type_counts >= MIN_CLASS_SAMPLES].index)
    rule_only_classes = set(type_counts.index) - ml_supported_classes
    if rule_only_classes:
        print(f"\nℹ️ class ที่ตัวอย่างน้อยกว่า {MIN_CLASS_SAMPLES}: {sorted(rule_only_classes)} "
              "— classify_stroke() จะให้ rule-based ตัดสินเสมอสำหรับ class เหล่านี้ ไม่ใช้ ML")

    with open(model_file, "wb") as f:
        pickle.dump((clf, FEATURE_COLS, ml_supported_classes), f)

    print(f"\n✅ บันทึกโมเดลเสร็จสิ้น: {model_file}")
    print("classify_stroke() จะโหลดไฟล์นี้อัตโนมัติถ้ามี (ดู loeuf_cv/schema_builder/classifier.py)")


if __name__ == "__main__":
    main()
