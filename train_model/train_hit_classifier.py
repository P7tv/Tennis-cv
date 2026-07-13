import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main():
    csv_file = "hit_candidates.csv"
    model_file = "hit_classifier.pkl"
    
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
    feature_cols = ["wrist_speed", "ball_dist", "ball_vel_before", "ball_vel_after", "ball_vel_change", "ball_angle_change", "racket_dist"]
    
    X = df[feature_cols].fillna(0)
    y = df["is_hit"].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nกำลังเทรนโมเดล Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"ความแม่นยำ (Accuracy): {acc*100:.2f}%")
    print("\nรายงานผลการทดสอบ:")
    print(classification_report(y_test, y_pred))
    
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
