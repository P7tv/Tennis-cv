# 🎾 Tennis CV (Computer Vision for Tennis Analysis)

Tennis CV เป็นโปรเจกต์คอมพิวเตอร์วิทัศน์ (Computer Vision) ขั้นสูงที่ออกแบบมาเพื่อวิเคราะห์การแข่งขันเทนนิสจากวิดีโอแบบอัตโนมัติ โดยใช้สถาปัตยกรรม AI ล่าสุด (เช่น YOLO26, BoT-SORT และ Machine Learning) เพื่อติดตามผู้เล่น ลูกเทนนิส และวิเคราะห์วงสวิงแบบเฟรมต่อเฟรม

---

## 🌟 ฟีเจอร์หลัก (Features)

1. **Player Tracking (YOLO26s + BoT-SORT)**
   - ตรวจจับและติดตามนักเทนนิส (Player 1 & Player 2) ข้ามเฟรมได้อย่างแม่นยำ 
   - ป้องกันอาการสลับตัว (ID Switch) เมื่อผู้เล่นวิ่งทับไลน์กัน

2. **Tennis Ball Detection (Custom YOLO26s)**
   - ตรวจจับลูกเทนนิสขนาดจิ๋วที่เคลื่อนที่ด้วยความเร็วสูง โดยใช้โมเดล YOLO26s ที่ถูกเทรนมาเป็นพิเศษ

3. **Hit Detection & ML Classifier (Action Recognition)**
   - ระบบจับจังหวะการตีลูกผสมผสานระหว่างฟิสิกส์ (ความเร็ว/ทิศทาง) และ **Machine Learning (Random Forest)**
   - สกัดพิกัดข้อศอกและข้อมือด้วย **MediaPipe Pose** เพื่อคำนวณความเร็วในการสวิง
   - แยกแยะระหว่าง "การตีโดนลูกจริง" และ "การแกว่งแขนลม/วิ่ง" ได้อย่างแม่นยำ

4. **3D Court Calibration & Mapping**
   - คำนวณพิกัดมุมกล้อง 2D ให้กลายเป็น Top-down 3D Map
   - สร้าง Mini-map จำลองตำแหน่งผู้เล่นและจุดตกของลูก

5. **Interactive WebUI (Streamlit)**
   - หน้าจอควบคุมที่ใช้งานง่ายผ่านบราวเซอร์ รองรับการพล็อตกราฟความเร็วและวิเคราะห์สถิติ

---

## 🧠 สถาปัตยกรรม AI (AI Architecture)

ระบบประกอบด้วยโมเดล AI 5 ตัวที่ทำงานประสานกัน:
1. **YOLO26s (Pre-trained):** ค้นหาและสร้าง Bounding Box รอบตัวบุคคล
2. **YOLO26s (Custom):** ค้นหาและสร้าง Bounding Box รอบลูกเทนนิส
3. **BoT-SORT:** อัลกอริทึม Multi-Object Tracking เพื่อรักษา ID ของผู้เล่นและลูก
4. **MediaPipe Pose:** ตีเส้นโครงกระดูก (Skeleton) เพื่อหาความเร็วข้อมือ
5. **Hit Classifier (Random Forest):** ML ตัดสินจังหวะ Hit Event จากค่าสถิติ (Features)

---

## ⚙️ การติดตั้ง (Installation)

**ข้อกำหนดเบื้องต้น (Prerequisites):**
- Python 3.9 - 3.11
- การ์ดจอ NVIDIA ที่รองรับ CUDA (เช่น GTX 1660 หรือสูงกว่า) แนะนำ VRAM 6GB ขึ้นไป

1. **Clone repository และสร้าง Virtual Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **ติดตั้งไลบรารีที่จำเป็น**
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision ultralytics
   ```

---

## 🚀 การใช้งาน (Usage)

### 1. เปิดหน้าเว็บ (WebUI)
พิมพ์คำสั่งด้านล่างเพื่อเปิดหน้าควบคุมหลัก:
```bash
streamlit run webui/app.py
```
เมื่อหน้าเว็บเปิดขึ้น:
- อัปโหลดวิดีโอเทนนิส
- เลือกโหมด **"YOLO11 + BoT-SORT"**
- กดปุ่ม **Run YOLO11 + BoT-SORT** เพื่อเริ่มการวิเคราะห์

### 2. การเทรนโมเดลลูกเทนนิส (Custom Ball Detection)
หากต้องการให้ AI จับลูกเทนนิสได้แม่นยำขึ้นในสภาพแวดล้อมใหม่ๆ สามารถเทรน YOLO26s ได้เอง:
```bash
python train_model/train_yolo.py
```
*(เมื่อเทรนเสร็จ ไฟล์โมเดลใหม่จะไปอยู่ที่โฟลเดอร์ `runs/detect/` อัตโนมัติ)*

### 3. การเทรนโมเดลจับจังหวะตี (Hit Classifier)
ระบบ Hit Detection สามารถฉลาดขึ้นได้ผ่านกระบวนการ Machine Learning:
1. กดวิเคราะห์วิดีโอในหน้าเว็บ (ระบบจะสร้างไฟล์ `hit_candidates.csv`)
2. เปิดไฟล์ CSV และใส่ Label ในคอลัมน์ `is_hit` (`1` = ตีจริง, `0` = ตีลม/มั่ว)
3. รันคำสั่งเพื่อสอน AI:
   ```bash
   python train_model/train_hit_classifier.py
   ```
4. ระบบจะบันทึกสมอง AI ลงในไฟล์ `hit_classifier.pkl` และเว็บแอปจะนำไปใช้โดยอัตโนมัติในการรันครั้งต่อไป

---

## 📂 โครงสร้างโฟลเดอร์ (Directory Structure)
- `webui/` : ไฟล์ User Interface (app.py) และระบบติดตามด้วย YOLO
- `loeuf_cv/` : แกนหลักของ Computer Vision (Court Calibration, Hit Detection, Pose)
- `train_model/` : สคริปต์สำหรับดาวน์โหลด Dataset และเทรนโมเดล (YOLO & ML)
- `runs/` : โฟลเดอร์เก็บโมเดลที่ถูกเทรนเสร็จแล้ว
