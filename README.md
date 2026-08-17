# 🎾 Tennis CV (Computer Vision for Tennis Analysis)

Tennis CV เป็นโปรเจกต์คอมพิวเตอร์วิทัศน์ (Computer Vision) ขั้นสูงที่ออกแบบมาเพื่อวิเคราะห์การแข่งขันเทนนิสจากวิดีโอแบบอัตโนมัติ โดยใช้สถาปัตยกรรม AI ล่าสุด (เช่น YOLO11, BoT-SORT และ Machine Learning ขั้นสูง) เพื่อติดตามผู้เล่น ลูกเทนนิส และวิเคราะห์วงสวิงแบบเฟรมต่อเฟรม

---

## 🌟 ฟีเจอร์หลัก (Features)

1. **Player Tracking (YOLO11 + BoT-SORT)**
   - ตรวจจับและติดตามนักเทนนิส (Player 1 & Player 2) ข้ามเฟรมได้อย่างแม่นยำ 
   - ป้องกันอาการสลับตัว (ID Switch) เมื่อผู้เล่นวิ่งทับไลน์กัน

2. **Tennis Ball Detection (Custom YOLO11)**
   - ตรวจจับลูกเทนนิสขนาดจิ๋วที่เคลื่อนที่ด้วยความเร็วสูง โดยใช้โมเดล YOLO ที่ถูกเทรนมาเป็นพิเศษ

3. **Hit Detection & Keyframe Extraction (AutoGluon ML + Audio Refiner)**
   - สกัดพิกัดข้อต่อด้วย **MediaPipe Pose** เพื่อแปลงร่างคนให้กลายเป็นข้อมูลเชิงกลศาสตร์ (Biomechanics)
   - วิเคราะห์พฤติกรรมการสวิงโดยใช้ **AutoGluon Ensemble Classifier** ที่มีความแม่นยำสูง ดักจับการตีลูกได้ 100% Precision
   - ฟีเจอร์ **Audio Onset Refiner** ผสานการวิเคราะห์คลื่นเสียง (เสียงป๊อกกระทบไม้) ช่วยระบุเฟรมที่ไม้กระทบลูก (Impact) ให้แม่นยำระดับเสี้ยววินาที

4. **3D Court Calibration & Mapping**
   - คำนวณพิกัดมุมกล้อง 2D ให้กลายเป็น Top-down 3D Map
   - สร้าง Mini-map จำลองตำแหน่งผู้เล่นและจุดตกของลูก

5. **Interactive WebUI (Streamlit)**
   - หน้าจอควบคุมที่ใช้งานง่ายผ่านบราวเซอร์ รองรับการพล็อตกราฟความเร็วและวิเคราะห์สถิติ

---

## 🧠 สถาปัตยกรรม AI (AI Architecture)

ระบบประกอบด้วยโมเดล AI ที่ทำงานประสานกันอย่างลงตัว:
1. **YOLO11m/n:** ค้นหาและสร้าง Bounding Box รอบตัวบุคคลและลูกเทนนิส
2. **BoT-SORT:** อัลกอริทึม Multi-Object Tracking เพื่อรักษา ID ของผู้เล่นและลูก
3. **MediaPipe Pose:** ตีเส้นโครงกระดูก (Skeleton)
4. **AutoGluon Classifier (`hit_classifier_ag`):** สมองกล Machine Learning แบบรวมมิตรโมเดลกว่าร้อยตัว ที่ถูกเทรนและปรับแต่งความมั่นใจ (Threshold = 0.20) เพื่อใช้ตัดสินจังหวะ Hit Event จากค่าสถิติ (Features)
5. **Audio Refiner (`ffmpeg` & `librosa`):** ดึงคลื่นเสียงออกมากรอง Noise และจับจังหวะพีคเพื่อแม่นยำขั้นสุด

---

## ⚙️ การติดตั้ง (Installation)

**ข้อกำหนดเบื้องต้น (Prerequisites):**
- Python 3.9 - 3.11
- การ์ดจอ NVIDIA ที่รองรับ CUDA แนะนำ VRAM 6GB ขึ้นไป
- ติดตั้ง `ffmpeg` ลงในระบบ Windows/Mac (เพื่อใช้สำหรับประมวลผลเสียง)

**ขั้นตอนการติดตั้ง:**
1. Clone โฟลเดอร์โปรเจกต์และสร้าง Virtual Environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. ติดตั้งไลบรารีที่จำเป็น (รวมถึง PyTorch และ AutoGluon)
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision ultralytics autogluon librosa
   ```

---

## 🚀 การใช้งาน (Usage Out-of-the-Box)

โปรเจกต์นี้มาพร้อมกับ **โมเดลที่ถูกเทรนไว้สมบูรณ์แล้ว (Pre-trained)** คุณสามารถโหลดโค้ดแล้วใช้งานได้ทันทีโดยไม่ต้องเทรนใหม่!

### 1. วิเคราะห์วิดีโอผ่านหน้าเว็บ (WebUI)
เปิดหน้าควบคุมหลักเพื่อใช้งานง่ายๆ:
```bash
streamlit run webui/app.py
```
- อัปโหลดวิดีโอเทนนิสของคุณ
- เลือกโหมดที่ต้องการ เช่น "YOLO11 + BoT-SORT"
- กดรันและดูผลการวิเคราะห์ 3D พร้อมกราฟได้เลย!

### 2. รันประเมินความแม่นยำเต็มรูปแบบ (End-to-End Keyframe Benchmark)
หากต้องการทดสอบรันไปป์ไลน์ทั้งหมด (ตั้งแต่ Tracking ไปจนถึงหา Impact Frame ด้วย ML + Audio) กับวิดีโอในฐานข้อมูล ให้ใช้เครื่องมือในโฟลเดอร์ `tools/`:
```bash
python tools/run_keyframe_benchmark.py all --use-audio --force --ml-threshold 0.20
```
- ระบบจะทำนายการตีและขยับเฟรมด้วยเสียงอัตโนมัติ 
- ตรวจดูรายงานผลลัพธ์ความแม่นยำแบบละเอียดได้ที่ `docs/BENCHMARK_KEYFRAME.md`

### 3. การเทรนสอนโมเดลใหม่ (กรณีต้องการอัปเกรด AI)
ตามโครงสร้าง Handover ล่าสุด สคริปต์หลักสำหรับการเทรนถูกย้ายไปที่ `scripts/train.py`:
- รัน `python scripts/train.py --help` เพื่อดูวิธีเทรนโมเดลทั้งหมดในระบบ
- หรือใช้เครื่องมือเฉพาะทางในโฟลเดอร์ `tools/` เช่น `python tools/train_hit_classifier.py`

---

## 📂 โครงสร้างโฟลเดอร์ที่สำคัญ (Directory Structure)
- `webui/` : ไฟล์ User Interface (app.py) สำหรับการแสดงผลหน้าเว็บ
- `loeuf_cv/` : แกนหลักของ Computer Vision (Court Calibration, Hit Detection, Pose, Audio)
- `checkpoints/` : **[สำคัญ]** โฟลเดอร์เก็บน้ำหนักโมเดล (เช่น AutoGluon, YOLO) ที่เทรนเสร็จสมบูรณ์แล้ว
- `scripts/` : สคริปต์หลักตามโครงสร้าง Handover (train, predict, eval, preprocess)
- `tools/` : เครื่องมือสำหรับนักพัฒนาและการทดสอบ (Benchmark, Training Scripts)
- `_archive/` : ที่เก็บโค้ดเก่าและชุดข้อมูลที่ไม่ได้ใช้แล้ว
