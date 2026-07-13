import numpy as np

def extract_keyframes(impact_frame: int, wrist_path: np.ndarray, fps: float, max_frames: int):
    """
    วิเคราะห์หา 5 Keyframes หลักจาก trajectory ของข้อมือ
    wrist_path: numpy array (N, 2) พิกัด x, y ของข้อมือข้างที่ถือไม้ (dominant_wrist)
    """
    keyframes = {
        "unit_turn": None,
        "backswing_peak": None,
        "impact": impact_frame,
        "follow_through_peak": None,
        "recovery_position": None
    }
    
    if wrist_path is None or len(wrist_path) == 0:
        return keyframes
        
    N = len(wrist_path)
    
    # 1. Backswing Peak (B2): 
    # หาจุดที่ข้อมือเหวี่ยงไปด้านหลังสุด (มี X velocity เป็น 0 เปลี่ยนทิศ) ในช่วง 1.5 วิ ก่อน impact
    window_start = max(0, impact_frame - int(1.5 * fps))
    if impact_frame > window_start:
        path_before = wrist_path[window_start:impact_frame, 0]
        if not np.isnan(path_before).all():
            impact_x = wrist_path[impact_frame, 0] if not np.isnan(wrist_path[impact_frame, 0]) else 0
            distances = np.abs(path_before - impact_x)
            if not np.isnan(distances).all():
                peak_idx_local = np.nanargmax(distances)
                keyframes["backswing_peak"] = window_start + int(peak_idx_local)
                
    # 2. Unit Turn (B1):
    # เกิดก่อน backswing peak ประมาณ 0.5-1 วิ
    if keyframes["backswing_peak"] is not None:
        bp = keyframes["backswing_peak"]
        ut_search_start = max(0, bp - int(1.0 * fps))
        if bp > ut_search_start:
            keyframes["unit_turn"] = int(np.mean([ut_search_start, bp])) # Approximation

    # 3. Follow Through Peak (B4):
    # จุดสูงสุดของการเหวี่ยงข้อมือหลัง impact ในช่วง 1.0 วิ
    window_end = min(N, impact_frame + int(1.0 * fps))
    if window_end > impact_frame:
        path_after_y = wrist_path[impact_frame:window_end, 1]
        if not np.isnan(path_after_y).all():
            # Y น้อยสุด = จุดที่อยู่สูงที่สุดบนจอ (จอภาพ 0 อยู่บนสุด)
            peak_idx_local = np.nanargmin(path_after_y)
            keyframes["follow_through_peak"] = impact_frame + int(peak_idx_local)
            
    # 4. Recovery Position (B5):
    # จุดที่ผู้เล่นกลับมาหยุดนิ่ง หลัง follow_through
    if keyframes["follow_through_peak"] is not None:
        ftp = keyframes["follow_through_peak"]
        rec_end = min(N, ftp + int(2.5 * fps))
        if rec_end > ftp:
            vel_x = np.diff(wrist_path[ftp:rec_end, 0])
            vel_y = np.diff(wrist_path[ftp:rec_end, 1])
            vel_mag = np.sqrt(vel_x**2 + vel_y**2)
            for i, v in enumerate(vel_mag):
                if v < 0.5: # ความเร็วน้อยมาก
                    keyframes["recovery_position"] = ftp + i
                    break
            if keyframes["recovery_position"] is None:
                keyframes["recovery_position"] = rec_end - 1
                
    return keyframes
