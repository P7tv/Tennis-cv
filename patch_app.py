p = r"d:\Work\Tennis-cv\webui\app.py"
txt = open(p, encoding="utf-8").read()
txt = txt.replace(
    '["Classical CV (MOG2)", "SAM 2 Click-to-Track"]',
    '["Classical CV (MOG2)", "SAM 2 Click-to-Track", "YOLO11 + BoT-SORT"]'
)
txt = txt.replace(
    '    else:\n        run_btn = False\n',
    '    elif tracking_method == "YOLO11 + BoT-SORT":\n        run_btn = st.button("\u25b6\ufe0f Run YOLO11 + BoT-SORT", type="primary", disabled=uploaded is None)\n    else:\n        run_btn = False\n'
)
open(p, "w", encoding="utf-8").write(txt)
print("OK")
