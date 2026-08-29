import base64
import os
import webbrowser
from collections import Counter
import cv2
from ultralytics import YOLO

# 1. Bread Price Dictionary
pricebook = {
    "croissant": 55,
    "horn": 35,
    "pipeapple": 32,
    "porkfloss": 49,
    "toast": 36,
}

# 2. Load YOLO Model
model = YOLO("best.pt")


# 3. USB Camera Initialization Helper
def initialize_camera():
    # Try DirectShow on Index 0, 1, and 2
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        # Apply robust USB camera stream settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))

        if cap.isOpened():
            # Warm up stream and test frame grabbing
            for _ in range(5):
                ret, frame = cap.read()

            if ret and frame is not None:
                print(
                    f"✅ 成功啟用 USB 攝影機 (索引點: {index}, 後端: DirectShow)"
                )
                return cap
            cap.release()

    # Fallback: Try MSMF if DirectShow fails across all indices
    for index in [0, 1]:
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(
                    f"✅ 成功啟用 USB 攝影機 (索引點: {index}, 後端: MSMF)"
                )
                return cap
            cap.release()

    return None


# Launch Camera
cap = initialize_camera()

if cap is None:
    print("\n❌ 無法啟動攝影機！")
    print("請確認：")
    print(
        "1. Windows 設定 -> 隱私權與安全性 -> 相機 -> 已開啟「讓桌面應用程式存取您的相機」"
    )
    print("2. 沒有其他程式（如 Zoom, Teams, Windows 相機 App）正在占用鏡頭")
    print("3. USB 鏡頭已插緊（嘗試換到電腦後方獨立的 USB 3.0 插槽）\n")
    exit()

print("請按下 's' 拍照進行辨識，或按 'q' 離開")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("無法讀取畫面，嘗試重新獲取...")
            continue

        # Display Live Preview
        cv2.imshow('Camera - Press "s" to Scan, "q" to Quit', frame)

        # Cross-platform safe key detection
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            print("\n照片已拍下，開始辨識...")

            # Run YOLO model
            results = model(frame)[0]
            names = model.names
            detected_classes = [names[int(cls)] for cls in results.boxes.cls]

            # Calculate item count & price totals
            counts = Counter(detected_classes)
            total_price = sum(
                pricebook.get(bread, 0) * count
                for bread, count in counts.items()
            )

            print("--- 辨識結果 ---")
            for bread, count in counts.items():
                item_price = pricebook.get(bread, 0) * count
                print(f"{bread}: {count} 個 (小計: {item_price} 元)")

            print(f"總數量：{sum(counts.values())} 個")
            print(f"總價格：{total_price} 元\n")

            # Plot annotated detections
            annotated = results.plot()
            cv2.imshow("Detection Result - Press ANY KEY to continue", annotated)

            # Encode directly from memory (avoids OneDrive/disk locks)
            _, buffer = cv2.imencode(".jpg", annotated)
            base64_img = base64.b64encode(buffer).decode()

            # Generate HTML Report
            items_html = "".join(
                f"<li>{bread}: {count} 個</li>"
                for bread, count in counts.items()
            )
            html_content = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <title>辨識結果</title>
            </head>
            <body style="font-family: Arial, sans-serif; margin: 20px;">
                <h1>🍞 麵包辨識結果</h1>
                <img src="data:image/jpeg;base64,{base64_img}" style="max-width:600px; border-radius:8px;"><br>
                <h3>明細：</h3>
                <ul>{items_html}</ul>
                <p><strong>總數量：</strong>{sum(counts.values())} 個</p>
                <p><strong>總價格：</strong>{total_price} 元</p>
            </body>
            </html>
            """

            # Save and open web page report
            html_path = os.path.realpath("result.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            webbrowser.open("file://" + html_path)

            # Freeze result image until key press, then return to camera stream
            cv2.waitKey(0)
            cv2.destroyWindow("Detection Result - Press ANY KEY to continue")

        elif key == ord("q"):
            break

finally:
    # Safely release camera hardware on crash or close
    cap.release()
    cv2.destroyAllWindows()
    print("攝影機已順利關閉。")