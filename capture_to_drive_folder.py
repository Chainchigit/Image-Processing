import os
import time
from datetime import datetime
import cv2

CAM_IDS = [0, 1]          # กล้อง USB 2 ตัว
INTERVAL_SEC = 60

DRIVE_SYNC_DIR = r"G:\My Drive\test_cam"
FALLBACK_DIR   = os.path.join(os.path.expanduser("~"), "Desktop", "leaf_captures")

WIDTH, HEIGHT = 1280, 720

def pick_output_dir():
    if os.path.isdir(DRIVE_SYNC_DIR):
        os.makedirs(DRIVE_SYNC_DIR, exist_ok=True)
        return DRIVE_SYNC_DIR
    os.makedirs(FALLBACK_DIR, exist_ok=True)
    print(f"[WARN] DRIVE_SYNC_DIR not found -> fallback to: {FALLBACK_DIR}")
    return FALLBACK_DIR

def open_cameras():
    caps = []
    for cam_id in CAM_IDS:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera id={cam_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        caps.append(cap)
    return caps

def main():
    out_dir = pick_output_dir()
    caps = open_cameras()

    # warm up
    for _ in range(10):
        for cap in caps:
            cap.read()
        time.sleep(0.02)

    print("   Running (DUAL CAM). Click window, then press:")
    print("   c = capture now | q = quit")
    print("   Auto capture every", INTERVAL_SEC, "sec")
    print("   Save to:", out_dir)

    next_shot = time.time()
    last_manual = 0.0
    MANUAL_COOLDOWN = 0.4

    while True:
        frames = []
        for cap in caps:
            ok, frame = cap.read()
            if not ok:
                frame = None
            frames.append(frame)

        if any(f is None for f in frames):
            print("[WARN] read frame failed")
            continue

        # แสดงผลแยก 2 หน้าต่าง
        for i, frame in enumerate(frames):
            hud = frame.copy()
            cv2.putText(hud, f"CAM {i+1}  [c]=Capture  [q]=Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(f"LEAF_CAM_{i+1}", hud)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), ord("Q")):
            break

        now = time.time()
        do_time = now >= next_shot

        do_manual = False
        if k in (ord("c"), ord("C")) and (now - last_manual) > MANUAL_COOLDOWN:
            do_manual = True
            last_manual = now

        if do_time or do_manual:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for i, frame in enumerate(frames):
                path = os.path.join(out_dir, f"cam{i+1}_{ts}.jpg")
                ok2 = cv2.imwrite(path, frame)
                print("SAVED" if ok2 else " SAVE FAILED", path)
            next_shot = now + INTERVAL_SEC

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
