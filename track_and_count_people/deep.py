from ultralytics import YOLO
import cv2

# Load model YOLOv8
model = YOLO('/Users/owwl/Downloads/object_counting/best5.pt')

# Buka video input
video_path = '/Users/owwl/Downloads/object_counting/input.mp4'
cap = cv2.VideoCapture(video_path)

# Periksa apakah video input bisa dibuka
if not cap.isOpened():
    print("Error: Gagal membuka video input. Periksa path file.")
    exit(1)

# Dapatkan properti video untuk persiapan output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Buat VideoWriter untuk menyimpan output dengan ukuran 192x108
output_path = '/Users/owwl/Downloads/object_counting/output_video5_frames.mp4'
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Alternatif codec untuk MP4

try:
    out = cv2.VideoWriter(output_path, fourcc, fps, (192, 108))  # Set output size to 192x108
    if not out.isOpened():
        raise Exception("Gagal membuka VideoWriter. Periksa path atau codec.")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Counter untuk membatasi hanya 24 frame
frame_count = 0
max_frames = 945

# Loop melalui setiap frame dalam video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari video.")
        break
        
    # Resize frame menjadi 192x108
    frame = cv2.resize(frame, (192, 108))

    # Hanya proses 24 frame pertama
    if frame_count >= max_frames:
        break

    # Lakukan detection dan tracking menggunakan model YOLOv8
    results = model.track(frame, persist=True)

    # Visualisasi hasil detection dan tracking
    annotated_frame = results[0].plot()

    # Tulis frame yang telah di-annotasi ke output video
    out.write(annotated_frame)
    print(f"Frame {frame_count + 1} diproses dan ditulis ke output.")

    # Tampilkan frame yang telah di-annotasi (opsional)
    cv2.imshow('YOLOv8 Tracking - 24 Frames', annotated_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame counter
    frame_count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video (24 frame pertama) disimpan di: {output_path}")