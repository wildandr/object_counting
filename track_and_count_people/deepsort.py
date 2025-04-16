from ultralytics import YOLO
import cv2
import numpy as np
import time
from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSort

class DeepSort(_DeepSort):
    """
    Wrapper untuk DeepSort untuk kompatibilitas dan kemudahan penggunaan
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ObjectCounter:
    def __init__(self, line_position, direction='vertical'):
        """
        Inisialisasi counter objek dengan garis separator vertikal
        
        Args:
            line_position (tuple): Koordinat garis (x1, y1, x2, y2)
            direction (str): Arah penghitungan (default 'vertical')
        """
        self.line_position = line_position
        self.direction = direction
        self.counted_tracks = set()
        self.object_count = 0
        self.object_count_left = 0
        self.object_count_right = 0

    def check_line_crossing(self, track):
        """
        Cek apakah objek melewati garis separator
        
        Args:
            track: Track objek dari DeepSORT
        
        Returns:
            str: Arah objek melewati garis ('left', 'right', atau None)
        """
        # Ambil posisi terakhir objek
        ltrb = track.to_ltwh()  # Ubah ke to_ltwh() 
        x_center = ltrb[0] + ltrb[2] / 2
        y_center = ltrb[1] + ltrb[3] / 2

        # Koordinat garis
        x1, y1, x2, y2 = self.line_position

        # Logika pengecekan melewati garis vertikal
        if self.direction == 'vertical':
            # Pastikan melewati garis vertikal
            if (y1 <= y_center <= y2):
                # Cek apakah objek melewati garis
                if abs(x_center - x1) < 5:  # Toleransi 5 pixel
                    # Tentukan arah
                    if x_center < x1:
                        return 'left'
                    else:
                        return 'right'
        
        return None

    def update(self, tracks):
        """
        Update counter berdasarkan tracks yang melewati garis
        
        Args:
            tracks: Daftar tracks dari DeepSORT
        """
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            
            # Cek apakah track sudah pernah dihitung
            if track_id not in self.counted_tracks:
                crossing_direction = self.check_line_crossing(track)
                if crossing_direction == 'left':
                    self.object_count_left += 1
                    self.object_count += 1
                    self.counted_tracks.add(track_id)
                elif crossing_direction == 'right':
                    self.object_count_right += 1
                    self.object_count += 1
                    self.counted_tracks.add(track_id)

def detect_and_track(frame, model, tracker, counter):
    """
    Deteksi dan tracking objek dalam frame
    
    Args:
        frame (numpy.ndarray): Frame video
        model (YOLO): Model deteksi YOLOv8
        tracker (DeepSort): Tracker DeepSORT
        counter (ObjectCounter): Objek counter
    
    Returns:
        numpy.ndarray: Frame dengan annotasi tracking dan garis
    """
    # Deteksi objek menggunakan YOLOv8
    results = model(frame)[0]
    
    # Persiapkan deteksi untuk tracking
    detections = []
    for result in results.boxes:
        bbox = result.xyxy[0].tolist()  # [x1, y1, x2, y2]
        conf = result.conf[0]
        cls = result.cls[0]
        
        # Filter deteksi 
        if conf > 0.5:
            deep_sort_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            detections.append(([*deep_sort_bbox], conf, int(cls)))
    
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Update counter
    counter.update(tracks)
    
    # Gambar garis separator vertikal
    x1, y1, x2, y2 = counter.line_position
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    # Tambahkan counter
    cv2.putText(
        frame, 
        f"Total: {counter.object_count}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 255, 0), 
        2
    )
    # cv2.putText(
    #     frame, 
    #     f"Left: {counter.object_count_left}", 
    #     (10, 50), 
    #     cv2.FONT_HERSHEY_SIMPLEX, 
    #     0.5, 
    #     (0, 255, 0), 
    #     2
    # )
    # cv2.putText(
    #     frame, 
    #     f"Right: {counter.object_count_right}", 
    #     (10, 70), 
    #     cv2.FONT_HERSHEY_SIMPLEX, 
    #     0.5, 
    #     (0, 255, 0), 
    #     2
    # )
    
    # Gambar bounding box dan track ID
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltwh()  # Ubah ke to_ltwh()
        
        # Gambar bounding box
        cv2.rectangle(
            frame, 
            (int(ltrb[0]), int(ltrb[1])), 
            (int(ltrb[0] + ltrb[2]), int(ltrb[1] + ltrb[3])), 
            (0, 255, 0), 
            2
        )
        
        # # Tambahkan track ID
        # cv2.putText(
        #     frame, 
        #     f"ID: {track_id}", 
        #     (int(ltrb[0]), int(ltrb[1]) - 10), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     0.5, 
        #     (0, 255, 0), 
        #     2
        # )
    
    return frame

def main():
    # Inisialisasi model dan tracker
    model = YOLO('/Users/owwl/Downloads/object_counting/best5.pt')
    tracker = DeepSort(max_age=30)

    # Definisikan garis separator vertikal
    # Format: (x1, y1, x2, y2) - sesuaikan dengan resolusi video Anda
    # Untuk frame 192x108, garis di tengah secara vertikal
    line_position = (106, 0, 106, 108)  # Garis vertikal di tengah frame
    
    # Buat object counter
    counter = ObjectCounter(line_position, direction='vertical')

    # Buka video input
    video_path = '/Users/owwl/Downloads/object_counting/input.mp4'
    cap = cv2.VideoCapture(video_path)

    # Periksa apakah video input bisa dibuka
    if not cap.isOpened():
        print("Error: Gagal membuka video input. Periksa path file.")
        return

    # Dapatkan properti video untuk persiapan output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Hitung durasi video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = total_frames / original_fps
    
    # Tetapkan output fps menjadi 5 frame per detik
    output_fps = 4
    
    # Hitung berapa frame yang harus dilewati untuk mencapai 5 fps
    frames_to_skip = max(1, int(original_fps / output_fps))
    print(f"FPS asli: {original_fps}, FPS output: {output_fps}, Melewati {frames_to_skip-1} frame dari setiap {frames_to_skip} frame")

    # Buat VideoWriter untuk menyimpan output
    output_path = '/Users/owwl/Downloads/object_counting/output_video_5fps_counting.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    try:
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (192, 108))
        if not out.isOpened():
            raise Exception("Gagal membuka VideoWriter. Periksa path atau codec.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Counter untuk membatasi frame
    frame_count = 0
    max_frames = 945
    skip_frame_counter = 0

    # Mulai timer untuk mengukur waktu inferensi
    start_time = time.time()

    # Loop melalui setiap frame dalam video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Selesai memproses video.")
            break
        
        # Skip frame sesuai rasio fps
        skip_frame_counter += 1
        if skip_frame_counter % frames_to_skip != 0:
            # Lewati frame ini, lanjut ke frame berikutnya
            continue
            
        # Resize frame menjadi 192x108
        frame = cv2.resize(frame, (192, 108))

        # Hentikan jika mencapai max frame
        if frame_count >= max_frames:
            break

        # Deteksi, tracking, dan counting
        tracked_frame = detect_and_track(frame, model, tracker, counter)

        # Tulis frame yang telah di-tracking ke output video
        out.write(tracked_frame)
        print(f"Frame {frame_count + 1} diproses dan ditulis ke output. Count: {counter.object_count}")

        # Tampilkan frame (opsional)
        cv2.imshow('Object Counting with Vertical Separator Line', tracked_frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Increment frame counter
        frame_count += 1

    # Hitung total waktu inferensi
    end_time = time.time()
    inference_time = end_time - start_time

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Tampilkan informasi video dan waktu pemrosesan
    print(f"Output video (object counting) disimpan di: {output_path}")
    print(f"Total objek: {counter.object_count}")
    print(f"Objek di sisi kiri: {counter.object_count_left}")
    print(f"Objek di sisi kanan: {counter.object_count_right}")
    
    # Tampilkan informasi perbandingan waktu inferensi dengan durasi video
    print("\n=== Informasi Waktu Pemrosesan ===")
    print(f"Durasi video input: {video_duration_seconds:.2f} detik ({video_duration_seconds/60:.2f} menit)")
    print(f"Waktu inferensi: {inference_time:.2f} detik ({inference_time/60:.2f} menit)")
    print(f"Rasio waktu inferensi/durasi video: {inference_time/video_duration_seconds:.2f}x")
    print(f"Kecepatan pemrosesan: {video_duration_seconds/inference_time:.2f}x kecepatan real-time")

if __name__ == "__main__":
    main()