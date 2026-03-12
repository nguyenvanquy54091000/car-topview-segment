import cv2
from ultralytics import YOLO
import tempfile
import os
import imageio

model = YOLO(r'D:\Project\segment car with yolo\train\train\weights\best.pt')

def predict_image(img):
    """
    Hàm xử lý dự đoán đối với Ảnh.
    """
    if img is None:
        return None, "0"
    results = model(img,conf=0.5)
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    num_objects = len(results[0].boxes) if results[0].boxes is not None else 0
    return annotated_img, f"Số lượng: {num_objects}"


def predict_video(video_path):
    """
    Hàm xử lý dự đoán đối với Video.
    """
    if not video_path:
        return None, "0"
        
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    stride = 4

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "output_video.mp4")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=None)
    
    frame_counts_info = []
    annotated_frame = None
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % stride == 0:
            results = model.track(frame, persist=True, tracker='botsort.yaml', conf=0.5, retina_masks=True, iou=0.5) 
        
            current_count = len(results[0].boxes) if (results[0].boxes is not None) else 0
            frame_counts_info.append(f"Frame {frame_count}: {current_count} đối tượng")
                
            annotated_frame = results[0].plot(boxes=True, labels=True)
            
            # Giữa mỗi lần kiểm tra, ta có thể yield thông báo trên Textbox
            # Do Gradio Video Player không thể phát 1 file MP4 đang bị khóa (chưa đóng bằng writer.close()), 
            # Nên luồng Video ở giao diện vẫn phải là None trong quá trình làm, đến cuối cùng file mới hiện lên.
            yield None, "\n".join(frame_counts_info[-10:]) # Chỉ hiển thị 10 logs gần nhất cho đỡ rối
        
        # Để ghi vào imageio, hệ màu cần là RGB thay vì BGR mặc định của OpenCV
        if annotated_frame is not None:
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
            
        frame_count += 1
        
    cap.release()
    writer.close()
    
    # Ở lần yield cuối cùng, ta mới trả vể file video đã hoàn thiện cùng với toàn bộ nhật ký
    yield output_path, "\n".join(frame_counts_info)