import gradio as gr
import tempfile
import os
from model import *
from fastapi import FastAPI
import uvicorn


with gr.Blocks(title="YOLOv26 Nhận Diện") as app:
    gr.Markdown("# 🚀 Demo Nhận Diện Đối Tượng Bằng YOLO")
    gr.Markdown("Hãy tải lên hình ảnh hoặc video để xem kết quả từ mô hình được train của bạn.")
    
    with gr.Tabs():
        # --- TAB XỬ LÝ ẢNH ---
        with gr.TabItem("🖼️ Xử lý Ảnh"):
            with gr.Row():
                img_input = gr.Image(type="numpy", label="Tải ảnh đầu vào")
                with gr.Column():
                    img_output = gr.Image(type="numpy", label="Ảnh Kết quả")
                    img_count = gr.Textbox(label="Số lượng đối tượng (Ảnh)")
            
            img_btn = gr.Button("Bắt đầu Dự đoán Ảnh", variant="primary")
            img_btn.click(fn=predict_image, inputs=img_input, outputs=[img_output, img_count])

        # --- TAB XỬ LÝ VIDEO ---
        with gr.TabItem("🎥 Xử lý Video"):
            with gr.Row():
                vid_input = gr.Video(label="Tải video đầu vào")
                with gr.Column():
                    vid_output = gr.Video(label="Video Kết quả")
                    vid_count = gr.Textbox(label="Số lượng đối tượng từng frame", lines=10)
                
            vid_btn = gr.Button("Bắt đầu Dự đoán Video", variant="primary")
            vid_btn.click(fn=predict_video, inputs=vid_input, outputs=[vid_output, vid_count], queue=True)


# 3. Mount Gradio vào FastAPI để dùng Uvicorn
fastapi_app = FastAPI()
fastapi_app = gr.mount_gradio_app(fastapi_app, app, path="/")

if __name__ == "__main__":
    uvicorn.run("app:fastapi_app", host="127.0.0.1", port=7860, reload=True)    
