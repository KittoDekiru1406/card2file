import streamlit as st
import requests
from PIL import Image
import io
import os
from reportlab.pdfgen import canvas
from streamlit_webrtc import webrtc_streamer
import cv2

CARD2FILE_URL = os.getenv("CARD2FILE_URL", "http://localhost:8000/api/ocr")

def main():
    # Thiết lập trang
    st.set_page_config(
        page_title="OCR Căn Cước Công Dân",
        page_icon="🪪",
        layout="wide"
    )
    
    st.title("🪪 Trích xuất thông tin Căn Cước Công Dân")
    st.markdown("### Tải lên hoặc chụp ảnh căn cước công dân để trích xuất thông tin")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1️⃣ Chọn phương thức nhập ảnh")
        input_method = st.radio(
            "Chọn cách thức:",
            ("Tải ảnh lên", "Chụp ảnh")
        )
        
        if input_method == "Tải ảnh lên":
            uploaded_file = st.file_uploader(
                "Chọn ảnh căn cước công dân",
                type=['png', 'jpg', 'jpeg']
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
                
                if st.button("Trích xuất thông tin"):
                    try:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format)
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        response = requests.post(
                            CARD2FILE_URL,
                            files={"file": ("file", img_byte_arr, "image/jpeg")},
                            data={"filename": uploaded_file.name} 
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            display_results(data, col2)
                        else:
                            st.error(f"Lỗi API: {response.json().get('detail')}")
                    
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}")
        
        elif input_method == "Chụp ảnh":
            st.write("📸 Camera")
            webrtc_ctx = webrtc_streamer(key="camera")
            
            if st.button("Chụp ảnh"):
                if webrtc_ctx and webrtc_ctx.video_transformer:
                    frame = webrtc_ctx.video_transformer.frame
                    if frame is not None:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        st.image(img, caption="Ảnh đã chụp", use_container_width=True)
                        
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        try:
                            response = requests.post(
                                CARD2FILE_URL,
                                files={"file": ("file", img_byte_arr, "image/jpeg")},
                                data={"filename": uploaded_file.name} 
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                display_results(data, col2)
                            else:
                                st.error(f"Lỗi API: {response.json().get('detail')}")
                        
                        except Exception as e:
                            st.error(f"Lỗi: {str(e)}")


def display_results(data, col):
    """Hiển thị kết quả key-value và lưu ra tệp PDF"""
    with col:
        st.subheader("2️⃣ Kết quả trích xuất")
        print(f'Data: {data}')
        # Hiển thị key-value theo từng dòng
        data = data['results']

        for key, value in data.items():
            st.markdown(f"**{key}:** {value}")
        
        pdf_output = io.BytesIO()
        c = canvas.Canvas(pdf_output)
        c.setFont("Helvetica", 12)

        c.drawString(100, 800, "Kết quả trích xuất căn cước công dân")
        y = 780
        for key, value in data.items():
            c.drawString(100, y, f"{key}: {value}")
            y -= 20

        c.save()
        pdf_output.seek(0)
        
        st.download_button(
            label="📥 Tải xuống PDF",
            data=pdf_output,
            file_name="cccd_data.pdf",
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()