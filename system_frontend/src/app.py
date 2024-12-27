import streamlit as st
import requests
from PIL import Image
import io
import os
import json
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
from fpdf import FPDF

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
        
        # Thêm nút xuất kết quả ra PDF
        if st.button("📥 Lưu kết quả dưới dạng PDF"):
            try:
                # Tạo PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Kết quả trích xuất căn cước công dân", ln=True, align='C')

                for key, value in data.items():
                    pdf.cell(0, 10, txt=f"{key}: {value}", ln=True)

                # Lưu PDF vào BytesIO
                pdf_output = io.BytesIO()
                pdf.output(pdf_output)
                pdf_output.seek(0)  # Đảm bảo con trỏ ở đầu stream

                # Tạo nút tải xuống PDF
                st.download_button(
                    label="📥 Tải xuống PDF",
                    data=pdf_output,
                    file_name="cccd_data.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Lỗi khi tạo PDF: {str(e)}")


if __name__ == "__main__":
    main()