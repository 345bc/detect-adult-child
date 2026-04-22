import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.title("Adult/Child detection")
upload_file=st.file_uploader(
    label="Chọn file để upload",
    type=["mp4","png","jpg"],
    accept_multiple_files=False,
    help="Chỉ chấp nhận file mp4, png, jpg"
)


#Nếu sử dụng nhiều ảnh
# if upload_file is not None:
#     for file in upload_file:  # ← DUYỆT QUA TỪNG FILE TRONG LIST
#         if file.type.startswith('image'):  # Chỉ xử lý file ảnh
#             image = Image.open(file)
            
#             st.write(f"### Ảnh: {file.name}")
#             st.image(image, caption=file.name, use_column_width=False)
            
#             st.write("Thông tin ảnh")
#             st.write(f"Kích thước ảnh: {image.size}")
#             st.write(f"Định dạng: {image.format}")
#             st.write("---")


# cam_photo=st.camera_input("Chụp ảnh")

if upload_file is  not None:
    image=Image.open(upload_file)

    st.write("Ảnh bạn vừa upload")
    #use_column_width=False: giữ nguyên ảnh gốc
    st.image(image, caption=upload_file.name,use_container_width=False)

    st.write("Thông tin ảnh")
    st.write(f"Kích thước ảnh: {image.size}")
    st.write(f"Định dạng: {image.format}")