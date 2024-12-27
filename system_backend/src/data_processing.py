import pandas as pd
import re
from difflib import SequenceMatcher
import os
from dotenv import load_dotenv

# load_dotenv()

                        #===================sắp xếp theo file ảnh ==============
def load_and_sort_csv(path_file_csv):
    df = pd.read_csv(path_file_csv)
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1
    df['file_number'] = df['File Path'].apply(lambda x: extract_number(x))
    df = df.sort_values(by='file_number').drop(columns='file_number')
    df.to_csv(path_file_csv, index=False)
    return df

                        #====================gom tọa độ=================================
def coordinates(path_img_detect):
    for i in os.listdir(path_img_detect):
        if i.endswith('.txt'):
            name_txt = i
    path_file_txt = os.path.join(path_img_detect, name_txt)

    with open(path_file_txt, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        values = line.strip().split(',')
        if len(values) == 8: 
            data.append(values)   
    columns = ['tren_trai_x', 'tren_trai_y', 'tren_phai_x', 'tren_phai_y', 'duoi_phai_x', 'duoi_phai_y', 'duoi_trai_x', 'duoi_trai_y']
    df_coordinates = pd.DataFrame(data, columns=columns)
    return df_coordinates

                        #====================ghép 2 file=================================
def merged(df, df_coordinates):
    result = pd.concat([df, df_coordinates], axis=1)
    return result

                        #====================gom dữ liệu theo hàng gom tọa độ============

def gom(result):
    result['tren_trai_x'] = pd.to_numeric(result['tren_trai_x'], errors='coerce')
    result['tren_trai_y'] = pd.to_numeric(result['tren_trai_y'], errors='coerce')

    data_1 = result[['tren_trai_y', 'tren_trai_x', 'Extracted Text']].dropna().sort_values(by='tren_trai_x')

    values_1 = data_1[['tren_trai_y', 'tren_trai_x', 'Extracted Text']].values.tolist()

    x_groups = []
    current_x_group = [values_1.pop(0)]

    i = 0
    while i < len(values_1):
        if abs(values_1[i][1] - current_x_group[0][1]) <= 200:  # So sánh tọa độ x
            current_x_group.append(values_1.pop(i))
        else:
            i += 1
    x_groups.append(current_x_group)

    data_1 = data_1[~data_1[['tren_trai_y', 'tren_trai_x', 'Extracted Text']].apply(tuple, axis=1).isin([tuple(item) for group in x_groups for item in group])]

    result_1 = []

    for x_group in x_groups:
        y_values = sorted(x_group, key=lambda x: x[0])  # Sắp xếp theo tọa độ y
        y_groups = []
        current_y_group = [y_values[0]]

        for i in range(1, len(y_values)):
            if abs(y_values[i][0] - current_y_group[-1][0]) <= 10:  # So sánh tọa độ y
                current_y_group.append(y_values[i])
            else:
                y_groups.append(current_y_group)
                current_y_group = [y_values[i]]
        y_groups.append(current_y_group)

        for y_group in y_groups:
            sorted_group = sorted(y_group, key=lambda x: x[1])  # Sắp xếp theo tọa độ x
            extracted_texts = " ".join(row[2] for row in sorted_group)
            result_1.append(extracted_texts)

    data_2 = data_1.dropna().sort_values(by='tren_trai_y')
    values_2 = data_2[['tren_trai_y', 'tren_trai_x', 'Extracted Text']].values.tolist()

    groups = []
    current_group = [values_2[0]]

    for i in range(1, len(values_2)):
        if abs(values_2[i][0] - current_group[-1][0]) <= 10:  # So sánh tọa độ y
            current_group.append(values_2[i])
        else:
            groups.append(current_group)
            current_group = [values_2[i]]
    groups.append(current_group)

    result_2 = []

    for group in groups:
        sorted_group = sorted(group, key=lambda x: x[1])  # Sắp xếp theo tọa độ x
        extracted_texts = " ".join(row[2] for row in sorted_group)
        result_2.append(extracted_texts)
    
    for idx, group_text in enumerate(result_2):
        print(f"Nhóm {idx + 1}: {group_text}")

    return result_1, result_2


                        #====================xử lí ngày hết hạn thẻ======================

def ngay_het_han(result_1):
    ngay_het_han = [] 
    pattern_expiration = r'\b\d{2}/\d{2}/\d{4}\b'
    for text in result_1:
        matches = re.findall(pattern_expiration, text)
        ngay_het_han.extend(matches)  
    ngay_het_han_str = ', '.join(ngay_het_han)
    return ngay_het_han_str

                        #====================xử lí ngày sinh ============================
def ngay_sinh(result_2):
    ngay_sinh = []
    pattern_birthday = r'\b\d{2}/\d{2}/\d{4}\b'

    result_2 = [text.replace(' ', '') for text in result_2]

    for text in result_2:
        matches = re.findall(pattern_birthday, text)
        ngay_sinh.extend(matches)

    result_2 = [text for text in result_2 if not re.findall(pattern_birthday, text)]

    ngay_sinh_str = ', '.join(ngay_sinh)
    
    return ngay_sinh_str

                        #=====================xử lí số căn cước =========================

def so_can_cuoc(result_2):
    ID_numbers = []

    pattern_ID_numbers = r'\b\d{12}\b'

    for text in result_2:
        matches = re.findall(pattern_ID_numbers, text)
        ID_numbers.extend(matches)

    ID_numbers_str = ', '.join(ID_numbers)

    result_2 = [text for text in result_2 if not re.findall(pattern_ID_numbers, text)]

    return ID_numbers_str


                        #=====================xử lí quê quán =========================

def preprocess_text(text):
        return re.sub(r'\W+', '', text).lower()

def que_quan(result_2):
    text_to_check_1 = "Quêquán/Placeoforigin"

    data = result_2

    processed_text_to_check = preprocess_text(text_to_check_1)

    highest_similarity_1 = 0
    most_similar_text_1 = None  

    for item in data:
        processed_item = preprocess_text(item)
        similarity = SequenceMatcher(None, processed_text_to_check, processed_item).ratio()
        if similarity > highest_similarity_1:
            highest_similarity_1 = similarity
            most_similar_text_1 = item

    if most_similar_text_1:
        print(f"Đoạn văn bản trùng khớp cao nhất: '{most_similar_text_1}'")
        print(f"Mức độ tương đồng: {highest_similarity_1 * 100:.2f}%")
    else:
        print("Không có đoạn văn bản nào phù hợp.")

    for index, i in enumerate(result_2):
        if i == most_similar_text_1:  
            if index + 1 < len(result_2): 
                combined_text = f"{i} {result_2[index + 1]}"
                remaining_text = combined_text.replace(most_similar_text_1, "").strip() 
                result_2[index] = remaining_text
                del result_2[index + 1]  
            else:
                remaining_text = i.replace(most_similar_text_1, "").strip()
                result_2[index] = remaining_text

    else:
        print("Không tìm thấy đoạn văn bản trùng khớp.")
    
    return remaining_text

                        #=====================xử lí nơi thường trú =========================
def thuong_tru(result_2):
    text_to_check_2 = "Nơithườngtrú/Placeofresidence"

    data = result_2

    processed_text_to_check = preprocess_text(text_to_check_2)

    highest_similarity_2 = 0
    most_similar_text_2 = None  

    for item in data:
        processed_item = preprocess_text(item)
        similarity = SequenceMatcher(None, processed_text_to_check, processed_item).ratio()
        if similarity > highest_similarity_2:
            highest_similarity_2 = similarity
            most_similar_text_2 = item

    if most_similar_text_2:
        print(f"Đoạn văn bản trùng khớp cao nhất: '{most_similar_text_2}'")
        print(f"Mức độ tương đồng: {highest_similarity_2 * 100:.2f}%")
    else:
        print("Không có đoạn văn bản nào phù hợp.")

    for index, i in enumerate(result_2):
        if i == most_similar_text_2:  
            if index + 1 < len(result_2): 
                combined_text = f"{i} {result_2[index + 1]}"
                remaining_text = combined_text.replace(most_similar_text_2, "").strip() 
                result_2[index] = remaining_text
                del result_2[index + 1]  
            else:
                remaining_text = i.replace(most_similar_text_2, "").strip()
                result_2[index] = remaining_text
            
    else:
        print("Không tìm thấy đoạn văn bản trùng khớp.")
    return remaining_text

                        #=====================xử lí họ ten =========================
def ho_ten(result_2):
    text_to_check_3 = "Họ và tên/Full name"

    def preprocess_text(text):
        return re.sub(r'[^a-zA-Z0-9 ]+', '', text).lower().strip()

    processed_text_to_check = preprocess_text(text_to_check_3)

    highest_similarity_3 = 0
    most_similar_text_3 = None
    most_similar_index = -1

    for index, item in enumerate(result_2):
        processed_item = preprocess_text(item)
        similarity = SequenceMatcher(None, processed_text_to_check, processed_item).ratio()
        print(f"Comparing: '{item}' -> Similarity: {similarity:.2f}")
        
        if similarity > highest_similarity_3:
            highest_similarity_3 = similarity
            most_similar_text_3 = item
            most_similar_index = index

    if most_similar_text_3:
        print(f"Đoạn văn bản trùng khớp cao nhất: '{most_similar_text_3}'")
        print(f"Mức độ tương đồng: {highest_similarity_3 * 100:.2f}%")
    else:
        print("Không có đoạn văn bản nào phù hợp.")
        return None  

    if most_similar_index + 1 < len(result_2):  
        value = result_2[most_similar_index + 1]
        print(f"Giá trị ở hàng bên dưới: '{value}'")
        return value
    else:
        print("Không có hàng bên dưới đoạn văn bản trùng khớp.")
        return None



def add_dict(TT, result_1, result_2):
    TT["Số/No"] = so_can_cuoc(result_2)
    TT["Họ và tên/Full name"] = ho_ten(result_2)

    if any('Nam' in item for item in result_2):  
        TT["Giới tính/Sex"] = "Nam"
    else: 
        TT["Giới tính/Sex"] = "Nữ"

    TT["Ngày sinh/Date of birthday"] = ngay_sinh(result_2)
    TT["Quốc tịch/Nationality"] = 'Việt Nam'
    TT["Quê quán/Place of origin"] = que_quan(result_2)
    TT["Nơi thường trú/Place of residence"] = thuong_tru(result_2)
    TT["Có giá trị đến/Date of expiry"] = ngay_het_han(result_1)



def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            os.unlink(file_path)


def post_processing(path_file_csv: str, path_dir_img_detect: str) -> dict:
    df = load_and_sort_csv(path_file_csv=path_file_csv)
    df_coordinates = coordinates(path_dir_img_detect)
    result = merged(df, df_coordinates)
    result_1, result_2 = gom(result)
    TT = {}
    add_dict(TT, result_1, result_2)

    return TT
    
# clear_folder('./database/after_preprocessing')
# clear_folder('./database/craft_image/img_detect')
# clear_folder('./database/craft_image/cutting_image')


