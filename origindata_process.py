import os
import json

# with open('sample_mapping_dict.json', 'r', encoding="utf-8") as f:
#     sample_mapping_dict = json.load(f)

# def get_json_files(folder):
#     target_files = []
#     for root, dirs, files in os.walk(folder):
#         for file in files:
#             if file == 'eye.json':
#                 target_files.append(os.path.join(root, file))
#     return target_files

# def file_filter(file_paths):
#     target_names = sample_mapping_dict.keys()
#     print(f"target name len: {len(target_names)}")
    
#     target_names = ['陈子豪']
    
#     filtered_files = []
#     for file_path in file_paths:
#         patient_name = file_path.split('\\')[-3]
#         patient_name = patient_name.split('_')[0]
#         # print(patient_name)
#         if patient_name in target_names:
#             filtered_files.append(file_path)
#     return filtered_files

# def save_to_new(file_path):
#     patient_name = file_path.split('\\')[-3]
#     patient_name = patient_name.split('_')[0]
#     patient_name = '陈子豪2'
#     id = sample_mapping_dict[patient_name]["id"]
#     label = sample_mapping_dict[patient_name]["label"]
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     eye_data = data["gazePoints"]
#     result = []
#     for item in eye_data:
#         new_item = {
#             "x": item["position"]["x"],
#             "y": item["position"]["y"]
#         }
#         result.append(new_item)
        
#     new_file = {
#         "id": id,
#         "name": patient_name,
#         "label": label,
#         "eye_data": result
#     }
    
#     new_file_path = os.path.join('F:\processed_eye', f"{id}_eye.json")
#     with open(new_file_path, 'w', encoding='utf-8') as f:
#         json.dump(new_file, f, indent=4, ensure_ascii=False)

# def main():
#     target_files = get_json_files('F:\ADHD数据')
#     print(f"-- number of target_files: {len(target_files)}")
    
#     target_files = file_filter(target_files)
#     print(f"-- after filtered: {len(target_files)}")
    
#     for file_path in target_files:
#         save_to_new(file_path)
#         print(f"- {file_path} done")

#     print(f"-- all done!")
    
# main()

def data_filter(folder_path):
    all_files = os.listdir(folder_path)
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        eye_data = data["eye_data"]
        
        new_data = []

        for idx, point in enumerate(eye_data):
            x, y = point["x"], point["y"]
            new_data.append([x, y])
        
        
        with open('test.json', 'w', encoding='utf-8') as f:
            data["eye_data"] = new_data
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        break
    
data_filter('exampledata')