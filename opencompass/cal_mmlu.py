import os
import json


def read_all_json_files(directory):
    json_data = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            # 读取JSON文件内容
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
                json_data.append(data)

    return json_data


# 使用示例
directory_path = "/data/zfr/finalTest/opencompass/outputs/f_hh_mt_r_s/eval_mmlu/20240615_143153/results/llama-3-8b-ragga-disturb"
all_json_data = read_all_json_files(directory_path)

total = 0
num = 0
# 算平均acc
for json_content in all_json_data:
    d = json_content["details"]
    size = len(d)
    num += size
    total += size * json_content["accuracy"]
    print(json_content["accuracy"])

print('***************')
print(num)
print("asdassda")
print(total / num)

# directory_path = "/data/zfr/finalTest/opencompass/outputs/f_hyde_mt_r_s/eval_mmlu/20240611_130927/results/llama-3-8b-ragga-disturb"
# all_json_data = read_all_json_files(directory_path)


# num = 0
# num_wo = 0
# # 算wo检索数
# for json_content in all_json_data:
#     wo = 0
#     for item in json_content:
#         num += 1
#         if json_content[item]["context"] == "\n\n":
#             wo += 1
#     num_wo += wo
# print(num_wo)
# print(num)
