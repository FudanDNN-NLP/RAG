
filename = 'time.out'  # 假设文件名为 numbers.txt
with open(filename, 'r') as f:
    lines = f.readlines()  # 逐行读取文件内容


total_sum = 0.0
count = 0

# 遍历每一行，将其转换为浮点数并累加到总和中
for line in lines:
    try:
        number = float(line.strip())  # 去除首尾空白并转换为浮点数
        total_sum += number
        count += 1
    except ValueError:
        # 如果某行无法转换为浮点数，可以根据实际情况处理，这里假设忽略这些行
        continue

# 计算平均值
if count > 0:
    average = total_sum
    print(f"文件中浮点数的平均值为: {average}")
else:
    print("文件中没有有效的浮点数可以计算平均值。")