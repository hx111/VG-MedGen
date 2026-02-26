import os
import pandas as pd


def create_raw_metadata(train_dir, output_csv_path):
    if not os.path.isdir(train_dir):
        print(f"错误: 训练目录 '{train_dir}' 不存在。请检查路径。")
        return

    file_data = []

    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    print(f"在 '{train_dir}' 中找到以下类别: {class_names}")

    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)

        for filename in os.listdir(class_dir):
            if filename.lower().endswith('.jpg'):
                natural_class_name = class_name
                prompt_text = f"A dermoscopic image of {natural_class_name}"

                relative_path = os.path.join(class_name, filename)

                file_data.append({
                    'file_name': relative_path,
                    'text': prompt_text
                })

    if not file_data:
        print("警告: 在指定目录中没有找到任何图片文件。")
        return

    df = pd.DataFrame(file_data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

    print("-" * 50)
    print(f"成功生成 'raw.csv' 文件！")
    print(f"文件位置: {output_csv_path}")
    print(f"总计包含 {len(df)} 条记录。")
    print("现在您可以运行 MyVSG1.py 脚本了。")


if __name__ == '__main__':
    TRAIN_DATA_DIRECTORY = '../dataset/HAM10000/train'

    OUTPUT_CSV_FILE = os.path.join(TRAIN_DATA_DIRECTORY, 'raw.csv')

    create_raw_metadata(TRAIN_DATA_DIRECTORY, OUTPUT_CSV_FILE)
