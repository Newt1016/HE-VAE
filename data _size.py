def count_users_and_items(file_path):
    # 初始化集合来存储唯一的用户ID和项目ID
    user_ids = set()
    item_ids = set()

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 分割每行数据
                parts = line.strip().split()
                if len(parts) >= 2:  # 确保至少有用户ID和项目ID
                    user_ids.add(parts[0])
                    item_ids.add(parts[1])

        print(f"用户数量: {len(user_ids)}")
        print(f"项目数量: {len(item_ids)}")

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")


# 使用示例
count_users_and_items('./datasets/processed/ML/ratings.data')