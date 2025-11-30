def compute_interaction_stats(file_path):
    user_ids = set()
    item_ids = set()
    interaction_count = 0

    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:  # 确保至少有用户ID和项目ID
                    user_ids.add(parts[0])
                    item_ids.add(parts[1])
                    interaction_count += 1

        num_users = len(user_ids)
        num_items = len(item_ids)
        total_possible_interactions = num_users * num_items

        if total_possible_interactions == 0:
            sparsity = 0.0  # 避免除零错误
        else:
            sparsity = 1 - (interaction_count / total_possible_interactions)

        print(f"用户数量: {num_users}")
        print(f"项目数量: {num_items}")
        print(f"交互数量: {interaction_count}")
        print(f"交互稀疏度: {sparsity:.6f} (即 {sparsity * 100:.2f}%)")

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# 使用示例
compute_interaction_stats('./datasets/processed/dianping/ratings.data')