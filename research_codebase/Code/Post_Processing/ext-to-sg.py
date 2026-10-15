import pkl
import h5py
import numpy as np

with open('SG_To_Ext_Map.pkl', 'rb') as file:
    dic = pickle.load(file)

def sgToExt(x):
    return dic[x]

def create_ext_groups(input_path, output_path):
    vectorized_func = np.vectorize(sgToExt)
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        y_val_transformed = vectorized_func(f_in['y_val'][:])
        y_train_transformed = vectorized_func(f_in['y_train'][:])
        y_test_transformed = vectorized_func(f_in['y_test'][:])

        X_all = np.concatenate([f_in['X_train'], f_in['X_val'], f_in['X_test']], axis=0)
        y_all = np.concatenate([y_train_transformed, y_val_transformed, y_test_transformed], axis=0)

        label_to_indices = defaultdict(list)
        for idx, label in enumerate(y_all):
            label_to_indices[label].append(idx)

        sampled_indices = []
        for label in range(1, 100):
            indices = label_to_indices[label]
            if len(indices) == 0:
                continue
            sampled = np.random.choice(indices, size=min(int(len(y_all)/230), len(indices)), replace=False)
            sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)

        X_sampled = X_all[sampled_indices]
        y_sampled = y_all[sampled_indices]
        #counts = Counter(y_sampled)
        #plt.bar(counts.keys(), counts.values())
        # Split into train (90%), val (5%), test (5%) — stratified
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_sampled, y_sampled, test_size=0.10, stratify=y_sampled, random_state=42)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        f_out.create_dataset('X_train', data=X_train)
        f_out.create_dataset('y_train', data=y_train)
        f_out.create_dataset('X_val', data=X_val)
        f_out.create_dataset('y_val', data=y_val)
        f_out.create_dataset('X_test', data=X_test)
        f_out.create_dataset('y_test', data=y_test)

        print("Balanced and stratified data saved.")
        print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
