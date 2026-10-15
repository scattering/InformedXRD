import h5py
import argparse

def repack_hdf5(input_file, output_file, batch_size=128, buffer_size=10_000):
    with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "w") as f_out:

        def copy_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Repacking dataset: {name}")

                # Build valid chunking tuple
                if len(obj.shape) == 1:
                    chunks = (batch_size,)
                else:
                    chunks = (batch_size,) + obj.shape[1:]

                # Create dataset in output
                d_out = f_out.create_dataset(
                    name,
                    shape=obj.shape,
                    dtype=obj.dtype,
                    chunks=chunks,
                    compression=None
                )

                # Stream copy
                n_samples = obj.shape[0]
                for i in range(0, n_samples, buffer_size):
                    end = min(i + buffer_size, n_samples)
                    d_out[i:end] = obj[i:end]

            elif isinstance(obj, h5py.Group):
                f_out.create_group(name)

        f_in.visititems(copy_dataset)

    print(f"✅ Repacked file saved to {output_file}")

# usage:
# cd project/Code/ViT_NVIDIA
#
python repack_hdf5.py \
    --input /scratch/10486/UserB/Data/rruff_aug_10M.hdf5.hdf5 \
    --output /scratch/10486/UserB/Data/rruff_aug_10M.hdf5_repacked.hdf5 \
    --batch_size 128 \
    --buffer_size 10000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=10000)
    args = parser.parse_args()

    repack_hdf5(args.input, args.output, args.batch_size, args.buffer_size)
