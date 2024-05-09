import pickle
from adsorbdiff.datasets import LmdbDataset as LD
from tqdm import tqdm

if __name__ == "__main__":
    lmdb = LD(
        {
            "src": "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/train_is2re"
        }
    )

    with open("oc20dense_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    system_ids = {}
    for i in tqdm(range(len(lmdb))):
        data = lmdb[i]
        config = data.sid
        system_id = mapping[config]["system_id"]
        try:
            system_ids[system_id] += 1
        except KeyError:
            system_ids[system_id] = 1

    with open("unique_train_system_id.txt", "w") as f:
        for system_id in system_ids.keys():
            f.write(system_id)
            f.write("\n")
