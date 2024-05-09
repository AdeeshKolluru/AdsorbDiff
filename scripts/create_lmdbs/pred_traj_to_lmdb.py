"""
Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import random
import sys
from typing import Literal
import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from adsorbdiff.utils.atoms_to_graphs import AtomsToGraphs
from adsorbdiff.placement import Adsorbate, Bulk, Slab, AdsorbateSlabConfig


def adsorbate_placement(
    adslab,
    sid: str,
    interstitial_gap: float = 0.1,
):

    ads_mask = tags_map[sid] == 2
    adslab.set_tags(tags_map[sid])
    ads_site = adslab.get_positions()[ads_mask].mean(axis=0)
    # adsorbate_atoms = Adsorbate(
    #     adsorbate_id_from_db=int(sid.split("_")[0]),
    #     adsorbate_db_path="/home/jovyan/repos/Open-Catalyst-Dataset/ocdata/databases/pkls/adsorbates.pkl",
    # )
    adsorbate_atoms = Adsorbate(adsorbate_atoms=adslab[ads_mask])
    bulk = Bulk(
        bulk_id_from_db=int(sid.split("_")[1]),
        bulk_db_path="/home/jovyan/repos/Open-Catalyst-Dataset/ocdata/databases/pkls/bulks.pkl",
    )
    slab_atoms = Slab(bulk=bulk, slab_atoms=adslab[~ads_mask])
    random_adslabs = AdsorbateSlabConfig(
        slab_atoms,
        adsorbate_atoms,
        interstitial_gap=interstitial_gap,
        sites=ads_site,
    )
    return random_adslabs.atoms_list[0]


def write_images_to_lmdb(mp_arg):
    a2g, db_path, samples, sampled_ids, idx, pid, args = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    print(db_path)

    with tqdm(total=len(samples), position=pid) as pbar:
        for traj_path in samples:
            # sid, fid = traj_path.split("/")[-1].split(".")[0].rsplit("_", 1)
            if traj_path.split("/")[-1].count("_") == 3:
                sid, fid = (
                    traj_path.split("/")[-1].split(".")[0].rsplit("_", 1)
                )
            elif traj_path.split("/")[-1].count("_") == 2:
                sid = traj_path.split("/")[-1].split(".")[0]
                fid = 0
            traj = ase.io.read(traj_path)

            # traj = adsorbate_placement(traj, sid)

            image = a2g.convert(traj)
            image.sid = sid
            image.fid = fid
            image.tags = torch.tensor(tags_map[sid])
            ads_idx = tags_map[sid] == 2
            surface_idx = tags_map[sid] == 1
            ads_pos = image.pos[ads_idx]
            surface_pos = image.pos[surface_idx]
            surface_pos_z_max = surface_pos[:, -1].max()
            ads_pos_z_min = ads_pos[:, -1].min()
            diff = ads_pos_z_min - surface_pos_z_max
            if diff < 0.1:
                ads_pos[:, -1] = ads_pos[:, -1] + abs(diff) + 0.1
                image.pos[ads_idx] = ads_pos
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(image, protocol=-1))
            txn.commit()
            idx += 1
            pbar.update(1)

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=False,
        default="/home/jovyan/shared-scratch/adeesh/denoising/relaxations/relaxations/randheur100_I0.1",
    )
    parser.add_argument(
        "--out-path",
        required=False,
        default="/home/jovyan/shared-scratch/adeesh/denoising/lmdbs/randheur100_I0.1_fmax0.03",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )

    parser.add_argument(
        "--tags", action="store_true", default=True, help="Add atom tags"
    )
    parser.add_argument(
        "--chunk", default=1, type=int, help="Chunk to of inputs to preprocess"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="doesn't parallelize for easy debugging",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    traj_dirs = glob.glob(f"{args.data_path}/*.traj")

    num_trajectories = len(traj_dirs)

    if args.tags:
        tag_path = "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_tags.pkl"
        with open(os.path.join(tag_path), "rb") as h:
            tags_map = pickle.load(h)

    with open(
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_mapping.pkl",
        "rb",
    ) as f:
        oc20_dense_mapping = pickle.load(f)

    print("### Found %d trajectories" % (num_trajectories))

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_fixed=True,
        r_distances=False,
    )

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    # Chunk the trajectories into args.num_workers splits
    chunked_traj_dirs = np.array_split(traj_dirs, args.num_workers)

    # Extract features
    sampled_ids, idx = [[]] * args.num_workers, [0] * args.num_workers

    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            a2g,
            db_paths[i],
            chunked_traj_dirs[i],
            sampled_ids[i],
            idx[i],
            i,
            args,
        )
        for i in range(args.num_workers)
    ]
    if args.debug:
        op = []
        for i in range(args.num_workers):
            op.append(write_images_to_lmdb(mp_args[i]))
            op = list(zip(*op))
    else:
        op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
