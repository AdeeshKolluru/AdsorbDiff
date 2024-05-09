import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import ase.io
import torch
import math

MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 1000
X_N = 2000

"""
    Preprocessing for the SO(3) sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""


def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rigid_transform_Kabsch_3D_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T
    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert (
        math.fabs(torch.linalg.det(R) - 1) < 3e-3
    )  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B
    return R, t


omegas = np.linspace(0, np.pi, X_N + 1)[1:]


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(
        Rotation.from_rotvec(r1).as_matrix()
        @ Rotation.from_rotvec(r2).as_matrix()
    ).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2)
            * np.sin(omega * (l + 1 / 2))
            / np.sin(omega / 2)
        )
    return p


def _density(
    expansion, omega, marginal=True
):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return (
            expansion / 8 / np.pi**2
        )  # the constant factor doesn't affect any actual calculations though


def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2)
            * (lo * dhi - hi * dlo)
            / lo**2
        )
    return dSigma / exp


PATH = "/home/jovyan/shared-scratch/adeesh/denoising/so3_precompute/"
if os.path.exists(os.path.join(PATH, "so3_omegas_array2.npy")):
    _omegas_array = np.load(os.path.join(PATH, "so3_omegas_array2.npy"))
    _cdf_vals = np.load(os.path.join(PATH, "so3_cdf_vals2.npy"))
    _score_norms = np.load(os.path.join(PATH, "so3_score_norms2.npy"))
    _exp_score_norms = np.load(os.path.join(PATH, "so3_exp_score_norms2.npy"))
else:
    print("Precomputing and saving to cache SO(3) distribution table")
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    _exp_vals = np.asarray(
        [_expansion(_omegas_array, eps) for eps in _eps_array]
    )
    _pdf_vals = np.asarray(
        [_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals]
    )
    _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals])
    _score_norms = np.asarray(
        [
            _score(_exp_vals[i], _omegas_array, _eps_array[i])
            for i in range(len(_eps_array))
        ]
    )

    _exp_score_norms = np.sqrt(
        np.sum(_score_norms**2 * _pdf_vals, axis=1)
        / np.sum(_pdf_vals, axis=1)
        / np.pi
    )

    np.save(os.path.join(PATH, "so3_omegas_array2.npy"), _omegas_array)
    np.save(os.path.join(PATH, "so3_cdf_vals2.npy"), _cdf_vals)
    np.save(os.path.join(PATH, "so3_score_norms2.npy"), _score_norms)
    np.save(os.path.join(PATH, "so3_exp_score_norms2.npy"), _exp_score_norms)


def sample(eps):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def sample_vec(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps, vec):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps):
    eps = eps.numpy()
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()


def rotate_atoms(atoms):

    # Rotate around the z-axis
    zrot = torch.rand(1) * 360
    zrot_rad = zrot * (math.pi / 180)  # Convert to radians
    rotation_matrix = torch.tensor(
        [
            [torch.cos(zrot_rad), -torch.sin(zrot_rad), 0],
            [torch.sin(zrot_rad), torch.cos(zrot_rad), 0],
            [0, 0, 1],
        ],
        device=self.device,
    )
    center = atoms.mean(dim=0)
    atoms_centered = atoms - center
    atoms_rotated_z = torch.mm(atoms_centered, rotation_matrix) + center

    # Generate a random rotation vector
    z = torch.rand(1) * 2 - 1
    phi = torch.rand(1) * 2 * math.pi
    rotvec = torch.tensor(
        [
            torch.sqrt(1 - z**2) * torch.cos(phi),
            torch.sqrt(1 - z**2) * torch.sin(phi),
            z,
        ],
        device=self.device,
    )

    # Rotate atoms using the generated rotation vector
    rotation_matrix = torch.tensor(
        [
            [
                1 - 2 * rotvec[1] ** 2 - 2 * rotvec[2] ** 2,
                2 * rotvec[0] * rotvec[1] - 2 * rotvec[2] * rotvec[2],
                2 * rotvec[0] * rotvec[2] + 2 * rotvec[1] * rotvec[2],
            ],
            [
                2 * rotvec[0] * rotvec[1] + 2 * rotvec[2] * rotvec[2],
                1 - 2 * rotvec[0] ** 2 - 2 * rotvec[2] ** 2,
                2 * rotvec[1] * rotvec[2] - 2 * rotvec[0] * rotvec[2],
            ],
            [
                2 * rotvec[0] * rotvec[2] - 2 * rotvec[1] * rotvec[2],
                2 * rotvec[1] * rotvec[2] + 2 * rotvec[0] * rotvec[2],
                1 - 2 * rotvec[0] ** 2 - 2 * rotvec[1] ** 2,
            ],
        ],
        device=self.device,
    )

    center = atoms_rotated_z.mean(dim=0)
    atoms_centered = atoms_rotated_z - center
    atoms_rotated = torch.mm(atoms_centered, rotation_matrix) + center

    return atoms_rotated


if __name__ == "__main__":

    tag_path = (
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_tags.pkl"
    )
    with open(os.path.join(tag_path), "rb") as h:
        tags_map = pickle.load(h)
    ads_idx = tags_map["2_2861_5"] == 2

    traj = ase.io.read(
        "/home/jovyan/shared-scratch/adeesh/denoising/overfit_pbccorr/overfit-xy_std0.01-10_numstep50x10_lr1.e-4_sample1/2_2861_5.traj",
        ":",
    )
    init_system = traj[0]
    ads_positions = init_system.positions[ads_idx]
