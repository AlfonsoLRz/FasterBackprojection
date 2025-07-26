import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat, savemat
from scipy.fft import fftn, ifftn
from scipy.sparse import spdiags, lil_matrix
import time


def cnlos_reconstruction(scene=7):
    """
    Confocal Non-Line-of-Sight (C-NLOS) reconstruction procedure

    Parameters:
    scene (int): Integer between 1 and 9 corresponding to different scenes:
        1 - resolution chart at 40cm from wall
        2 - resolution chart at 65cm from wall
        3 - dot chart at 40cm from wall
        4 - dot chart at 65cm from wall
        5 - mannequin
        6 - exit sign
        7 - "SU" scene (default)
        8 - outdoor "S"
        9 - diffuse "S"
    """
    # Constants
    bin_resolution = 4e-12  # Native bin resolution for SPAD is 4 ps
    c = 3e8  # Speed of light (meters per second)

    # Adjustable parameters
    isbackprop = 0  # Toggle backprojection
    isdiffuse = 0  # Toggle diffuse reflection
    K = 1  # Downsample data to (4 ps) * 2^K = 16 ps for K = 2
    snr = 8e-1  # SNR value
    z_trim = 600  # Set first 600 bins to zero

    # Load scene & set visualization parameter
    if scene == 1:
        data = loadmat('lct/data_resolution_chart_40cm.mat')
        savemat('C:/Datasets/transient/lct/data_resolution_chart_40cm.mat', data)
        z_offset = 350
    elif scene == 2:
        data = loadmat('lct/data_resolution_chart_65cm.mat')
        savemat('C:/Datasets/transient/lct/data_resolution_chart_65cm.mat', data)
        z_offset = 700
    elif scene == 3:
        data = loadmat('lct/data_dot_chart_40cm.mat')
        savemat('C:/Datasets/transient/lct/data_dot_chart_40cm.mat', data)
        z_offset = 350
    elif scene == 4:
        data = loadmat('lct/data_dot_chart_65cm.mat')
        savemat('C:/Datasets/transient/lct/data_dot_chart_65cm.mat', data)
        z_offset = 700
    elif scene == 5:
        data = loadmat('lct/data_mannequin.mat')
        savemat('C:/Datasets/transient/lct/data_mannequin.mat', data)
        z_offset = 300
    elif scene == 6:
        data = loadmat('lct/data_exit_sign.mat')
        savemat('C:/Datasets/transient/lct/data_exit_sign.mat', data)
        z_offset = 600
    elif scene == 7:
        data = loadmat('lct/data_s_u.mat')
        savemat('C:/Datasets/transient/lct/data_s_u.mat', data)
        z_offset = 800
    elif scene == 8:
        data = loadmat('lct/data_outdoor_s.mat')
        savemat('C:/Datasets/transient/lct/data_outdoor_s.mat', data)
        z_offset = 700
    elif scene == 9:
        data = loadmat('lct/data_diffuse_s.mat')
        savemat('C:/Datasets/transient/lct/data_diffuse_s.mat', data)
        z_offset = 100
        isdiffuse = 1
        snr = snr * 1e-1

    rect_data = data['rect_data']  # Load rectangular data
    width = float(data['width'])

    N = rect_data.shape[0]  # Spatial resolution of data
    M = rect_data.shape[2]  # Temporal resolution of data
    range_ = M * c * bin_resolution  # Maximum range for histogram

    # Downsample data to 16 picoseconds
    # for k in range(K):
    #     M = M // 2
    #     bin_resolution *= 2
    #     rect_data = rect_data[:, :, ::2] + rect_data[:, :, 1::2]
    #     z_trim = round(z_trim / 2)
    #     z_offset = round(z_offset / 2)

    # Set first group of histogram bins to zero (to remove direct component)
    rect_data[:, :, :z_trim] = 0

    print(rect_data[0, 0, 1024:1034])  # Print first 10 bins of the first pixel for debugging

    # Define NLOS blur kernel
    psf = define_psf(N, M, width / 2.0 / range_, plot=True)

    # Compute inverse filter of NLOS blur kernel
    fpsf = fftn(psf)
    if not isbackprop:
        invpsf = np.conj(fpsf) / (np.abs(fpsf) ** 2 + 1 / snr)
    else:
        invpsf = np.conj(fpsf)

    # Define transform operators
    mtx, mtxi = resampling_operator(M)

    # Permute data dimensions
    data = np.transpose(rect_data, (2, 1, 0))

    # Define volume representing voxel distance from wall
    grid_z = np.tile(np.linspace(0, 1, M)[:, np.newaxis, np.newaxis], (1, N, N))

    print('Inverting...')
    start_time = time.time()

    # Step 1: Scale radiometric component
    if isdiffuse:
        data = data * (grid_z ** 4)
    else:
        data = data * (grid_z ** 2)

    # Step 2: Resample time axis and pad result
    data_reshaped = data.reshape(M, N * N)
    transformed_data = mtx @ data_reshaped
    tdata_mnn = transformed_data.reshape(M, N, N)

    # Initialize tdata with zeros (2*M, 2*N, 2*N)
    tdata = np.zeros((2 * M, 2 * N, 2 * N), dtype=np.complex128)  # Use complex for FFT
    # Assign the transformed data to the first M x N x N part
    tdata[:M, :N, :N] = tdata_mnn

    # Step 3: Convolve with inverse filter and unpad result
    # MATLAB: tvol = ifftn(fftn(tdata).*invpsf);
    # The dimensions of tdata (2M, 2N, 2N) and invpsf (2M, 2N, 2N) must match for element-wise multiplication.
    # `definePsf` creates psf of size (2*V, 2*U, 2*U) which is (2M, 2N, 2N). This matches.

    tvol = np.fft.ifftn(np.fft.fftn(tdata) * invpsf)

    # MATLAB: tvol = tvol(1:end./2,1:end./2,1:end./2);
    tvol = tvol[:M, :N, :N]  # Truncate back to M x N x N

    # Step 4: Resample depth axis and clamp results
    tvol_reshaped = tvol.reshape(M, N * N)
    transformed_tvol = mtxi @ tvol_reshaped
    vol = transformed_tvol.reshape(M, N, N)

    print('... done.')
    time_elapsed = time.time() - start_time

    print(
        f'Reconstructed volume of size {vol.shape[2]} x {vol.shape[1]} x {vol.shape[0]} in {time_elapsed:.2f} seconds')

    tic_z = np.linspace(0, range_ / 2, vol.shape[0])
    tic_y = np.linspace(-width, width, vol.shape[1])
    tic_x = np.linspace(-width, width, vol.shape[2])

    # Crop and flip reconstructed volume for visualization
    ind = round(M * 2 * width / (range_ / 2))
    vol = vol[:, :, ::-1]  # Flip along x-axis
    vol = vol[z_offset:z_offset + ind, :, :]

    # convert from complex128 to float64
    vol = np.abs(vol).astype(np.float64)

    tic_z = tic_z[z_offset:z_offset + ind]

    # View result
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(np.max(vol, axis=0), extent=[tic_x[0], tic_x[-1], tic_y[0], tic_y[-1]], cmap='gray')
    plt.title('Front view')
    plt.xticks(np.linspace(tic_x[0], tic_x[-1], 3))
    plt.yticks(np.linspace(tic_y[0], tic_y[-1], 3))
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().set_aspect('equal')

    plt.subplot(1, 3, 2)
    plt.imshow(np.max(vol, axis=1), extent=[tic_x[0], tic_x[-1], tic_z[0], tic_z[-1]], cmap='gray')
    plt.title('Top view')
    plt.xticks(np.linspace(tic_x[0], tic_x[-1], 3))
    plt.yticks(np.linspace(tic_z[0], tic_z[-1], 3))
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.gca().set_aspect('equal')

    plt.subplot(1, 3, 3)
    plt.imshow(np.max(vol, axis=2).T, extent=[tic_z[0], tic_z[-1], tic_y[0], tic_y[-1]], cmap='gray')
    plt.title('Side view')
    plt.xticks(np.linspace(tic_z[0], tic_z[-1], 3))
    plt.yticks(np.linspace(tic_y[0], tic_y[-1], 3))
    plt.xlabel('z (m)')
    plt.ylabel('y (m)')
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.show()


def define_psf(U, V, slope, plot=False):
    print(32 * slope * slope)

    """Compute NLOS blur kernel"""
    x = np.linspace(-1, 1, 2 * U)
    y = np.linspace(-1, 1, 2 * U)
    z = np.linspace(0, 2, 2 * V)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')

    # Define PSF
    psf = np.abs(((4 * slope) ** 2) * (grid_x ** 2 + grid_y ** 2) - grid_z)
    psf = (psf == np.min(psf, axis=0, keepdims=True)).astype(float)
    psf = psf / np.sum(psf[:, U, U])
    psf = psf / np.linalg.norm(psf)
    psf = np.roll(psf, U, axis=(1, 2))

    # Maximum slice (z) with non-zero elements
    max_slice = np.max(np.where(psf > 0, psf, 0), axis=(1, 2))
    print("Maximum slice of PSF:", max_slice)

    if plot:
        # Get the 3D coordinates of the non-zero PSF elements (the ellipse surface)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        mask = psf > 0

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xx[mask], yy[mask], zz[mask], s=1, c='blue', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Ellipse from PSF')
        ax.view_init(elev=20, azim=135)
        plt.tight_layout()
        plt.show()

    return psf


def resampling_operator(M):
    """Define resampling operators"""
    mtx = lil_matrix((M ** 2, M))

    x = np.arange(1, M ** 2 + 1)
    rows = x - 1
    cols = np.ceil(np.sqrt(x)) - 1
    mtx[rows, cols.astype(int)] = 1

    # Normalize
    mtx = spdiags(1 / np.sqrt(x), 0, M ** 2, M ** 2) @ mtx
    mtxi = mtx.T

    K = int(np.round(np.log2(M)))
    for k in range(K):
        mtx = 0.5 * (mtx[::2] + mtx[1::2])
        mtxi = 0.5 * (mtxi[:, ::2] + mtxi[:, 1::2])

    return mtx, mtxi

# Example usage:
cnlos_reconstruction(scene=5)