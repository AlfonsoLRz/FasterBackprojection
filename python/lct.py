import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import os  # For checking file existence


def cnlos_reconstruction(scene=7):
    """
    Confocal Non-Light-of-Sight (C-NLOS) reconstruction procedure.

    Function takes as input an integer corresponding to a scene:
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
    isbackprop = False  # Toggle backprojection (0 in MATLAB is False)
    isdiffuse = False  # Toggle diffuse reflection (0 in MATLAB is False)
    K = 1  # Downsample data to (4 ps) * 2^K = 16 ps for K = 2
    snr = 8e-1  # SNR value
    z_trim = 600  # Set first 600 bins to zero

    # Load scene & set visualization parameter
    data_file = ""
    z_offset = 0

    if scene == 1:
        data_file = 'lct/data_resolution_chart_40cm.mat'
        z_offset = 350
    elif scene == 2:
        data_file = 'lct/data_resolution_chart_65cm.mat'
        z_offset = 700
    elif scene == 3:
        data_file = 'lct/data_dot_chart_40cm.mat'
        z_offset = 350
    elif scene == 4:
        data_file = 'lct/data_dot_chart_65cm.mat'
        z_offset = 700
    elif scene == 5:
        data_file = 'lct/data_mannequin.mat'
        z_offset = 300
    elif scene == 6:
        data_file = 'lct/data_exit_sign.mat'
        z_offset = 600
    elif scene == 7:
        data_file = 'lct/data_s_u.mat'
        z_offset = 800
    elif scene == 8:
        data_file = 'lct/data_outdoor_s.mat'
        z_offset = 700
    elif scene == 9:
        data_file = 'lct/data_diffuse_s.mat'
        z_offset = 100
        # Because the scene is diffuse, toggle the diffuse flag and
        # adjust SNR value correspondingly.
        isdiffuse = True  # 1 in MATLAB is True
        snr = snr * 1e-1
    else:
        raise ValueError("Invalid scene number. Choose between 1 and 9.")

    # Check if the data file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found. "
                                "Please ensure the .mat files are in the same directory "
                                "as the script or provide the full path.")

    mat_contents = scipy.io.loadmat(data_file)
    rect_data = mat_contents['rect_data']
    # Assuming 'width' is also in the .mat file or globally defined for the MATLAB code.
    # If not, you need to define 'width' here. For this example, I will assume it's loaded.
    # If 'width' is NOT in your .mat files, you will need to set it manually.
    if 'width' in mat_contents:
        width = mat_contents['width'].item()  # .item() to get scalar from 1x1 array
    else:
        # Placeholder: You MUST replace this with the actual 'width' value.
        # This is a critical parameter for definePsf.
        print("WARNING: 'width' variable not found in .mat file. Using a default value.")
        width = 0.4  # Example default, adjust as needed based on your scenes.

    print(rect_data.shape)
    N = rect_data.shape[0]  # Spatial resolution of data (size(rect_data,1))
    M = rect_data.shape[2]  # Temporal resolution of data (size(rect_data,3))
    range_max = M * c * bin_resolution  # Maximum range for histogram (range is a keyword in Python, renamed)

    # Downsample data to 16 picoseconds
    for k_downsample in range(K):  # MATLAB's 1:K means K iterations
        M = M // 2  # Integer division
        bin_resolution = 2 * bin_resolution
        # MATLAB: rect_data(:,:,1:2:end) + rect_data(:,:,2:2:end);
        # Python: rect_data[:, :, 0::2] + rect_data[:, :, 1::2]
        rect_data = rect_data[:, :, 0::2] + rect_data[:, :, 1::2]
        z_trim = round(z_trim / 2)
        z_offset = round(z_offset / 2)

    # Set first group of histogram bins to zero (to remove direct component)
    # MATLAB: rect_data(:,:,1:z_trim) = 0;
    rect_data[:, :, :z_trim] = 0

    # Define NLOS blur kernel
    psf = define_psf(N, M, width / range_max)

    # Compute inverse filter of NLOS blur kernel
    fpsf = np.fft.fftn(psf)
    if not isbackprop:
        # MATLAB: invpsf = conj(fpsf) ./ (abs(fpsf).^2 + 1./snr);
        invpsf = np.conj(fpsf) / (np.abs(fpsf) ** 2 + 1.0 / snr)
    else:
        invpsf = np.conj(fpsf)

    # Define transform operators
    mtx, mtxi = resampling_operator(M)

    # Permute data dimensions
    # MATLAB: data = permute(rect_data,[3 2 1]);
    # rect_data shape: (N_x, N_y, M_t)
    # desired data shape: (M_t, N_y, N_x)
    data = np.transpose(rect_data, (2, 1, 0))  # (dim2, dim1, dim0)

    # Define volume representing voxel distance from wall
    # MATLAB: grid_z = repmat(linspace(0,1,M)',[1 N N]);
    # linspace(0,1,M)' makes it a column vector (M,1)
    grid_z_col = np.linspace(0, 1, M)[:, np.newaxis]  # (M, 1)
    print(grid_z_col.shape)
    grid_z = np.tile(grid_z_col, (1, N, N)).reshape(M, N, N)  # (M, N, N) - replicates the column vector along new N, N dims
    print(grid_z.shape)

    print('Inverting...')
    tic_start = time.perf_counter()  # High resolution timer

    # Step 1: Scale radiometric component
    # MATLAB: data = data.*(grid_z.^4) or data = data.*(grid_z.^2);
    if isdiffuse:
        data = data * (grid_z ** 4)
    else:
        data = data * (grid_z ** 2)

    # Step 2: Resample time axis and pad result
    # MATLAB: tdata = zeros(2.*M,2.*N,2.*N);
    # MATLAB: tdata(1:end./2,1:end./2,1:end./2)  = reshape(mtx*data(:,:),[M N N]);

    # Reshaping data to 2D for matrix multiplication: (M, N*N)
    data_reshaped = data.reshape(M, N * N)

    # Matrix multiplication: (M_sq, M) @ (M, N*N) -> (M_sq, N*N)
    # The output of mtx * data(:,:) will have shape (M^2, N*N)
    # Then reshape to (M, N, N) according to MATLAB comment. This implies mtx is M x M^2.
    # This part of the MATLAB code is a bit ambiguous in its reshape after mtx*data(:,:) given the resamplingOperator.
    # The `resamplingOperator` for `mtx` leads to a matrix of `M^2 x M`. So `mtx * data_reshaped` would be `(M^2, N*N)`.
    # Reshaping this `(M^2, N*N)` result to `(M, N, N)` implies a complex transformation, likely incorrect dimension usage in the MATLAB comment.
    # Assuming the intent is that `mtx` is applied along the time dimension (first dim of `data`),
    # such that the result also has `M` time bins, but after a specific transformation.
    # The standard interpretation of `reshape(mtx*data(:,:),[M N N])` is that the output of `mtx*data(:,:)` has a total of `M*N*N` elements
    # and is then reshaped into `(M, N, N)`. This means `mtx*data(:,:)` must itself be `(M, N*N)`.
    # But `mtx` from `resamplingOperator` is `(M^2, M)`.
    # This implies that `mtx` is applied to each (N,N) slice along the M dimension, or some form of upsampling.
    # Given the previous context from C++/CUDA translation, this is likely a custom non-linear resampling.
    # Re-evaluating the `resamplingOperator` output:
    # `mtx` is `M^2 x M`. `data(:,:)` (reshaped data) is `M x (N*N)`.
    # So `mtx @ data_reshaped` will be `(M^2, N*N)`.
    # Then, `reshape(mtx*data(:,:),[M N N])` implies taking the first M*N*N elements and reshaping.
    # This is a critical point. If it's `tdata(1:end./2,1:end./2,1:end./2)` that gets the `M N N` data,
    # it means the `mtx*data` operation *produces* a shape that is then truncated/reshaped.
    # The most likely interpretation of `tdata(1:end./2,1:end./2,1:end./2) = reshape(...)` is that the result
    # of `reshape(mtx*data(:,:), [M N N])` directly fits into the *first* quarter of the `tdata` volume (M, N, N).

    # Let's re-implement `resamplingOperator` with dense matrices for clarity, as M is likely small enough
    # for these matrices (up to ~1024, M^2 up to ~1M, which is manageable for a sparse operation or direct construction).
    # After `resamplingOperator`'s `for k` loop, `mtx` is `M x M` and `mtxi` is `M x M`.
    # Yes, the loop in `resamplingOperator` *reduces* the `M^2 x M` matrix down to `M x M`.
    # So `mtx` is `M x M`. `data_reshaped` is `M x (N*N)`.
    # `mtx @ data_reshaped` gives `M x (N*N)`. This fits `reshape(... [M N N])`. This makes sense.

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
    # MATLAB: vol  = reshape(mtxi*tvol(:,:),[M N N]);
    # MATLAB: vol  = max(real(vol),0);

    tvol_reshaped = tvol.reshape(M, N * N)
    transformed_tvol = mtxi @ tvol_reshaped
    vol = transformed_tvol.reshape(M, N, N)

    vol = np.maximum(np.real(vol), 0)  # Clamp results and take real part

    toc_end = time.perf_counter()
    time_elapsed = toc_end - tic_start

    print('... done.')
    print(
        f"Reconstructed volume of size {vol.shape[2]} x {vol.shape[1]} x {vol.shape[0]} in {time_elapsed:.4f} seconds")

    # --- Visualization (translated as requested, but you might want to adjust for actual display) ---
    tic_z = np.linspace(0, range_max / 2, vol.shape[0])
    tic_y = np.linspace(-width, width, vol.shape[1])
    tic_x = np.linspace(-width, width, vol.shape[2])

    # Crop and flip reconstructed volume for visualization
    # MATLAB: ind = round(M.*2.*width./(range./2));
    ind = round(M * 2 * width / (range_max / 2))

    # MATLAB: vol = vol(:,:,end:-1:1); (Flips the third dimension)
    vol = vol[:, :, ::-1]  # Pythonic way to reverse the last axis

    # MATLAB: vol = vol((1:ind)+z_offset,:,:);
    # Python uses 0-based indexing. If (1:ind) means indices 1 to ind (inclusive of ind in MATLAB's world for 1:ind),
    # then in Python it's indices `z_offset` to `z_offset + ind`.
    # Note: If `z_offset` is already 0-based from the load, then it's `z_offset : z_offset + ind`.
    # Let's assume z_offset means the starting index.
    vol = vol[z_offset: z_offset + ind, :, :]

    # MATLAB: tic_z = tic_z((1:ind)+z_offset);
    tic_z = tic_z[z_offset: z_offset + ind]

    # View result
    plt.figure(figsize=(10, 4))  # Adjust figure size

    plt.subplot(1, 3, 1)
    # MATLAB: imagesc(tic_x,tic_y,squeeze(max(vol,[],1)));
    # max(vol,[],1) means max along dim 1 (first dim in MATLAB, which is 0 in Python)
    # squeeze removes singleton dimensions
    plt.imshow(np.max(vol, axis=0), cmap='gray',
               extent=[tic_x.min(), tic_x.max(), tic_y.min(), tic_y.max()],
               origin='lower', aspect='auto')  # origin='lower' to match MATLAB's imagesc behavior
    plt.title('Front view')
    plt.xticks(np.linspace(tic_x.min(), tic_x.max(), 3))
    plt.yticks(np.linspace(tic_y.min(), tic_y.max(), 3))
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar()  # Add colorbar for intensity
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(1, 3, 2)
    # MATLAB: imagesc(tic_x,tic_z,squeeze(max(vol,[],2)));
    # max along dim 2 (second dim in MATLAB, which is 1 in Python)
    plt.imshow(np.max(vol, axis=1), cmap='gray',
               extent=[tic_x.min(), tic_x.max(), tic_z.min(), tic_z.max()],
               origin='lower', aspect='auto')
    plt.title('Top view')
    plt.xticks(np.linspace(tic_x.min(), tic_x.max(), 3))
    plt.yticks(np.linspace(tic_z.min(), tic_z.max(), 3))
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(1, 3, 3)
    # MATLAB: imagesc(tic_z,tic_y,squeeze(max(vol,[],3))')
    # max along dim 3 (third dim in MATLAB, which is 2 in Python)
    # ' after squeeze means transpose. Python's imshow expects (rows, cols) where rows are y and cols are x.
    # If the output of max(vol,[],3) is (M, N), and we want (z,y) as (x,y) for imshow,
    # then it should be (y,z) for 'imshow'. So transpose is needed.
    plt.imshow(np.max(vol, axis=2).T, cmap='gray',
               extent=[tic_z.min(), tic_z.max(), tic_y.min(), tic_y.max()],
               origin='lower', aspect='auto')
    plt.title('Side view')
    plt.xticks(np.linspace(tic_z.min(), tic_z.max(), 3))
    plt.yticks(np.linspace(tic_y.min(), tic_y.max(), 3))
    plt.xlabel('z (m)')
    plt.ylabel('y (m)')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def define_psf(U, V, slope):
    """
    Local function to compute NLOS blur kernel.
    Translates MATLAB's definePsf.
    """
    # MATLAB: x = linspace(-1,1,2.*U); y = linspace(-1,1,2.*U); z = linspace(0,2,2.*V);
    x_coords = np.linspace(-1, 1, 2 * U)
    y_coords = np.linspace(-1, 1, 2 * U)
    z_coords = np.linspace(0, 2, 2 * V)

    # MATLAB: [grid_z,grid_y,grid_x] = ndgrid(z,y,x);
    # NumPy's meshgrid default is (X,Y) from (x_coords, y_coords).
    # For ndgrid, which is more like column-major in output for (dim1, dim2, dim3),
    # we need to be careful with the order.
    # MATLAB's ndgrid(A,B,C) gives GRID_A from A, GRID_B from B, GRID_C from C.
    # The output shapes will be (len(A), len(B), len(C)).
    # So ndgrid(z,y,x) means grid_z varies along 1st dim, grid_y along 2nd, grid_x along 3rd.
    grid_z, grid_y, grid_x = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    # After meshgrid(z,y,x, indexing='ij'), shapes are (2V, 2U, 2U) for all three.
    # This matches MATLAB's (Z,Y,X) dimension order.

    # Define PSF
    # MATLAB: psf = abs(((4.*slope).^2).*(grid_x.^2 + grid_y.^2) - grid_z);
    psf = np.abs(((4.0 * slope) ** 2) * (grid_x ** 2 + grid_y ** 2) - grid_z)

    # MATLAB: psf = double(psf == repmat(min(psf,[],1),[2.*V 1 1]));
    # min(psf,[],1) finds minimum along the first dimension (z) for each (y,x) slice.
    # Resulting shape is (1, 2U, 2U).
    min_psf_z = np.min(psf, axis=0, keepdims=True)  # (1, 2U, 2U)
    # repmat to (2V, 1, 1) means repeat min_psf_z along the first axis 2V times.
    min_psf_rep = np.tile(min_psf_z, (2 * V, 1, 1))  # (2V, 2U, 2U)

    # Convert to boolean, then to double (float)
    psf = (psf == min_psf_rep).astype(float)

    # MATLAB: psf = psf./sum(psf(:,U,U));
    # sum(psf(:,U,U)) sums along the first dimension (z) at the center (U,U) slice.
    # MATLAB's U is 1-based index, so U in Python is U-1.
    sum_center_col = np.sum(psf[:, U - 1, U - 1])  # Sum along z at (y=U-1, x=U-1)
    psf = psf / sum_center_col

    # MATLAB: psf = psf./norm(psf(:));
    psf_flat = psf.flatten()
    norm_psf = np.linalg.norm(psf_flat)
    psf = psf / norm_psf

    # MATLAB: psf = circshift(psf,[0 U U]);
    # The shifts are (0 for z, U for y, U for x)
    psf = np.roll(psf, shift=(0, U, U), axis=(0, 1, 2))  # Axis order (z, y, x)
    # Note: np.roll is for circular shifts, matching circshift.
    # It takes a tuple of shifts and a tuple of axes.

    return psf


def resampling_operator(M):
    """
    Local function that defines resampling operators.
    Translates MATLAB's resamplingOperator.
    """
    # MATLAB: mtx = sparse([],[],[],M.^2,M,M.^2);
    # This initializes a sparse matrix of size M^2 x M with 0 non-zeros.
    # We'll construct it directly.

    # MATLAB: x = 1:M.^2;
    # MATLAB: mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
    # Python: indices start from 0. So x will be 0 to M^2 - 1.
    # `sub2ind(size(mtx),x,ceil(sqrt(x)))` means for each linear index `i` from `x`,
    # set the element `mtx[i, ceil(sqrt(i))] = 1`.

    # Create the base matrix (M^2 x M)
    mtx_base = np.zeros((M ** 2, M), dtype=float)

    linear_indices = np.arange(M ** 2)
    col_indices = np.ceil(np.sqrt(linear_indices + 1)) - 1  # +1 for 1-based sqrt, then -1 for 0-based col

    # Handle edge case where col_indices might go out of bounds if M=1 and ceil(sqrt(0)) goes to 0-1=-1
    # or if linear_indices+1 goes past M^2 for M*M
    col_indices = np.clip(col_indices, 0, M - 1).astype(int)

    # Set the values to 1
    mtx_base[linear_indices, col_indices] = 1.0

    # MATLAB: mtx = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx;
    # spdiags(V,D,m,n) creates a sparse matrix with diagonals.
    # 1./sqrt(x)' creates a column vector of (1/sqrt(1), 1/sqrt(2), ...).
    # This forms a diagonal matrix with these values.

    # Diagonal values: 1./sqrt(1:M^2)
    diag_values = 1.0 / np.sqrt(np.arange(1, M ** 2 + 1))

    # Create the diagonal matrix (M^2 x M^2)
    # Using np.diag creates a dense matrix. For large M, this could be an issue.
    # A sparse diagonal matrix would be better, but SciPy's sparse operations can be tricky
    # with direct multiplication if not used carefully.
    # For now, let's assume M is not so large that M^2 x M^2 is prohibitive for memory if directly computed in a product.
    # However, the structure means we can apply the scaling directly.

    # The multiplication `spdiags(...) * mtx` means row `i` of `mtx` is scaled by `1/sqrt(i+1)`.
    # (Since spdiags(V,0,m,n) creates a diagonal matrix whose (i,i) entry is V[i]).
    mtx = mtx_base * diag_values[:, np.newaxis]  # Apply scaling to each row of mtx_base

    mtxi = mtx.T  # Transpose

    # MATLAB: K = log(M)./log(2);
    K_resample = np.log2(M)  # K is already used, so K_resample

    # MATLAB: for k = 1:round(K)
    #             mtx  = 0.5.*(mtx(1:2:end,:)  + mtx(2:2:end,:));
    #             mtxi = 0.5.*(mtxi(:,1:2:end) + mtxi(:,2:2:end));
    # This is a pooling/downsampling operation applied iteratively.
    for k_iter in range(round(K_resample)):
        # For mtx: concatenate odd and even rows then sum.
        # This effectively halves the number of rows.
        # Example: if mtx has 4 rows (0,1,2,3), new mtx has 2 rows:
        # new_mtx[0] = 0.5 * (mtx[0,:] + mtx[1,:])
        # new_mtx[1] = 0.5 * (mtx[2,:] + mtx[3,:])

        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2, :])

        # For mtxi: concatenate odd and even columns then sum.
        # This effectively halves the number of columns.
        mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])

    return mtx, mtxi


# Example usage:
if __name__ == '__main__':
    cnlos_reconstruction(scene=3)