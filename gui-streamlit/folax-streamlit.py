import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import jax.numpy as jnp
import subprocess
import sys
import os
import glob
import math
import time
import plotly.graph_objects as go
import shutil
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.filters import gaussian
import base64
import pyvista as pv
from streamlit.components.v1 import html
import stpyvista



# Constants
IMAGE_WIDTH = 6
IMAGE_HEIGHT = 6
DPI = 400

st.title("FOLAX INTERACTIVE MICROSTRUCTURE SIMULATION")


logo_path = "logo.png"

# Encode image to base64
with open(logo_path, "rb") as f:
    logo_bytes = f.read()
logo_base64 = base64.b64encode(logo_bytes).decode()

# Display logo at bottom-right corner
st.markdown(
    f"""
    <div style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 100px;   /* adjust size */
        height: auto;
        z-index: 9999;">
        <img src="data:image/png;base64,{logo_base64}" style="width:100%; height:auto;">
    </div>
    """,
    unsafe_allow_html=True
)


# --- Initialize session state ---
if "running_solver" not in st.session_state:
    st.session_state.running_solver = False

# --- Tabs ---
tabs = st.tabs(["2D", "3D", "Image Upload"])


def run_solver(cmd, results_folder=None):
    st.session_state.running_solver = True
    try:
        with st.spinner("Running Deep Learning solver... please wait"):
            process = subprocess.run(cmd, capture_output=True, text=True)

        # stdout, stderr
        # st.subheader("Solver Output")
        # st.text_area("stdout", process.stdout, height=200)
        # st.text_area("stderr", process.stderr, height=200)

        if process.returncode != 0:
            st.error("Solver failed. Check stderr.")
        else:
            st.success("Deep Learning Solver finished!")

            # Use the specified folder if provided
            folder = results_folder if results_folder else "./meta_implicit_mechanical_2D"
            result_images = glob.glob(os.path.join(folder, "*.png"))
            cols_per_row = 2
            for i in range(0, len(result_images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, img_path in enumerate(result_images[i:i+cols_per_row]):
                    cols[j].image(img_path, caption=os.path.basename(img_path), use_container_width=True)

            # ZIP for download
            zip_filename = "FOL_results.zip"
            shutil.make_archive("FOL_results", "zip", folder)
            with open(zip_filename, "rb") as f:
                st.download_button(
                    label="Download Results",
                    data=f,
                    file_name=zip_filename,
                    mime="application/zip"
                )
    finally:
        st.session_state.running_solver = False

def save_microstructure_image(fig, filename="microstructure.png", folders=None):
    """Save the figure into one or more folders."""
    if folders is None:
        folders = ["./meta_implicit_mechanical_2D"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        fig.savefig(filepath, dpi=400, bbox_inches="tight")
    return [os.path.join(folder, filename) for folder in folders]
# =========================================================
# Voronoi Microstructure
# =========================================================
with tabs[0]:
    st.subheader("2D Microstructures")
    selection_2d = st.selectbox("Select Microstructure Type", ["Voronoi", "Periodic Voronoi", "Fourier"], key="select_2d")

    if selection_2d == "Voronoi":
        L = 1.0
        N_voronoi = st.slider("Grid Size", 10, 150, 50, key="N_voronoi")
        num_seed_points = st.slider("Number of Seeds", 5, 50, 10, key="seeds_voronoi")
        if st.button("Generate Voronoi"):
            x_coord = np.random.rand(num_seed_points) * L
            y_coord = np.random.rand(num_seed_points) * L
            feature_values = np.random.rand(num_seed_points)

            X, Y = np.meshgrid(np.linspace(0, L, N_voronoi), np.linspace(0, L, N_voronoi))
            seed_points = np.vstack((x_coord, y_coord)).T
            tree = KDTree(seed_points)
            grid_points = np.vstack([X.ravel(), Y.ravel()]).T
            _, regions = tree.query(grid_points)

            K = np.zeros_like(X)
            for i, region in enumerate(regions):
                K.ravel()[i] = feature_values[region]

            coeffs_matrix = np.concatenate([x_coord, y_coord, feature_values]).reshape(1, -1)
            K_matrix = jnp.array(K.reshape(1, -1))
            np.save("K_matrix.npy", np.array(K_matrix))

            st.session_state['voronoi'] = (K, coeffs_matrix, K_matrix)

        if 'voronoi' in st.session_state:
            K, coeffs_matrix, K_matrix = st.session_state['voronoi']
            fig, ax = plt.subplots(figsize=(6,6))
            im = ax.imshow(K, origin='upper', aspect='equal', extent=(0,L,0,L))
            plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
            st.pyplot(fig)
            # Save in both folders for OTF and pretrained models
            save_microstructure_image(fig, filename="microstructure.png", 
                        folders=["./meta_implicit_mechanical_2D", "./mechanical_2d_base_from_ifol_meta"])
            plt.close(fig)

        st.divider()

        st.subheader("Deep Learning & Solver Options")
        epochs = st.slider("Number of Epochs", 100, 5000, 1000, step=100)

        run_fe = st.checkbox("Run Finite Element Solver (compare results)", value=True, key="fe_voronoi")
        # --- Button 1: Run OTF Deep Learning Solver ---
        if st.button("Run OTF Deep Learning Model", disabled=st.session_state.running_solver):
            if 'voronoi' not in st.session_state:
                st.error("Generate Voronoi microstructure first!")
            else:
                K, coeffs_matrix, K_matrix = st.session_state['voronoi']
                np.save("K_matrix.npy", np.array(K_matrix))
                cmd = [
                    sys.executable,
                    "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control.py",
                    f"N={N_voronoi}",
                    f"ifol_num_epochs={epochs}",
                    f"fe_solver={run_fe}",
                    "clean_dir=False"
                ]
                run_solver(cmd, results_folder="./meta_implicit_mechanical_2D")
        st.divider()
        # --- Button 2: Run Pretrained NeoHookean DL Model ---
        if st.button("Run Pre-Trained NeoHookean Deep Learning Model", key="neo_voronoi", disabled=st.session_state.running_solver):
            if 'voronoi' not in st.session_state:
                st.error("Generate Voronoi microstructure first!")
            else:
                K, coeffs_matrix, K_matrix = st.session_state['voronoi']
                np.save("K_matrix.npy", np.array(K_matrix))
                cmd = [
                    sys.executable,
                    "run_pretrained_neohookean.py",
                    f"N={N_voronoi}"
                ]
                run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")

    # =========================================================
    # Periodic Voronoi
    # =========================================================
    elif selection_2d == "Periodic Voronoi":
        L = 1.0
        N_periodic = st.slider("Grid Size", 10, 150, 50, key="N_p")
        num_seed_points = st.slider("Number of Seeds", 5, 50, 10, key="seeds_p")

        if st.button("Generate Periodic Voronoi", key="run_periodic"):
            x_coord = np.random.rand(num_seed_points) * L
            y_coord = np.random.rand(num_seed_points) * L
            feature_values = np.random.rand(num_seed_points)

            X, Y = np.meshgrid(np.linspace(0,L,N_periodic), np.linspace(0,L,N_periodic))
            K = np.zeros_like(X)
            seed_points = np.vstack((x_coord, y_coord)).T

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = np.array([X[i,j], Y[i,j]])
                    distances = []
                    for dx in [-L,0,L]:
                        for dy in [-L,0,L]:
                            for sx, sy in seed_points:
                                distances.append(np.sqrt((point[0]-(sx+dx))**2 + (point[1]-(sy+dy))**2))
                    K[i,j] = feature_values[np.argmin(distances)%len(feature_values)]

            st.session_state['periodic_2d'] = K

        if 'periodic_2d' in st.session_state:
            K = st.session_state['periodic_2d']
            fig, ax = plt.subplots(figsize=(6,6))
            im = ax.imshow(K, extent=(0,L,0,L), origin='upper', aspect='equal', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
            st.pyplot(fig)
            save_microstructure_image(fig, filename="microstructure.png", 
                        folders=["./meta_implicit_mechanical_2D", "./mechanical_2d_base_from_ifol_meta"])

            plt.close(fig)

            st.divider()
            st.subheader("Deep Learning & Solver Options")
            epochs_periodic = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_periodic")

            run_fe = st.checkbox("Run Finite Element Solver (compare results)", value=True, key="fe_periodic")

            if st.button("Run OTF Deep Learning Model", key="fol_periodic", disabled=st.session_state.running_solver):
                K_matrix = np.array(K.reshape(1, -1))
                np.save("K_matrix.npy", K_matrix)
                cmd = [
                    sys.executable,
                    "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control.py",
                    f"N={N_periodic}",
                    f"ifol_num_epochs={epochs_periodic}",
                    f"fe_solver={run_fe}",
                    "clean_dir=False"
                ]
                run_solver(cmd)
            st.divider()
            # --- Button 2: Run Pretrained NeoHookean DL Model ---
            if st.button("Run Pre-Trained NeoHookean Deep Learning Model", key="neo_periodic", disabled=st.session_state.running_solver):
                K_matrix = np.array(K.reshape(1, -1))
                np.save("K_matrix.npy", K_matrix)
                cmd = [
                    sys.executable,
                    "run_pretrained_neohookean.py",
                    f"N={N_periodic}"
                ]
                run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")
    # =========================================================
    # Fourier
    # =========================================================
    elif selection_2d == "Fourier":
        L = 1.0
        N_fourier = st.slider("Grid Size", 10, 100, 50, key="N_f")
        x_freqs = st.text_input("x Frequencies (comma-separated)", "1,2", key="x_freqs")
        y_freqs = st.text_input("y Frequencies (comma-separated)", "1,2", key="y_freqs")
        K_max = st.number_input("K_max", 1.0)
        K_min = st.number_input("K_min", 0.1)
        beta = st.number_input("Beta", 1.0)

        try:
            x_freqs_list = list(map(float, x_freqs.split(',')))
            y_freqs_list = list(map(float, y_freqs.split(',')))
        except:
            x_freqs_list, y_freqs_list = [], []

        coeffs = []
        if x_freqs_list and y_freqs_list and len(x_freqs_list) == len(y_freqs_list):
            for i in range(len(x_freqs_list)+1):
                coeffs.append(st.slider(f"Coefficient {i}", -5.0, 5.0, 0.0, 0.1, key=f"coeff_{i}"))

        if st.button("Generate Fourier Field", key="run_fourier"):
            x = np.linspace(0, L, N_fourier)
            y = np.linspace(0, L, N_fourier)
            X, Y = np.meshgrid(x, y)

            K = coeffs[0] / 2.0
            for i, (xf, yf) in enumerate(zip(x_freqs_list, y_freqs_list)):
                K += coeffs[i+1] * np.cos(2*np.pi*xf*X/L) * np.cos(2*np.pi*yf*Y/L)

            sigmoid = lambda x: 1/(1+np.exp(-x))
            K_mapped = (K_max-K_min) * sigmoid(beta * (K - 0.5)) + K_min

            st.session_state['fourier_2d'] = K_mapped
            np.save("K_matrix.npy", K_mapped.reshape(1, -1))

        if 'fourier_2d' in st.session_state:
            K = st.session_state['fourier_2d']
            fig, ax = plt.subplots(figsize=(IMAGE_WIDTH, IMAGE_HEIGHT), dpi=DPI)
            im = ax.imshow(K, extent=(0,L,0,L), origin='upper', aspect='equal', vmin=0,vmax=1)
            plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
            st.pyplot(fig)
            save_microstructure_image(fig, filename="microstructure.png", 
                        folders=["./meta_implicit_mechanical_2D", "./mechanical_2d_base_from_ifol_meta"])
            plt.close(fig)

            epochs_fourier = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_fourier")

            run_fe = st.checkbox("Run Finite Element Solver (compare results)", value=True, key="fe_fourier_2d")
            if st.button("Run OTF Deep Learning Solver", key="fol_fourier", disabled=st.session_state.running_solver):
                if 'fourier_2d' not in st.session_state:
                    st.error("Generate the Fourier field first!")
                else:
                    cmd = [
                        sys.executable,
                        "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control.py",
                        f"N={N_fourier}",
                        f"ifol_num_epochs={epochs_fourier}",
                        f"fe_solver={run_fe}",
                        "clean_dir=False"
                    ]
                    run_solver(cmd)
            st.divider()
            # --- Button 2: Run Pretrained NeoHookean DL Model ---
            if st.button("Run Pre-Trained NeoHookean Deep Learning Model", key="neo_fourier", disabled=st.session_state.running_solver):
                if 'fourier_2d' not in st.session_state:
                    st.error("Generate the Fourier field first!")
                else:
                    cmd = [
                        sys.executable,
                        "run_pretrained_neohookean.py",
                        f"N={N_fourier}"
                    ]
                    run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")
# =========================================================
# 3D TAB
# =========================================================
with tabs[1]:
    st.subheader("3D Microstructures")
    selection_3d = st.selectbox("Select 3D Microstructure Type", ["Fourier 3D"], key="select_3d")

    if selection_3d == "Fourier 3D":
        L3D = 1.0
        N3D = st.slider("Grid Size (N, 3D)", 5, 50, 20, key="N3D")
        x_freqs_3d = st.text_input("x Frequencies (comma-separated)", "1,2", key="x3d")
        y_freqs_3d = st.text_input("y Frequencies (comma-separated)", "1,2", key="y3d")
        z_freqs_3d = st.text_input("z Frequencies (comma-separated)", "1,2", key="z3d")
        K_max3d = st.number_input("K_max (3D)", 1.0)
        K_min3d = st.number_input("K_min (3D)", 0.0)
        beta3d = st.number_input("Beta", 0.1, 10.0, 1.0)
        shift3d = st.number_input("Shift", -5.0, 5.0, 0.0)

        try:
            xf_list = list(map(float, x_freqs_3d.split(',')))
            yf_list = list(map(float, y_freqs_3d.split(',')))
            zf_list = list(map(float, z_freqs_3d.split(',')))
        except:
            xf_list = yf_list = zf_list = []

        coeffs3d = []
        if xf_list and yf_list and zf_list and len(xf_list) == len(yf_list) == len(zf_list):
            for i in range(len(xf_list)+1):
                coeffs3d.append(st.slider(f"Coefficient {i} (3D)", -5.0, 5.0, 0.0, 0.1, key=f"coeff3d_{i}"))

        if st.button("Generate 3D Fourier", key="run_3d"):
            x = np.linspace(0, L3D, N3D)
            y = np.linspace(0, L3D, N3D)
            z = np.linspace(0, L3D, N3D)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

            K3D = coeffs3d[0] / 2.0
            for i, (xf, yf, zf) in enumerate(zip(xf_list, yf_list, zf_list)):
                K3D += coeffs3d[i+1] * np.cos(2*np.pi*xf*X/L3D) * np.cos(2*np.pi*yf*Y/L3D) * np.cos(2*np.pi*zf*Z/L3D)

            sigmoid = lambda x: 1/(1+np.exp(-x))
            K3D_mapped = (K_max3d-K_min3d) * sigmoid(beta3d*(K3D-shift3d)) + K_min3d
            st.session_state['fourier_3d'] = (X, Y, Z, K3D_mapped)

        if 'fourier_3d' in st.session_state:
            X, Y, Z, K3D_mapped = st.session_state['fourier_3d']

            fig = go.Figure(data=go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=K3D_mapped.flatten(),
                isomin=0, isomax=1,
                surface_count=25,
                colorscale='Viridis',
                colorbar=dict(title="Young's Modulus (E)")
            ))
            fig.update_layout(
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                title="3D Fourier Microstructure"
            )
            st.plotly_chart(fig)

            fe_3d = st.checkbox("Run Finite Element Solver (compare results)", value=True, key="fe_3d")
            epochs_3d = st.slider("Number of Epochs", 100, 2000, 1000, step=100, key="epochs_3d")


            if st.button("Run OTF Deep Learning Solver", key="fol_3d"):
                fol_result_3d = run_fol_async(K3D_mapped, fol_num_epochs=epochs_3d, display_plot=True, is_3d=True)
                st.session_state['fourier_3d_fol_result'] = fol_result_3d



with tabs[2]:  # Image Upload tab
    st.subheader("Upload Microstructure Image or VTK")

    uploaded_file = st.file_uploader("Upload an image or VTK file", type=["png", "jpg", "jpeg", "vtk"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # =======================
        # VTK Upload Path 
        # =======================
        if file_ext == "vtk":

            vtk_path = f"temp_{uploaded_file.name}"
            with open(vtk_path, "wb") as f:
                f.write(uploaded_file.read())

            mesh = pv.read(vtk_path)

            # ---- scalar field ----
            scalar_name = list(mesh.point_data.keys())[0]
            scalar = mesh.point_data[scalar_name]
            scalar_norm = (scalar - scalar.min()) / (scalar.max() - scalar.min())

            # ---- bounds ----
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

            # =============================================================
            # NORMALIZATION
            # =============================================================

            # ImageData if mesh has uniform spacing
            if hasattr(mesh, "spacing") and mesh.spacing is not None:

                dims = mesh.dimensions  # (nx, ny, nz)
                nx, ny, nz = dims[0] - 1, dims[1] - 1, dims[2] - 1

                # New spacing so final domain is [0,1]^3
                new_spacing = (
                    1.0 / nx if nx > 0 else 1,
                    1.0 / ny if ny > 0 else 1,
                    1.0 / nz if nz > 0 else 1,
                )

                new_origin = (0.0, 0.0, 0.0)

                # Create scaled VTK Image
                scaled_mesh = pv.ImageData(dimensions=dims, spacing=new_spacing, origin=new_origin)
                scaled_mesh.point_data[scalar_name] = scalar_norm

            else:
                # ---- PolyData / Unstructured ----
                pts = mesh.points.copy()
                pts[:, 0] = (pts[:, 0] - xmin) / (xmax - xmin)
                pts[:, 1] = (pts[:, 1] - ymin) / (ymax - ymin)
                pts[:, 2] = (pts[:, 2] - zmin) / (zmax - zmin)
                mesh.points = pts
                mesh.point_data[scalar_name] = scalar_norm
                scaled_mesh = mesh

            # =============================================================
            # 3D Render
            # =============================================================
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(scaled_mesh, scalars=scalar_name, cmap="viridis", show_edges=False)
            plotter.view_isometric()
            screenshot = plotter.screenshot()
            st.image(screenshot, caption="Normalized 3D Visualization")

            # =============================================================
            # NORMALIZED SLICE
            # =============================================================
            y_norm = st.slider("Y Slice (normalized)", 0.0, 1.0, 0.5)

            # convert normalized y in original mesh coordinate system
            y_real = ymin + y_norm * (ymax - ymin)
            slice_plane = mesh.slice(normal="y", origin=(0, y_real, 0))

            pts_slice = slice_plane.points.copy()

            # Normalize slice point coordinates
            pts_slice[:, 0] = (pts_slice[:, 0] - xmin) / (xmax - xmin)
            pts_slice[:, 1] = (pts_slice[:, 1] - ymin) / (ymax - ymin)
            pts_slice[:, 2] = (pts_slice[:, 2] - zmin) / (zmax - zmin)

            # ---------- NEW: Normalize slice scalar values ----------
            slice_values = slice_plane.point_data[scalar_name]
            slice_values_norm = (slice_values - slice_values.min()) / (slice_values.max() - slice_values.min())

            x = pts_slice[:, 0]
            z = pts_slice[:, 2]

            # =============================================================
            # 2D Slice Plot (normalized)
            # =============================================================
            fig, ax = plt.subplots(figsize=(6,5))
            sc = ax.scatter(x, z, c=slice_values_norm, cmap="viridis", s=10)
            plt.colorbar(sc, ax=ax, label=f"Normalized {scalar_name}")
            ax.set_xlabel("Normalized X")
            ax.set_ylabel("Normalized Z")
            ax.set_title(f"2D Slice at normalized y = {y_norm:.2f}")
            ax.axis("equal")
            st.pyplot(fig)

            # =============================================================
            # Convert Slice to K-matrix
            # =============================================================

            st.subheader("Set Resolution of Microstructure Field")

            # User chooses output grid size
            N_slice = st.slider("Output Resolution (N × N)", 10, 200, 50)

            # Create regular grid in the X–Z plane
            grid_x = np.linspace(0, 1, N_slice)
            grid_z = np.linspace(0, 1, N_slice)
            GX, GZ = np.meshgrid(grid_x, grid_z)

            # Interpolate slice scalar values onto grid
            from scipy.interpolate import griddata

            K_slice = griddata(
                points=np.vstack([x, z]).T,
                values=slice_values_norm,
                xi=(GX, GZ),
                method="linear",
                fill_value=0.0
            )

            # ---- FIX: flip vertically so orientation matches the VTK slice ----
            K_slice = np.flipud(K_slice)

            # Store microstructure
            st.session_state["vtk_microstructure"] = K_slice

            # Show preview (note corrected title)
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            im2 = ax2.imshow(K_slice, cmap="viridis", origin='upper')
            plt.colorbar(im2, ax=ax2, label="Normalized Scalar")
            ax2.set_title("Normalized Microstructure")
            st.pyplot(fig2)

            # Save like other microstructure generators
            np.save("K_matrix.npy", K_slice.reshape(1, -1))
            save_microstructure_image(fig2, filename="microstructure.png",
                                    folders=["./meta_implicit_mechanical_2D",
                                            "./mechanical_2d_base_from_ifol_meta"])


            # =============================================================
            # Add OTF & Pretrained Solver Buttons
            # =============================================================

            st.subheader("Deep Learning & Solver Options")

            epochs_vtk = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_vtk")
            run_fe_vtk = st.checkbox("Run Finite Element Solver (compare results)", value=True, key="fe_vtk")

            # ---- OTF Solver ----
            if st.button("Run OTF Deep Learning Solver (VTK)", disabled=st.session_state.running_solver):
                if "vtk_microstructure" not in st.session_state:
                    st.error("Generate the microstructure slice first!")
                else:
                    np.save("K_matrix.npy", K_slice.reshape(1, -1))
                    cmd = [
                        sys.executable,
                        "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control.py",
                        f"N={N_slice}",
                        f"ifol_num_epochs={epochs_vtk}",
                        f"fe_solver={run_fe_vtk}",
                        "clean_dir=False"
                    ]
                    run_solver(cmd, results_folder="./meta_implicit_mechanical_2D")

            st.divider()

            # ---- Pretrained NeoHookean Model ----
            if st.button("Run Pre-Trained NeoHookean Model (VTK)", disabled=st.session_state.running_solver):
                if "vtk_microstructure" not in st.session_state:
                    st.error("Generate the microstructure slice first!")
                else:
                    np.save("K_matrix.npy", K_slice.reshape(1, -1))
                    cmd = [
                        sys.executable,
                        "run_pretrained_neohookean.py",
                        f"N={N_slice}"
                    ]
                    run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")




        # =======================
        # Image Upload Path
        # =======================
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # --- Select Segmentation Method ---
            method = st.selectbox("Select Segmentation Method", ["KMeans", "Mask R-CNN", "Unet"])

            N_img = st.slider("Grid Size (reduce image to N×N)", 10, 200, 50)

            if method == "KMeans":
                image_resized = image.resize((N_img, N_img))
                img_array = np.array(image_resized)
                flat_pixels = img_array.reshape(-1, 3)

                n_phases = st.slider("Number of Phases (clusters)", 2, 10, 3)
                kmeans = KMeans(n_clusters=n_phases, n_init=10, random_state=0)
                labels = kmeans.fit_predict(flat_pixels)

                feature_values = np.linspace(0.1, 1.0, n_phases)
                K_matrix = feature_values[labels].reshape(N_img, N_img)

                np.save("K_matrix.npy", K_matrix.reshape(1, -1))
                st.session_state["uploaded_microstructure"] = K_matrix

                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(K_matrix, cmap="viridis", origin="upper")
                plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
                save_microstructure_image(fig, filename="microstructure.png",
                                          folders=["./meta_implicit_mechanical_2D",
                                                   "./mechanical_2d_base_from_ifol_meta"])
                st.pyplot(fig)

            elif method == "Mask R-CNN":
                st.warning("Mask R-CNN segmentation not implemented yet. Coming soon!")

            elif method == "Unet":
                st.warning("Unet segmentation not implemented yet. Coming soon!")

            st.divider()
            st.subheader("Deep Learning & Solver Options")

            # --- Option to run solver ---
            epochs_upload = st.slider("Number of Epochs", 100, 5000, 2000, step=100, key="epochs_imageupload")
            run_fe = st.checkbox("Run Finite Element Solver (compare results)", value=True, key="fe_imageupload")
            if st.button("Run OTF Deep Learning Solver", key="solver_imageupload", disabled=st.session_state.running_solver):
                cmd = [
                    sys.executable,
                    "meta_alpha_implicit_pr_lr_mechanical_2D_identity_control2.py",
                    f"N={N_img}",
                    f"ifol_num_epochs={epochs_upload}",
                    f"fe_solver={run_fe}",
                    "clean_dir=False"
                ]
                run_solver(cmd)

            st.divider()
            # --- Button 2: Run Pretrained NeoHookean DL Model ---
            if st.button("Run Pre Trained NeoHookean Deep Learning Model", key="neo_imageupload", disabled=st.session_state.running_solver):
                cmd = [
                    sys.executable,
                    "run_pretrained_neohookean.py",
                    f"N={N_img}"
                ]
                run_solver(cmd, results_folder="./mechanical_2d_base_from_ifol_meta")