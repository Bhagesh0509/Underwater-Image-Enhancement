import os
import time
import base64
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import skimage.metrics
from torch.autograd import Variable
from torchvision.utils import save_image
from net.Ushape_Trans import Generator  # Make sure this import path is correct

# --------- Streamlit Page Config ---------
st.set_page_config(page_title="🌊 Underwater Image Restoration", layout="centered")

# --------- Background Image ---------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image_path = "C:\\Users\\Hi\\Underwater_Image\\new final\\U-shape_Transformer_for_Underwater_Image_Enhancement-main\\U-shape_Transformer_for_Underwater_Image_Enhancement-main\\pngtree-underwater-sunken-city-landscape-with-seaweeds-and-sea-animals-aquatic-life-png-image_.png"
bg_base64 = get_base64_image(bg_image_path)

st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)),
                    url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .main {{ background-color: rgba(0,0,0,0.3); padding: 2rem; border-radius: 15px; }}
    img {{ border-radius: 12px; }}
    .stButton > button {{ background-color: #00b4d8; color: white; padding: 10px 20px; border-radius: 10px; }}
    </style>
""", unsafe_allow_html=True)



# --------- Title ---------
st.markdown("<h1 style='text-align:center;'>🌊 Underwater Image Restoration</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an underwater image to enhance and restore it using a deep learning model.</p>", unsafe_allow_html=True)
st.markdown("---")

# --------- Placeholder Metrics ---------
def calculate_uicqe(image):
    return float(np.mean(image) / 255.0)

def calculate_uiqm(image):
    return float(np.std(image) / 64.0)

# --------- Load Model ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = Generator().to(device)
    model.load_state_dict(torch.load(
        r"C:\Users\Hi\Underwater_Image\new final\saved_models\saved_models\G\generator_795.pth",  # Replace with actual path
        map_location=device
    ))
    model.eval()
    return model

generator = load_model()

# --------- File Upload ---------
uploaded_file = st.file_uploader("📷 Upload your underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.subheader("🔍 Uploaded Image Preview")
    st.image(img, caption="📥 Original Image", use_column_width=True)

    imgx = img.astype('float32') / 255.0
    imgx = torch.from_numpy(imgx).permute(2, 0, 1).unsqueeze(0).to(device)

    # --------- Inference ---------
    start_time = time.time()
    with torch.no_grad():
        output = generator(imgx)
        out = output[3].data
    inference_time = time.time() - start_time

    out_img = out.cpu().numpy().squeeze(0).transpose(1, 2, 0)
    out_img = np.clip(out_img, 0, 1)

    st.markdown("---")
    st.subheader("✨ Restored Output")
    st.image(out_img, caption="🧼 Cleaned & Enhanced Image", use_column_width=True)

    # --------- Metrics ---------
    psnr_val = skimage.metrics.peak_signal_noise_ratio(img / 255.0, out_img)
    ssim_val = skimage.metrics.structural_similarity(img / 255.0, out_img, data_range=1.0, channel_axis=2, win_size=7)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].bar(["Enhanced"], [psnr_val], color=["green"])
    axes[0].set_title("PSNR (vs original)")

    axes[1].bar(["Enhanced"], [ssim_val], color=["blue"])
    axes[1].set_title("SSIM (vs original)")

    st.pyplot(fig)

    # --------- UCIQE & UIQM ---------
    uicqe_b = calculate_uicqe(img)
    uicqe_a = calculate_uicqe((out_img * 255).astype(np.uint8))
    uiqm_b = calculate_uiqm(img)
    uiqm_a = calculate_uiqm((out_img * 255).astype(np.uint8))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].bar(["Orig", "Enh"], [uicqe_b, uicqe_a], color=["red", "green"])
    ax[0].set_title("UCIQE – Underwater Color Image Quality Evaluation Score")
    ax[1].bar(["Orig", "Enh"], [uiqm_b, uiqm_a], color=["orange", "purple"])
    ax[1].set_title("UIQM – Underwater Image Quality Measure")
    st.pyplot(fig)

    # --------- RGB Histogram ---------
    fig, axes = plt.subplots(1, 3, figsize=(30, 20))
    colors = ["r", "g", "b"]
    for i, c in enumerate(colors):
        axes[i].hist(img[:, :, i].ravel(), bins=256, color=c, alpha=0.5, label="Orig")
        axes[i].hist((out_img[:, :, i] * 255).astype(np.uint8).ravel(),
                     bins=256, color="black", alpha=0.5, label="Enh")
        axes[i].set_title(f"{c.upper()} Channel Histogram")
        axes[i].legend()
    st.pyplot(fig)

    from skimage import color
    from mpl_toolkits.mplot3d import Axes3D

    # -------- ΔE Color Difference Map --------
    lab_orig = color.rgb2lab(img / 255.0)
    lab_enh = color.rgb2lab(out_img)
    delta_e = np.linalg.norm(lab_orig - lab_enh, axis=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(delta_e, cmap='plasma')
    plt.colorbar(im, ax=ax)
    ax.set_title("ΔE Color Difference Map")
    ax.axis('off')
    st.pyplot(fig)

    # -------- Frequency Domain Analysis (FFT) --------
    def fft_magnitude(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        return magnitude_spectrum

    fft_orig = fft_magnitude(img / 255.0)
    fft_enh = fft_magnitude(out_img)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(fft_orig, cmap='gray')
    axs[0].set_title("FFT of Original")
    axs[0].axis('off')

    axs[1].imshow(fft_enh, cmap='gray')
    axs[1].set_title("FFT of Enhanced")
    axs[1].axis('off')
    st.pyplot(fig)

    # -------- Zoomed-In Patch Comparison --------
    def show_zoom_patch(orig, enh, x, y, size=64):
        patch_orig = orig[y:y+size, x:x+size]
        patch_enh = enh[y:y+size, x:x+size]
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(patch_orig)
        axs[0].set_title("Original Patch")
        axs[0].axis('off')
        axs[1].imshow(patch_enh)
        axs[1].set_title("Enhanced Patch")
        axs[1].axis('off')
        st.pyplot(fig)

    show_zoom_patch(img / 255.0, out_img, x=100, y=100)

    # -------- 3D LAB Color Histogram --------
    def plot_3d_lab_hist(img, title):
        lab = color.rgb2lab(img)
        L, A, B = lab[:, :, 0].flatten(), lab[:, :, 1].flatten(), lab[:, :, 2].flatten()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(L, A, B, c=img.reshape(-1, 3), s=1)
        ax.set_xlabel('L')
        ax.set_ylabel('a')
        ax.set_zlabel('b')
        ax.set_title(title)
        st.pyplot(fig)

    plot_3d_lab_hist(img / 255.0, "3D LAB Histogram - Original")
    plot_3d_lab_hist(out_img, "3D LAB Histogram - Enhanced")
        

    # --------- Inference Time ---------
    st.markdown(f"<h4 style='color:red; font-weight:bold;'>⏱️ Inference Time: {inference_time:.2f} seconds</h4>", unsafe_allow_html=True)


    # --------- Save Button ---------
    if st.button("💾 Save Restored Image"):
        output_dir = "./test/output/"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'restored_image.jpg')
        save_image(out, save_path, normalize=True)
        st.success(f"✅ Image saved at: `{save_path}`")
