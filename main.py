import os
import cv2
import numpy as np
from skimage.exposure import exposure
from skimage.feature import canny, hog, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from athec.athec import saliency, misc, edge


def load_images(image_folder):
    """Load images from the specified folder."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'png', 'jpeg'))]
    images = [cv2.imread(os.path.join(image_folder, file)) for file in image_files]
    return images


def extract_features_and_save(images, saliency_folder, output_folder):
    """Extract edges, HOG, entropy, saliency, and GLCM from each image."""
    features = []
    for i, img in enumerate(images):
        # Convert to grayscale
        gray_img = rgb2gray(img)
        gray_img_uint8 = (gray_img * 255).astype(np.uint8)

        # Create folder for output images
        img_output_folder = os.path.join(output_folder, f'image_{i}')
        os.makedirs(img_output_folder, exist_ok=True)

        # 1. Edge Features
        edge_img_path = os.path.join(img_output_folder, "edges.png")
        edges = edge.tf_edge_canny(img, save_path=edge_img_path, thresholds=(50, 150), otsu_ratio=None,
                                   gaussian_blur_kernel=None)

        # 2. HOG Features
        hog_features, hog_img = hog(
            gray_img,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            block_norm='L2-Hys'
        )

        inverted_hog_img = cv2.bitwise_not(hog_img)
        hog_image_rescaled = exposure.rescale_intensity(inverted_hog_img, in_range=(0, 10))
        hog_img_path = os.path.join(img_output_folder, "hog.png")
        cv2.imwrite(hog_img_path, hog_image_rescaled)

        # 3. Entropy
        entropy_img = entropy(gray_img_uint8, disk(5))
        entropy_img_path = os.path.join(img_output_folder, "entropy.png")
        cv2.imwrite(entropy_img_path, (entropy_img * 255).astype(np.uint8))  # Save entropy image

        # 4. Saliency
        img_path = os.path.join(saliency_folder, f"temp_{i}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Save image as RGB for saliency

        # Compute saliency maps using athec
        saliency_spectral = saliency.tf_saliency_spectral_residual(img_path)
        saliency_fine = saliency.tf_saliency_fine_grained(img_path)

        # Convert saliency maps to numpy arrays and normalize
        saliency_spectral = np.array(saliency_spectral, dtype=np.float32)
        saliency_fine = np.array(saliency_fine, dtype=np.float32)

        # Optionally binarize saliency maps
        saliency_spectral_bin = misc.tf_binary(saliency_spectral, threshold=60)
        saliency_fine_bin = misc.tf_binary(saliency_fine, threshold=60)

        # Calculate total saliency (visual complexity) using athec
        total_saliency = saliency.attr_complexity_saliency(saliency_spectral,
                                                           threshold=0.7,
                                                           nblock=10,
                                                           return_block=False)

        # Save total saliency image
        total_saliency_image = np.full((100, 100), total_saliency['saliency_total'] * 255, dtype=np.uint8)
        total_saliency_path = os.path.join(img_output_folder, "total_saliency.png")
        cv2.imwrite(total_saliency_path, total_saliency_image)

        # 5. GLCM (Gray-Level Co-occurrence Matrix) Features
        distances = [1]  # You can adjust this
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135° in radians

        glcm = graycomatrix(gray_img_uint8, distances=distances, angles=angles, symmetric=True, normed=True)

        # Extract texture features from the GLCM
        contrast = graycoprops(glcm, 'contrast').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()

        # Store features for each image
        features.append({
            'edges': edges.astype(np.float32),
            'hog': hog_features,
            'entropy': entropy_img,
            'saliency_total': total_saliency['saliency_total'],
            'glcm_contrast': contrast,
            'glcm_correlation': correlation,
            'glcm_energy': energy,
            'glcm_homogeneity': homogeneity
        })

    return features


def fit_gaussian_models_and_plot(features, output_folder):
    """Fit a Gaussian model to each feature and plot the distributions individually."""
    gaussian_models = {}

    for feature_name in features[0].keys():
        # Flatten feature data and combine from all images
        feature_data = np.concatenate([f[feature_name].flatten() for f in features])

        # Reshape data to ensure it is 2D
        feature_data = feature_data.reshape(-1, 1)

        # Ensure there is enough data to fit the GMM
        if feature_data.shape[0] < 2:
            print(f"Not enough data to fit GMM for {feature_name}.")
            continue

        # Fit Gaussian model (using a single Gaussian component)
        gmm = GaussianMixture(n_components=1, covariance_type='full')
        gmm.fit(feature_data)

        # Store the Gaussian model
        gaussian_models[feature_name] = gmm

        # Create a new figure for each feature
        plt.figure(figsize=(10, 6))

        # Plot histogram of the feature data
        plt.hist(feature_data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black', label='Feature Data')

        # Plot Gaussian fit
        x = np.linspace(np.min(feature_data), np.max(feature_data), 1000).reshape(-1, 1)
        logprob = gmm.score_samples(x)
        pdf = np.exp(logprob)
        plt.plot(x, pdf, '-k', label='Gaussian Fit')

        # Add titles and labels
        plt.title(f'Gaussian Fit for {feature_name}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        # Save the plot
        plot_path = os.path.join(output_folder, f'{feature_name}_gaussian_fit.png')
        plt.savefig(plot_path)
        plt.close()

    return gaussian_models


def process_images_and_plot(image_folder, saliency_folder, output_folder):
    """Load images, extract features, fit Gaussian models, and plot the distributions."""
    # Step 1: Load images
    images = load_images(image_folder)

    # Step 2: Extract features from images and save processed images
    features = extract_features_and_save(images, saliency_folder, output_folder)

    # Step 3: Fit Gaussian models to the features and plot the distributions
    gaussian_models = fit_gaussian_models_and_plot(features, output_folder)

    return gaussian_models


# Paths to the folders
image_folder = './normal_type'
saliency_folder = './saliency_temp'
output_folder = './processed_images'

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Process the images, extract features, fit Gaussian models, and plot the distributions
gaussian_models = process_images_and_plot(image_folder, saliency_folder, output_folder)
