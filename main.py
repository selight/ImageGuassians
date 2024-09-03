import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from skimage.exposure import exposure
from skimage.feature import canny, hog, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk
from athec.athec import saliency, misc, edge


def load_images(image_folder):
    """Load images from the specified folder."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'png', 'jpeg'))]
    images = [cv2.imread(os.path.join(image_folder, file)) for file in image_files]
    return images, image_files


def extract_features_and_save(images, image_files, saliency_folder, output_folder):
    """Extract edges, HOG, entropy, saliency, and GLCM from each image."""
    features = []
    for i, (img, file_name) in enumerate(zip(images, image_files)):
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
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135° in radians

        glcm = graycomatrix(gray_img_uint8, distances=distances, angles=angles, symmetric=True, normed=True)

        # Extract texture features from the GLCM
        contrast = graycoprops(glcm, 'contrast').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()

        # Store features for each image
        feature_dict = {'file_name': file_name, 'entropy_mean': np.mean(entropy_img), 'edges_mean': np.mean(edges),
                        'saliency_total': total_saliency['saliency_total'], 'glcm_contrast_mean': np.mean(contrast),
                        'glcm_correlation_mean': np.mean(correlation), 'glcm_energy_mean': np.mean(energy),
                        'glcm_homogeneity_mean': np.mean(homogeneity), 'hog_mean': np.mean(hog_features)}

        # For HOG, store the mean value of HOG features

        features.append(feature_dict)

    return features


def generate_synthetic_data(features, use_varying_threshold=True, uniform_threshold=20):
    """Generate synthetic data based on the extracted features with varying or uniform perturbations."""
    synthetic_features = []

    # Define different variance percentages for each feature
    variance_dict = {
        'entropy_mean': 10,  # Low variance for slight texture changes
        'edges_mean': 30,  # Higher variance to account for new edges or objects
        'saliency_total': 15,  # Moderate variance to detect prominent changes
        'glcm_contrast_mean': 20,  # Moderate variance for new textures or shadows
        'glcm_correlation_mean': 12,  # Low variance; minor impact from new objects
        'glcm_energy_mean': 18,  # Moderate variance; changes in uniformity
        'glcm_homogeneity_mean': 25,  # Higher variance for significant texture changes
        'hog_mean': 25  # Higher variance for detecting shape and orientation changes
    }
    for feature in features:
        synthetic_feature = {'file_name': feature['file_name']}  # Keep the original file name

        for key, value in feature.items():
            if key == 'file_name':
                continue  # Skip the file_name field

            # Determine the variance percentage to use
            if use_varying_threshold:
                variance_percentage = variance_dict.get(key, 20)  # Use varying threshold from variance_dict
            else:
                variance_percentage = uniform_threshold  # Use uniform threshold for all features

            # Generate synthetic values with the chosen variance
            if isinstance(value, np.ndarray):  # Handle numpy arrays
                synthetic_value = value + (variance_percentage / 100.0) * np.abs(value) * np.random.choice([-1, 1])
            elif isinstance(value, (int, float)):  # Handle scalar values
                synthetic_value = value + (variance_percentage / 100.0) * np.abs(value) * np.random.choice([-1, 1])
            else:
                synthetic_value = value  # Keep other types unchanged

            synthetic_feature[key] = synthetic_value

        synthetic_features.append(synthetic_feature)

    return synthetic_features


def plot_gaussian_distributions_and_save(df, output_folder):
    """Plot Gaussian distributions for each feature in the DataFrame and save them."""
    # Filter out the 'file_name' column
    feature_columns = df.select_dtypes(include=[np.number]).columns

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for feature in feature_columns:
        plt.figure(figsize=(10, 6))

        # Plot the Gaussian distribution
        sns.histplot(df[feature], kde=True, stat="density", linewidth=0)

        # Fit a normal distribution to the data
        mu, std = norm.fit(df[feature])

        # Plot the Gaussian curve
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)

        # Set titles and labels
        plt.title(f"Gaussian Distribution for {feature}")
        plt.xlabel(f"{feature}")
        plt.ylabel("Density")

        # Save the plot
        plot_path = os.path.join(output_folder, f'{feature}_gaussian_distribution.png')
        plt.savefig(plot_path)
        plt.close()

        print(f'Saved Gaussian distribution plot for {feature} to {plot_path}')


def process_images_and_save(image_folder, saliency_folder, output_folder):
    """Load images, extract features, and save to CSV."""
    # Step 1: Load images
    images, image_files = load_images(image_folder)

    # Step 2: Extract features from images and save processed images
    features = extract_features_and_save(images, image_files, saliency_folder, output_folder)
    variance_percentage = 20
    synthetic_features = generate_synthetic_data(features)
    synthetic_features_uniform = generate_synthetic_data(features, False, 20)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(features)
    synthetic_features_df = pd.DataFrame(synthetic_features)
    synthetic_features_uniform_df = pd.DataFrame(synthetic_features_uniform)
    # Save DataFrame to CSV
    csv_output_path = os.path.join(output_folder, 'extracted_features.csv')
    df.to_csv(csv_output_path, index=False)

    synthetic_output_path = os.path.join(output_folder, f'synthetic_features_varying.csv')
    synthetic_features_df.to_csv(synthetic_output_path, index=False)
    synthetic_uniform_output_path = os.path.join(output_folder, f'synthetic_features_uniform.csv')
    synthetic_features_uniform_df.to_csv(synthetic_uniform_output_path, index=False)
    df_synthetic_varying = pd.DataFrame(synthetic_features)
    df_synthetic_uniform = pd.DataFrame(synthetic_features_uniform)

    # Plot and compare distributions
    plot_comparison_of_distributions(df, df_synthetic_varying, df_synthetic_uniform, output_folder)

    return df


def plot_comparison_of_distributions(original_df, synthetic_df_varying, synthetic_df_uniform, output_folder):
    """Plot Gaussian distributions for each feature and compare varying vs. uniform thresholds."""
    feature_columns = original_df.select_dtypes(include=[np.number]).columns

    os.makedirs(output_folder, exist_ok=True)

    for feature in feature_columns:
        plt.figure(figsize=(15, 6))

        # Original data
        plt.subplot(1, 3, 1)
        sns.histplot(original_df[feature], kde=True, stat="density", linewidth=0, color='blue', label='Original')
        mu, std = norm.fit(original_df[feature])
        x = np.linspace(*plt.xlim(), 100)
        plt.plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
        plt.title(f"Original Data - {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()

        # Synthetic data with varying thresholds
        plt.subplot(1, 3, 2)
        sns.histplot(synthetic_df_varying[feature], kde=True, stat="density", linewidth=0, color='green',
                     label='Varying Threshold')
        mu, std = norm.fit(synthetic_df_varying[feature])
        x = np.linspace(*plt.xlim(), 100)
        plt.plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
        plt.title(f"Varying Threshold - {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()

        # Synthetic data with uniform thresholds
        plt.subplot(1, 3, 3)
        sns.histplot(synthetic_df_uniform[feature], kde=True, stat="density", linewidth=0, color='red',
                     label='Uniform Threshold')
        mu, std = norm.fit(synthetic_df_uniform[feature])
        x = np.linspace(*plt.xlim(), 100)
        plt.plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
        plt.title(f"Uniform Threshold - {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(output_folder, f'comparison_{feature}.png')
        plt.savefig(plot_path)
        plt.close()

        print(f'Saved comparison plot for {feature} to {plot_path}')


# Paths to the folders
image_folder = './normal_type'
saliency_folder = './saliency_temp'
output_folder = './processed_images'

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Process the images, extract features, and save to CSV
df = process_images_and_save(image_folder, saliency_folder, output_folder)
plot_gaussian_distributions_and_save(df, output_folder)
