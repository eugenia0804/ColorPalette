import os
from colorthief import ColorThief
from sklearn.cluster import KMeans
from colorsys import rgb_to_hls
from itertools import combinations
import numpy as np
from PIL import Image, ImageDraw


def generate_palette(folder_path, color_count_for_each_image, quality_of_palette_detection):
    files = os.listdir(folder_path)
    images = [file for file in files if file.endswith(('jpg', 'jpeg', 'png'))]

    palette = []

    for idx, image in enumerate(images):
        if idx == 4:
            break
        image_path = os.path.join(folder_path, image)
        color_thief = ColorThief(image_path)
        dominant_colors = color_thief.get_palette(color_count=color_count_for_each_image,quality=quality_of_palette_detection) # Get multiple dominant colors
        palette.extend(dominant_colors)  # Extend the palette with the dominant colors

    return palette


def get_representative_colors(rgb_tuples, num_colors):
    # Convert RGB tuples to numpy array
    rgb_array = np.array(rgb_tuples)

    # Define the number of clusters to be generated
    num_clusters = len(rgb_tuples) // 3
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(rgb_array)

    # Get the cluster centroids
    cluster_centers = kmeans.cluster_centers_

    # Compute pairwise distances between cluster centroids
    pairwise_distances = np.linalg.norm(cluster_centers[:, np.newaxis] - cluster_centers[np.newaxis, :], axis=-1)

    # Find num_colors clusters with maximum dissimilarity
    cluster_indices = np.arange(num_clusters)
    max_dissimilarity = -1
    best_cluster_combination = None
    for combination in combinations(cluster_indices, num_colors):
        # Compute average dissimilarity for the combination of clusters
        avg_dissimilarity = np.mean(pairwise_distances[np.ix_(combination, combination)])
        if avg_dissimilarity > max_dissimilarity:
            max_dissimilarity = avg_dissimilarity
            best_cluster_combination = combination

    # Get the representative colors for the selected clusters
    representative_colors = np.array([cluster_centers[i] for i in best_cluster_combination])

    # Convert representative colors to RGB tuples
    representative_colors_rgb = [tuple(color.astype(int)) for color in representative_colors]

    return representative_colors_rgb


def colorFader(c1, c2, mix=0):
    """
    Fade (linear interpolate) from color c1 (at mix=0) to c2 (at mix=1).
    """
    c1 = np.array(c1)
    c2 = np.array(c2)
    return tuple((1 - mix) * c1 + mix * c2)


def generate_color_gradient(colors, filepath, n=500):
    # Define image size and create a new blank image
    width, height = 300, 300  # Adjust according to your requirement
    gradient_image = Image.new("RGB", (width, height))

    # Create a draw object
    draw = ImageDraw.Draw(gradient_image)

    # Draw a rectangle filled with gradient
    for y in range(height):
        for x in range(width):
            # Calculate the color at each pixel based on the distance from corners
            color = (
                int((colors[0][0] * (width - x) * (height - y) + colors[1][0] * x * (height - y) + colors[2][0] * (width - x) * y + colors[3][0] * x * y) / (width * height)),
                int((colors[0][1] * (width - x) * (height - y) + colors[1][1] * x * (height - y) + colors[2][1] * (width - x) * y + colors[3][1] * x * y) / (width * height)),
                int((colors[0][2] * (width - x) * (height - y) + colors[1][2] * x * (height - y) + colors[2][2] * (width - x) * y + colors[3][2] * x * y) / (width * height))
            )
            draw.point((x, y), fill=color)
    gradient_image.save(filepath)


def get_folder_list():
    current_directory = os.getcwd()
    folders = [folder for folder in os.listdir(current_directory) if os.path.isdir(folder) and folder != ('palettes' or '.git' or '.config')]
    return folders

def get_user_input(prompt, default):
    user_input = input(prompt + f" (default: {default}): ")
    if user_input.strip() == "":
        return default
    return user_input



# Example usage:
folder_list = get_folder_list()
color_count_for_each_image = int(get_user_input("Enter the number of colors detected for each image", 20))
quality_of_palette_detection = int(get_user_input("Enter quality of palette detection", 30))


for folder in folder_list:
    folder = f'ColorPalette/{folder}'
    print("start processing folder: " + folder)
    colors_8 = generate_palette(folder, color_count_for_each_image, quality_of_palette_detection)
    print("palette v2 generated: " + str(colors_8))
    colors_4 = get_representative_colors(colors_8, 4)
    print("palette v2 generated: " + str(colors_4))
    generate_color_gradient(colors_4, f"palettes/{folder}.png")
    print("image generated and saved!")
