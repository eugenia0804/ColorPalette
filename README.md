# Color Palette Generator

#### Also available at: [Google Colab](https://colab.research.google.com/drive/14ghjFK6xqRiUMLrZPRChIipQ9a3ZUM7o?usp=sharing)

### Project Description:
This project generates representative colors based on input images and outputs gradient descent images. The code is specifically modified to allow easy user interaction. Generated images can be controlled by user-input parameters.

### Methodology:
Step 1: Extracting colors from images, powered by [Colorthief](https://github.com/fengsp/color-thief-py), outputting a list of RGB tuples. Available parameters: `color_count_for_each_image`, `quality_of_palette_detection`. 

Step 2: Generate 4 representative colors from each group, utilizing k-means regression model but limit the number of colors in each cluster, outputting the four clusters that are most different from each other to generate a contrasting color palette. 

Step 3: Generate gradient descent final images, powered by [PIL](https://python-pillow.org/).


