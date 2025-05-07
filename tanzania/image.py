import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class Colors:
    text_and_background_1 = '#f4f3ec'
    text_and_background_2 = '#004852'
    text_and_background_3 = '#e1ff00'
    text_and_background_4 = '#ced7d6'

    accent_1 = '#a2c1c6'
    accent_2 = '#d8e4d6'
    accent_3 = '#aac1a9'
    accent_4 = '#1f2124'
    accent_5 = '#678465'
    accent_6 = '#8598a7'

    link = '#004852'

    slide_background = '#f4f3ec'
    dark_blue_text = '#004852'
    canary_lemon = '#e1ff00'
    gray_blue = '#ced7d6'
    gray_blue_accent = '#a2c1c6'
    low_saturation_light_green = '#d8e4d6'
    low_saturation_middle_reen = '#aac1a9'
    very_dark_blue = '#1f2124'
    dark_matcha_green = '#678465'
    night_blue = '#8598a7'

    palette = [
        link,
        accent_5,
        accent_4,
        accent_6,
        accent_3,
        accent_2,
        accent_1,
        text_and_background_3,
        text_and_background_2
    ]

    colormap = LinearSegmentedColormap.from_list(
        "tanzania_colormap", [
            text_and_background_2,
            text_and_background_3,
        ]
    )
    colormap2 = LinearSegmentedColormap.from_list(
        "tanzania_colormap_2", [
            accent_5, accent_6, link
        ]
    )

    @staticmethod
    def init_colors():
        """ this function initializes seaborn, mathplotlib, pyplot palette with 
            tanzania colors, sets some default parameters
        """        
        sns.set_palette(Colors.palette)
        sns.set_style('ticks')
        plt.rcParams['figure.figsize'] = (7, 4)

Colors.palette2 = [Colors.colormap(x)  for x in np.linspace(0, 1, 10)]
Colors.palette3 = [Colors.colormap2(x) for x in np.linspace(0, 1, 10)]

def save_png(file_name: str):
    """save image in acceptable quality as PNG type keeping transparency

        ATTENTION!!!. this function should be called just after the graph 
        created and configured, but BEFORE plt.show()
    Args:
        file_name (str): destination file name
    """    
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

if __name__ == '__main__':
    Colors.init_colors()
