import os

# Get the absolute path to the folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path for the Chroma database folder
CHROMA_DIR = os.path.join(BASE_DIR, "chroma")