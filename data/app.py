import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Function to move file
def move_file(destination):
    global current_image_index, image_list
    shutil.move(image_list[current_image_index], destination)
    current_image_index += 1
    update_status()
    if current_image_index < len(image_list):
        display_image(image_list[current_image_index])
    else:
        root.quit()

# Function to display image
def display_image(image_path):
    img = Image.open(image_path)
    # img = img.resize((500, 500), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

# Key press event handlers
def on_key_press(event):
    if event.char == '1':
        move_file(folder0)
    elif event.char == '0':
        move_file(folder1)
    elif event.char == '2':
        move_file(folder2)

# Function to update the status label
def update_status():
    remaining_files = len(image_list) - current_image_index
    status_label.config(text=f"Remaining files: {remaining_files}")

# Select directory containing images
image_dir = filedialog.askdirectory(title='Select Directory')
folder0 = os.path.join(image_dir, '/Users/greg/Desktop/Education/Project/pdf2LaTeX/dataset_5')
folder1 = os.path.join(image_dir, '/Users/greg/Desktop/Education/Project/pdf2LaTeX/dataset_6')
folder2 = os.path.join(image_dir, '/Users/greg/Desktop/Education/Project/pdf2LaTeX/unclassified_image')
os.makedirs(folder0, exist_ok=True)
os.makedirs(folder1, exist_ok=True)
os.makedirs(folder2, exist_ok=True)

# Get list of images
image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
image_list.sort()
current_image_index = 0

# Setup GUI
root = tk.Tk()
root.title('Image Sorter')
root.bind('<Key>', on_key_press)  # Bind key press event

panel = tk.Label(root)
panel.pack()

# Add status label to show the number of remaining files
status_label = tk.Label(root, text="")
status_label.pack()

# Display the first image and update the status
if image_list:
    display_image(image_list[current_image_index])
    update_status()
else:
    print("No images found in the directory")
    root.quit()

# Start the GUI loop
root.mainloop()
