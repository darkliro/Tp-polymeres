
##ce programme est disponible publiquement sous licence MIT. Pour plus d'informations contacter le proprietaire du repo.



import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from collections import Counter
import pyexcel_ods
from datetime import datetime
from tkinter import ttk
from skimage import color
from skimage.exposure import rescale_intensity

# Enable OpenCL
cv2.ocl.setUseOpenCL(True)

class CircleDetectionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Circle Detection Application")
        self.image_path = None
        self.output_dir = None
        self.image_display = None
        self.thumbnail_label = None
        self.status_label = None
        self.file_prefix = tk.StringVar(value="test")
        self.image_paths = []
        self.batch_mode = False
        self.he_process = tk.BooleanVar(value=False)

        # Add H&E parameters
        self.he_params = {
            'h_vector': [0.644211, 0.716556, 0.266844],
            'e_vector': [0.092789, 0.954111, 0.283111],
            'lower_threshold': tk.IntVar(value=71),  # Default to 71%
            'upper_threshold': tk.IntVar(value=255),
            'dark_background': tk.BooleanVar(value=True),
            'stack_histogram': tk.BooleanVar(value=False),
            'dont_reset_range': tk.BooleanVar(value=True),
            'raw_values': tk.BooleanVar(value=False),
            'sixteen_bit': tk.BooleanVar(value=False)
        }

        self.current_img_cpu = None
        self.current_circles = None
        self.current_params = None
        self.save_button = None  # Add reference for save button

        self.params = {
            'minRadius': tk.IntVar(value=1),
            'maxRadius': tk.IntVar(value=100),
            'dp': tk.DoubleVar(value=1.0),
            'param2': tk.IntVar(value=30)  # Add param2 with default value
        }

        self.setup_gui()

    def setup_gui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left frame for controls
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(side='left', fill='y', padx=5)
        
        # Right frame for image display
        display_frame = ttk.LabelFrame(main_container, text="Image Preview")
        display_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        self.image_display = ttk.Label(display_frame)
        self.image_display.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Parameters with default values
        # File selection frame with thumbnail
        file_frame = ttk.LabelFrame(controls_frame, text="File Selection", padding=5)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        # Left side for file selection
        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.pack(side='left', fill='x', expand=True)
        
        self.image_label = ttk.Label(file_select_frame, text="No image selected")
        self.image_label.pack(side='left', padx=5)
        ttk.Button(file_select_frame, text="Select Image", command=self.select_image).pack(side='right', padx=5)
        
        # Right side for thumbnail
        thumbnail_frame = ttk.Frame(file_frame)
        thumbnail_frame.pack(side='right', padx=5)
        self.thumbnail_label = ttk.Label(thumbnail_frame)
        self.thumbnail_label.pack()

        # Add batch processing options to file selection frame
        batch_frame = ttk.Frame(file_select_frame)
        batch_frame.pack(side='right', padx=5)
        ttk.Button(batch_frame, text="Select Images", command=self.select_multiple_images).pack(side='top', pady=2)
        ttk.Button(batch_frame, text="Select Folder", command=self.select_image_folder).pack(side='bottom', pady=2)

        # Output directory frame
        output_frame = ttk.LabelFrame(controls_frame, text="Output Directory", padding=5)
        output_frame.pack(fill='x', padx=5, pady=5)
        
        self.output_label = ttk.Label(output_frame, text="No directory selected")
        self.output_label.pack(side='left', padx=5)
        
        ttk.Button(output_frame, text="Select Directory", command=self.select_output).pack(side='right', padx=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(controls_frame, text="Parameters", padding=5)
        params_frame.pack(fill='x', padx=5, pady=5)
        
        # Create labels to show values
        self.value_labels = {}
        
        # Create and pack sliders with their value labels
        for param_name, label_text in [
            ('minRadius', 'Minimum Radius:'),
            ('maxRadius', 'Maximum Radius:'),
            ('dp', 'DP:'),
            ('param2', 'Sensitivity:')  # Add param2 slider
        ]:
            frame = ttk.Frame(params_frame)
            frame.pack(fill='x', padx=5, pady=5)
            
            ttk.Label(frame, text=label_text).pack(side='left')
            self.value_labels[param_name] = ttk.Label(frame, text=f"Value: {self.params[param_name].get():.1f}")
            self.value_labels[param_name].pack(side='right')
            
            slider = ttk.Scale(
                params_frame,
                from_=(0 if param_name == 'dp' else 1),
                to=(1.0 if param_name == 'dp' else (100 if param_name == 'param2' else 300)),  # Different range for param2
                variable=self.params[param_name],
                orient='horizontal'
            )
            slider.pack(fill='x', padx=5)
            
            # Bind value change to label update
            self.params[param_name].trace_add('write', lambda *args, pn=param_name: self.update_label(pn))
        
        # Add H&E processing checkbox and parameters
        he_frame = ttk.LabelFrame(controls_frame, text="H&E Processing")
        he_frame.pack(fill='x', padx=5, pady=5)
        
        # Checkbox
        ttk.Checkbutton(he_frame, text="Apply H&E Processing", 
                       variable=self.he_process, 
                       command=self.toggle_he_params).pack(pady=5)
        
        # H&E Parameters (initially hidden)
        self.he_params_frame = ttk.Frame(he_frame)
        
        # Threshold controls
        thresh_frame = ttk.LabelFrame(self.he_params_frame, text="Threshold")
        thresh_frame.pack(fill='x', padx=5, pady=5)
        
        # Lower and upper threshold sliders
        lower_frame = ttk.Frame(thresh_frame)
        lower_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(lower_frame, text="Lower:").pack(side='left')
        ttk.Scale(lower_frame, from_=0, to=255,
                 variable=self.he_params['lower_threshold'],
                 orient='horizontal').pack(side='right', fill='x', expand=True)
        
        upper_frame = ttk.Frame(thresh_frame)
        upper_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(upper_frame, text="Upper:").pack(side='left')
        ttk.Scale(upper_frame, from_=0, to=255,
                 variable=self.he_params['upper_threshold'],
                 orient='horizontal').pack(side='right', fill='x', expand=True)
        
        # Checkboxes
        checks_frame = ttk.Frame(thresh_frame)
        checks_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Checkbutton(checks_frame, text="Dark background",
                       variable=self.he_params['dark_background']).pack(anchor='w')
        ttk.Checkbutton(checks_frame, text="Stack histogram",
                       variable=self.he_params['stack_histogram']).pack(anchor='w')
        ttk.Checkbutton(checks_frame, text="Don't reset range",
                       variable=self.he_params['dont_reset_range']).pack(anchor='w')
        ttk.Checkbutton(checks_frame, text="Raw values",
                       variable=self.he_params['raw_values']).pack(anchor='w')
        ttk.Checkbutton(checks_frame, text="16-bit histogram",
                       variable=self.he_params['sixteen_bit']).pack(anchor='w')

        # Process button and status
        process_frame = ttk.Frame(controls_frame)
        process_frame.pack(fill='x', padx=5)
        ttk.Button(process_frame, text="Process Image", command=self.process_image).pack(pady=(10,5))
        self.save_button = ttk.Button(process_frame, text="Save Iteration", 
                                    command=self.save_current_iteration, state='disabled')
        self.save_button.pack(pady=(0,5))
        self.status_label = ttk.Label(process_frame, text="Ready", font=('Arial', 9, 'italic'))
        self.status_label.pack(pady=(0,10))

        # File prefix input
        prefix_frame = ttk.Frame(output_frame)
        prefix_frame.pack(fill='x', expand=True)
        ttk.Label(prefix_frame, text="File prefix:").pack(side='left', padx=5)
        ttk.Entry(prefix_frame, textvariable=self.file_prefix).pack(side='left', fill='x', expand=True, padx=5)

    def update_label(self, param_name, *args):
        self.value_labels[param_name].config(text=f"Value: {self.params[param_name].get():.1f}")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        if file_path:
            self.image_path = file_path
            self.image_paths = []  # Clear batch mode
            self.batch_mode = False
            self.image_label.config(text=os.path.basename(file_path))
            # Create and show thumbnail
            self.update_thumbnail(file_path)

    def select_multiple_images(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        if file_paths:
            self.image_paths = list(file_paths)
            self.batch_mode = True
            self.image_label.config(text=f"Selected {len(self.image_paths)} images")
            if self.image_paths:
                self.image_path = self.image_paths[0]  # Set first image for thumbnail
                self.update_thumbnail(self.image_path)

    def select_image_folder(self):
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if folder_path:
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            self.image_paths = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(image_extensions)
            ]
            if self.image_paths:
                self.batch_mode = True
                self.image_label.config(text=f"Selected {len(self.image_paths)} images")
                self.image_path = self.image_paths[0]  # Set first image for thumbnail
                self.update_thumbnail(self.image_path)
            else:
                messagebox.showwarning("Warning", "No compatible images found in folder")

    def select_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_label.config(text=os.path.basename(directory))

    def toggle_he_params(self):
        """Show/hide H&E parameters based on checkbox state"""
        if self.he_process.get():
            self.he_params_frame.pack(fill='x', padx=5, pady=5)
        else:
            self.he_params_frame.pack_forget()

    def process_he_image(self, image):
        # Convert to float
        ihc_rgb = image.astype(float)
        ihc_rgb /= 255

        # H&E deconvolution matrix
        he_matrix = np.array([
            self.he_params['h_vector'],
            self.he_params['e_vector'],
            [0.0, 0.0, 0.0]
        ])

        # Perform deconvolution
        deconv = color.separate_stains(ihc_rgb, he_matrix)
        
        # Keep Eosin channel (pink) instead of Hematoxylin
        e_channel = deconv[:, :, 1]
        
        # Rescale and convert to 8-bit
        e_rescaled = rescale_intensity(e_channel, out_range=(0, 255))
        e_8bit = e_rescaled.astype(np.uint8)
        
        # Apply thresholding with FIJI-like controls
        if self.he_params['dark_background'].get():
            e_8bit = cv2.bitwise_not(e_8bit)
            
        _, binary = cv2.threshold(
            e_8bit,
            self.he_params['lower_threshold'].get(),
            self.he_params['upper_threshold'].get(),
            cv2.THRESH_BINARY
        )
        
        if self.he_params['dark_background'].get():
            binary = cv2.bitwise_not(binary)
            
        return binary

    def save_results(self, img_cpu, circles, params):
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = self.file_prefix.get()
        
        # Add image name to subfolder for batch mode
        if self.batch_mode:
            prefix = f"{prefix}_{os.path.splitext(os.path.basename(self.image_path))[0]}"
        
        # Create subfolder with prefix name
        subfolder = os.path.join(self.output_dir, prefix)
        os.makedirs(subfolder, exist_ok=True)
        
        # Save the image with detected circles
        img_path = os.path.join(subfolder, f"circles_{timestamp}.png")
        cv2.imwrite(img_path, img_cpu)
        
        # Save parameters to text file
        params_path = os.path.join(subfolder, f"params_{timestamp}.txt")
        with open(params_path, 'w') as f:
            f.write("Hough Transform Parameters:\n")
            f.write(f"dp: {params['dp']}\n")
            f.write(f"minDist: {params['minDist']}\n")
            f.write(f"param1: {params['param1']}\n")
            f.write(f"param2: {params['param2']}\n")
            f.write(f"minRadius: {params['minRadius']}\n")
            f.write(f"maxRadius: {params['maxRadius']}\n")
        
        # Create statistics for ODS file
        radius_counts = Counter(circle[2] for circle in circles[0])
        stats_data = [["Radius", "Count"]]
        for radius, count in sorted(radius_counts.items()):
            stats_data.append([int(radius), count])
        stats_data.append(["Total", len(circles[0])])
        
        # Save to ODS file
        ods_path = os.path.join(subfolder, f"stats_{timestamp}.ods")
        pyexcel_ods.save_data(ods_path, {"Circle Statistics": stats_data})

        # Save H&E processed image if enabled
        if self.he_process.get():
            he_processed = self.process_he_image(cv2.cvtColor(img_cpu, cv2.COLOR_BGR2RGB))
            he_path = os.path.join(subfolder, f"he_processed_{timestamp}.png")
            cv2.imwrite(he_path, he_processed)

    def save_current_iteration(self):
        """Save the current processed results"""
        if self.current_img_cpu is not None and self.current_circles is not None:
            self.save_results(self.current_img_cpu, self.current_circles, self.current_params)
            self.status_label.config(text="Results saved successfully!")

    def process_image(self):
        if not (self.image_path or self.image_paths):
            messagebox.showerror("Error", "Please select image(s) first")
            return
        if not self.output_dir:
            messagebox.showerror("Error", "Please select output directory first")
            return

        config_params = {
            'minRadius': self.params['minRadius'].get(),
            'maxRadius': self.params['maxRadius'].get(),
            'dp': self.params['dp'].get(),
            'param2': self.params['param2'].get()  # Add param2 to config
        }

        if self.batch_mode:
            self.process_batch(config_params)
        else:
            # Enable save button after processing
            self.detect_circles(self.image_path, config_params)

    def process_batch(self, config_params):
        total = len(self.image_paths)
        for i, img_path in enumerate(self.image_paths, 1):
            self.status_label.config(text=f"Processing image {i}/{total}...")
            self.root.update()
            
            try:
                self.detect_circles(img_path, config_params)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
            
        self.status_label.config(text=f"Completed processing {total} images")

    def update_thumbnail(self, image_path):
        # Read image
        if image_path.lower().endswith(('.tiff', '.tif')):
            pil_image = Image.open(image_path)
        else:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
        
        # Resize to thumbnail size (100x100)
        thumbnail_size = (100, 100)
        pil_image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.thumbnail = ImageTk.PhotoImage(pil_image)
        self.thumbnail_label.config(image=self.thumbnail)

    def update_image_display(self, cv2_image):
        # Convert from BGR to RGB
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Calculate scaling to fit in window (max 800x600)
        display_width = 800
        display_height = 600
        
        # Calculate scaling factor
        w_scale = display_width / pil_image.width
        h_scale = display_height / pil_image.height
        scale = min(w_scale, h_scale)
        
        # Resize image
        new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        self.image_display.config(image=self.photo)

    def detect_circles(self, image_path, config_params):
        # Read image, with special handling for TIFF
        if image_path.lower().endswith('.tiff') or image_path.lower().endswith('.tif'):
            # Use PIL to read TIFF
            pil_image = Image.open(image_path)
            # Convert PIL image to numpy array
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(image_path)
        
        if img is None:
            print("Error: Could not load image")
            return

        # Convert to UMat for GPU acceleration
        img_umat = cv2.UMat(img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_umat, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Store parameters
        hough_params = {
            'dp': config_params['dp'],
            'minDist': 50,
            'param1': 50,
            'param2': config_params['param2'],  # Use configured param2
            'minRadius': config_params['minRadius'],
            'maxRadius': config_params['maxRadius']
        }
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=hough_params['dp'],
            minDist=hough_params['minDist'],
            param1=hough_params['param1'],
            param2=hough_params['param2'],
            minRadius=hough_params['minRadius'],
            maxRadius=hough_params['maxRadius']
        )

        # Convert back to CPU for drawing
        img_cpu = img_umat.get()
        
        if circles is not None:
            # Convert UMat circles to numpy array before using np.around
            circles = circles.get()
            circles = np.uint16(np.around(circles))
            
            # Get image dimensions for sheet size report
            height, width = img_cpu.shape[:2]
            print(f"\nImage size: {width}x{height} pixels")
            
            # Draw circles and add IDs
            for idx, circle in enumerate(circles[0, :]):
                x, y, r = circle
                
                # Draw the outer circle
                cv2.circle(img_cpu, (x, y), r, (0, 255, 0), 2)
                # Draw the center
                cv2.circle(img_cpu, (x, y), 2, (0, 0, 255), 3)
                # Add ID number
                cv2.putText(img_cpu, f"#{idx+1}", (x-20, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                print(f"Circle #{idx+1}: Center({x},{y}), Radius={r}")

            # Store current results and enable save button
            self.current_img_cpu = img_cpu
            self.current_circles = circles
            self.current_params = hough_params
            self.save_button.configure(state='normal')

            # Update display with appropriate image
            if self.he_process.get():
                display_img = self.process_he_image(cv2.cvtColor(img_cpu, cv2.COLOR_BGR2RGB))
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            else:
                display_img = img_cpu
            
            self.update_image_display(display_img)
            self.status_label.config(text="Processing complete! Click Save Iteration to save results.")
        else:
            self.current_img_cpu = None
            self.current_circles = None
            self.current_params = None
            self.save_button.configure(state='disabled')
            messagebox.showinfo("Result", "No circles detected")
            self.status_label.config(text="No circles found")

    def run(self):
        # Set initial window size
        self.root.geometry("1200x800")
        
        # Add window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing event"""
        self.root.quit()
        self.root.destroy()

def main():
    if not cv2.ocl.haveOpenCL():
        print("OpenCL is not available")
        return

    print("Using OpenCL:", cv2.ocl.useOpenCL())
    print("OpenCL device:", cv2.ocl.Device.getDefault().name())
    
    app = CircleDetectionApp()
    app.run()

if __name__ == "__main__":

    main()

