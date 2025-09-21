import os
from PIL import Image

def process_images():
    base_dir = "custom_data"
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        print(f"Processing folder: {folder_name}")
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                
                try:
                    
                    img = Image.open(img_path)
                    
                    
                    width, height = img.size
                    
                    
                    max_side = max(width, height)
                    
                    
                    square_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
                    
                    
                    x_offset = (max_side - width) // 2
                    y_offset = (max_side - height) // 2
                    
                    
                    square_img.paste(img, (x_offset, y_offset))
                    
                    
                    final_img = square_img.resize((256, 256), Image.LANCZOS)
                    
                    
                    final_img.save(img_path, 'JPEG', quality=100)
                    
                    print(f"Processed: {filename}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    process_images()
    print("All images processed!")