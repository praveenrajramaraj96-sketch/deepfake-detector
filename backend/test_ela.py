from PIL import Image, ImageChops, ImageStat
import os

def compute_ela_tampering_score(image_path: str) -> float:
    image = Image.open(image_path).convert('RGB')
    
    # Save image at known quality
    temp_filename = "temp_ela_test.jpg"
    image.save(temp_filename, 'JPEG', quality=90)
    
    # Read back the saved image
    compressed_image = Image.open(temp_filename)
    
    # Calculate absolute difference
    diff = ImageChops.difference(image, compressed_image)
    
    # Calculate mean difference
    stat = ImageStat.Stat(diff)
    mean_diff = sum(stat.mean) / len(stat.mean)
    
    os.remove(temp_filename)
    
    print(f"Mean diff: {mean_diff}")
    
    # Map to 0-100 percentage
    # Typically, original JPEGs resaved at 90% have very low diff. Tampered regions have higher diff.
    tamper_score = min(100.0, max(0.0, (mean_diff - 1.5) * 25.0))
    return tamper_score

# create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img.save('test.jpg', quality=95)

print("Original image score:", compute_ela_tampering_score('test.jpg'))

# tamper it
img2 = Image.open('test.jpg')
img2.paste(Image.new('RGB', (20, 20), color='blue'), (10, 10))
img2.save('tampered.jpg', quality=95)

print("Tampered image score:", compute_ela_tampering_score('tampered.jpg'))
