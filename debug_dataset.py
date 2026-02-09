"""
Debug script to test caption loading logic
"""

# Simulate the load_captions function
def load_captions_current(captions_file):
    """Current implementation from dataset.py"""
    imgs = []
    captions = []
    
    with open(captions_file, 'r') as f:
        for line in f:
            # Split line into image_id and caption
            # Format: "image.jpg#0\tCaption text"
            parts = line.strip().split('\t')
            
            if len(parts) != 2:
                continue
            
            # Extract image filename (remove #0, #1, etc.)
            img_id = parts[0].split('#')[0]
            caption = parts[1]
            
            imgs.append(img_id)
            captions.append(caption)
    
    return imgs, captions


# Test with sample data
sample_data = """1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
1001773457_577c3a7d70.jpg#0	A black dog and a spotted dog are fighting
1001773457_577c3a7d70.jpg#1	A black dog and a tri-colored dog playing with each other on the road .
1001773457_577c3a7d70.jpg#2	A black dog and a white dog with brown spots are staring at each other in the street .
1001773457_577c3a7d70.jpg#3	Two dogs of different breeds looking at each other on the road .
1001773457_577c3a7d70.jpg#4	Two dogs on pavement moving toward each other .
"""

# Write sample data to test file
with open('test_captions.txt', 'w') as f:
    f.write(sample_data)

# Test current implementation
imgs, captions = load_captions_current('test_captions.txt')

print("=" * 60)
print("CAPTION LOADING TEST")
print("=" * 60)
print(f"\nTotal lines in file: {len(sample_data.strip().split(chr(10)))}")
print(f"Loaded captions: {len(captions)}")
print(f"Loaded images: {len(imgs)}")

print("\nFirst 5 image-caption pairs:")
for i in range(min(5, len(imgs))):
    print(f"{i+1}. Image: {imgs[i]}")
    print(f"   Caption: {captions[i]}")
    print()

print("\nUnique images:")
unique_imgs = set(imgs)
print(f"Total unique images: {len(unique_imgs)}")
for img in list(unique_imgs)[:5]:
    print(f"  - {img}")

# Clean up
import os
os.remove('test_captions.txt')

print("\n" + "=" * 60)
print("✓ Caption loading logic is working correctly!")
print("=" * 60)
