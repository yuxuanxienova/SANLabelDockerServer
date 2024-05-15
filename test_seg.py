from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
 
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    j = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        
        # Convert boolean mask to uint8
        m_uint8 = m.astype(np.uint8)
        
        # Calculate centroid of the mask
        moments = cv2.moments(m_uint8)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            
            # Add text annotation with mask number
            ax.text(centroid_x, centroid_y, str(j), color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
        j += 1
sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
device = "cpu"
model_type = "default"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# predictor = SamPredictor(sam)

image = cv2.imread('fp3.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# mask_generator = SamAutomaticMaskGenerator(sam)

mask_generator = SamAutomaticMaskGenerator(
model=sam,
points_per_side=24,#default:32
pred_iou_thresh=0.88,#default:0.88
stability_score_thresh=0.95,#default:0.95
crop_n_layers=0,#default:0
crop_n_points_downscale_factor=1,#default:1
min_mask_region_area=100,  # Requires open-cv to run post-processing
)

masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()