from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict,List,Tuple
class ImageProcessor:
    def __init__(self) -> None:
        sam_checkpoint = "./models/sam_vit_h_4b8939.pth"
        device = "cpu"
        model_type = "default"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,#default:32
        pred_iou_thresh=0.88,#default:0.88
        stability_score_thresh=0.95,#default:0.95
        crop_n_layers=0,#default:0
        crop_n_points_downscale_factor=1,#default:1
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        
    def process_SetOfMask(self,pil_image):
        image_cv2_rgb=self.pilTOcv2RGB(pil_image)
        masks = self.mask_generator.generate(image_cv2_rgb)
        
        # Plot the image with masks
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(pil_image)
        maskNumberToPixelCorDict = self.AddNumberTagsToMasks(ax, masks)
        
        # Convert the plot to a PIL Image object
        fig.canvas.draw()
        pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        
        plt.close(fig)  # Close the plot to release resources
        
        # Return the PIL Image object
        return pil_image, maskNumberToPixelCorDict
    def AddNumberTagsToMasks(self,ax,anns)->Dict[int,Tuple[int,int]]:
        '''
        AddNumber Tags to masks
        Return Dictionary, the key is tag number, value is cantroid in image pixel coordinate
        '''
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        j = 0
        maskNumberToPixelCorDict:Dict[int,Tuple[int,int]]={}
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
                centroid_u = int(moments["m10"] / moments["m00"])
                centroid_v = int(moments["m01"] / moments["m00"])
                
                # Add text annotation with mask number
                ax.text(centroid_u, centroid_v, str(j), color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
                maskNumberToPixelCorDict[j] = (centroid_u,centroid_v)
            j += 1
        return maskNumberToPixelCorDict
    def pilTOcv2RGB(self,pil_image):
        # Convert PIL Image to NumPy array
        image_np = np.array(pil_image)

        # Convert BGR to RGB
        image_cv2_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        return image_cv2_rgb

if __name__=="__main__":
    image = cv2.imread('test_fp.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processor = ImageProcessor()
    pil_image, maskNumberToPixelCorDict = processor.process_SetOfMask(image)
    
    # Visualize the PIL Image
    plt.imshow(pil_image)
    plt.axis('off')  # Optional: turn off axis
    plt.show()