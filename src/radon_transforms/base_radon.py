# class RadonTransformKwargs():
#     def __init__(self):
#         self.AAPMRadonTransformKwargs = {
#             "fanbeam": True,
#             "resolution": 256,
#             "source_distance": 1075,
#             "det_count": 672
#         }


class BaseRadonTransform:
    def __init__(self):
        pass
    
    def radon(self, image, num_projections=720):
        """
        Apply Radon transform to a 2D image or batch of 2D images.
        Returns a sinogram and thetas on the same device as the input image.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def iradon(self, sinogram, thetas):
        """
        Apply inverse Radon transform to a 2D sinogram or batch of 2D sinograms.
        Returns a torch.Tensor on the same device as the input sinogram.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    