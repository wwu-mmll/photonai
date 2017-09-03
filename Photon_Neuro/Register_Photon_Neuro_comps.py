# Register Photon_Neuro components
from Framework.Register import RegisterPipelineElement

package_name = 'PhotonNeuro'

photon_name = 'ResampleImgs'  # element name
class_str = 'Photon_Neuro.ImageBasics.ResamplingImgs'  # element info
element_type = 'Transformer'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).add()

photon_name = 'SmoothImgs'
class_str = 'Photon_Neuro.ImageBasics.SmoothImgs'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).add()

photon_name = 'BrainAtlas'
class_str = 'Photon_Neuro.BrainAtlas'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).add()

