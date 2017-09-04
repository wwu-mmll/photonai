# Register Photon_Neuro components
from Framework.Register import RegisterPipelineElement

package_name = 'PhotonNeuro'

photon_name = 'ResampleImgs'  # element name
class_str = 'PhotonNeuro.ImageBasics.ResamplingImgs'  # element info
element_type = 'Transformer'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).add()

photon_name = 'SmoothImgs'
class_str = 'PhotonNeuro.ImageBasics.SmoothImgs'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).add()

photon_name = 'BrainAtlas'
class_str = 'PhotonNeuro.BrainAtlas'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).add()

