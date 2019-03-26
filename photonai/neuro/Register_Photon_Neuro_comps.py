# Register Photon_Neuro components
from Framework.Register import RegisterPipelineElement

package_name = 'neuro'

photon_name = 'ResampleImgs'  # element name
class_str = 'neuro.ImageBasics.ResampleImages'  # element info
element_type = 'Transformer'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).save()

photon_name = 'SmoothImages'
class_str = 'neuro.ImageBasics.SmoothImages'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).save()

photon_name = 'BrainAtlas'
class_str = 'neuro.BrainAtlas'
RegisterPipelineElement(photon_name=photon_name,
                        photon_package=package_name,
                        class_str=class_str,
                        element_type=element_type).save()

