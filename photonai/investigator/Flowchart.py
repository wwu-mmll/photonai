from photonai.base.PhotonBase import Hyperpipe, PipelineElement, PipelineStack, PipelineBranch, PipelineSwitch


class Flowchart(object):

    def __init__(self,hyperpipe):
        self.hyperpipe = hyperpipe
        self.chart_str = ""

    def create_str(self):
        headerLayout = ""
        headerRelate = ""
        oldElement = ""
        for pipelineElement in self.hyperpipe.pipeline_elements:
            headerLayout = headerLayout+"["+pipelineElement.name+"]"
            if oldElement:
                headerRelate = headerRelate+"["+oldElement+"]"+"->"+"["+pipelineElement.name+"]\n"
            oldElement = pipelineElement.name
        self.chart_str = "Layout:\n"+headerLayout+"\nRelate:\n"+headerRelate+"\n"

        for pipelineElement in self.hyperpipe.pipeline_elements:
            self.chart_str = self.chart_str+self.recursivElement(pipelineElement,"")


    def recursivElement(self,pipe_element,parent):
        string = ""
        if hasattr(pipe_element,"pipeline_elements"):
            pipe_element.pipe_elements = pipe_element.pipeline_elements
        elif hasattr(pipe_element,"pipeline_element_list"):
            pipe_element.pipe_elements = pipe_element.pipeline_element_list

        if not hasattr(pipe_element,"pipe_elements"):
            if parent == "":
                string = "["+pipe_element.name+"]:\n"+"Define:\n"
            else:
                string = "[" + parent[1:]+"."+pipe_element.name + "]:\n" + "Define:\n"


        # Pipeline Stack
        elif isinstance(pipe_element, PipelineStack):
            if parent == "":
                string = "[" + pipe_element.name + "]:\n" + "Layout:\n"
            else:
                string = "["+parent[1:]+"."+pipe_element.name+"]:\n"+"Layout:\n"


            # Layout
            for pelement in list(pipe_element.pipe_elements.values()):
                string = string+"["+pelement.name+"]\n"
            string = string+"\n"
            for pelement in list(pipe_element.pipe_elements.values()):
                string = string+"\n"+self.recursivElement(pelement,parent=parent+"."+pipe_element.name)


        # Pipeline Switch
        elif isinstance(pipe_element, PipelineSwitch):
            if parent == "":
                string = "[" + pipe_element.name + "]:\n" + "Layout:\n"
            else:
                string = "[" + parent[1:]+"."+pipe_element.name + "]:\n" + "Layout:\n"

            # Layout
            for pelement in pipe_element.pipe_elements:
                string = string + "[" + pelement.name + "]\n"
            string = string + "\n"
            for pelement in pipe_element.pipe_elements:
                string = string + "\n" + self.recursivElement(pelement, parent=parent+"." + pipe_element.name)

        # Pipeline Branch
        elif isinstance(pipe_element, PipelineBranch):
            if parent == "":
                string = "[" + pipe_element.name + "]:\n" + "Layout:\n"
            else:
                string = "[" + parent[1:]+"."+pipe_element.name + "]:\n" + "Layout:\n"

            # Layout
            for pelement in pipe_element.pipe_elements:
                string = string + "[" + pelement.name + "]"
            string = string + "\n" + "Relate:\n"
            # Relate
            oldElement = ""
            for pelement in pipe_element.pipe_elements:
                if oldElement:
                    string = string + "[" + oldElement + "]" + "->" + "[" + pelement.name + "]\n"
                oldElement = pelement.name
                string = string + "\n"
                for pelement in pipe_element.pipe_elements:
                    string = string + "\n" + self.recursivElement(pelement, parent=parent+"." + pipe_element.name)


        return string





