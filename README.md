# Multimodal Fusion Recognition Model(for PD)
This project contains pytorch implementation of multimodal fusion and recognition. It has been used to detect Parkinson's disease, and the name of the paper is: 'Multi-Scale Deep Information Mining for Fusion Motor Symptom Evaluation in Detection of Parkinson's Disease'. The overall process is shown in the figure below.
![](overallarchitecture.png)
![](extraction_fusion.png)
## Usage:
    Spatial_attention & self_attention are used to enhance data features. It is recommended to use different methods based on different data forms.
    
    The processing of data is a bit cumbersome. It is recommended to follow the diagram above.

![](mvdconvnet.png)

    Generate fusion data in 'fusion', 'model' includes work: MVDConvnet.

