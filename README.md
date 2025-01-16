# FUCCI
FUCCI analysis pipeline to automate cell-cycle tracking and segmentation for motile cells\

Our automated image analysis pipeline can be broken into three main steps: 1) pre-processing, 2) cell tracking, and 3) data plotting. 


1) First, we pre-process raw time-series images over 4 channels (bright field, mCherry, mVenus, DAPI) through a Cellprofiler pipleline (FUCCI_preprocess.ccproj) which removes dead cells (through overlaying masks from the DAPI channel) from further processing steps in the mCherry and mVenus channels. Then, we use a separate Python script (formatstacks.py) to arrange the pre-processed images into Tiff stacks of mCherry and mVenus channels for each well. The specific script can be modified depending on the naming conventions of the user’s specific microscopy system.

  
2) At this stage, we run the image stacks through a FUCCI analysis (pipeline.py) to automate segmentation, cell tracking, and track merging with several quantitative variables. This pipeline was built off of confluentFUCCI, an open-source package developed by Leo Goldstein (Goldstein et al. 2024), to whom we owe an immense gratitude for his generous assistance in extending the functionality of his original code to integrate our desired specs. While confluentFUCCI normally takes care of segmentation through Cellpose, limited GPU compute forced us to substitute machine learning approaches for a simple binary masking method. These masks are then fed into TrackMate, the native confluentFUCCI stack, and our expanded utility functions to account for unmerged cell tracks which maintained a single color for the duration of the experiment.

3) After we generated .csv files for the tracking data, a separate plotting pipeline (graphs.py) was used to generate graphs for cell cycle durations, epithelial to mesenchymal phenotypes, migratory capacity, and cell cycle frequency across different drugs and concentrations.


Due to the computational resources required, we ran both the FUCCI analysis and graph generating pipelines as batch scripts (pscript.sh & gscript.sh) on the Tufts High Performance Computing Cluster (HPC). To implement this, we used Singularity to pull the confluentFUCCI container, created a Python shell and manually added its location in the Python interpreter path for our pipeline.py script. 

Goldstien, L., Lavi, Y., & Atia, L. (2024). ConfluentFUCCI for fully-automated analysis of cell-cycle progression in a highly dense collective of migrating cells. Plos one, 19(6), e0305491. 

