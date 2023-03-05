# Homer Challenge

My contribution to the ICFHR2022 Competition on Detection and Recognition of Greek Letters on Papyri.

Based on the dataset available under 
https://faubox.rrze.uni-erlangen.de/getlink/fi9GsXoxUHagRAjApzgk7ywo/


## Datasets
The datasets are set up initially via the script dataset.py
They compromise of the following sets:
* _ICFHR2022_train_ - Original Papyri dataset for training
* _ICFHR2022_test_ - First test dataset provided for the competition
* _ICFHR2022_artificial_ - Dataset containing the synthetic papyri created by Google Fonts, ancient greek texts, 
some textures and PIL. Has to be generated first.

**IMPORTANT:** First execute dataset.py, afterwards CreateArtificialPapyri, so the datasets are setup in a correct manner.

### Dataset inspection
Inspect the dataset via a webbrowser (by help of the fiftyone package)
just call the script "Inspect_Ground_Truth.py" via command line 

### Introduction to the GAN approach: 

https://medium.com/ixorthink/realistic-document-generation-using-generative-adversarial-networks-37d8188f0e5c 

https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Letters are created with a GAN and inserted on a Document
