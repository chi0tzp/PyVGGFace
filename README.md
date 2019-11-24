# PyVGGFace 

A [VGG-Face CNN descriptor](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) implemented in PyTorch. 

The VGG-Face CNN descriptors are computed using our  CNN implementation based on the VGG-Very-Deep-16 CNN architecture as  described in [1] and are evaluated on the Labeled Faces in the Wild [2] and the YouTube Faces  [3] dataset.



**Step 1:** Convert the original pre-trained LuaTorch weights into PyTorch VGGFace weights and save them at `models/vggface.pth`

~~~bash
python convert_weights.py 
~~~



**Step 2:** Run demo script:

~~~
python3 demo.py
~~~

Image `data/ak.png` is passed through pre-trained VGGFace and the prediction should be `2`, i.e., ??? 





In general, a VGGFace model can be build and loaded with pre-trained weights as follows:

~~~python

~~~





 







------

**References**

[1] Parkhi, Omkar M., Andrea Vedaldi, and Andrew Zisserman. "Deep face recognition." *BMVC*. Vol. 1. No. 3. 2015.

 [2] G. B. Huang, M. Ramesh, T. Berg, E. Learned-Miller Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, 2007.                   

 [3] L. Wolf, T. Hassner, I. Maoz "Face Recognition in Unconstrained Videos with Matched Background Similarity." Computer Vision and Pattern Recognition (CVPR), 2011.     

