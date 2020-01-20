# Doodle to UI Dataset
Doodle to UI Dataset contains 11 thousand drawings from 16 categories. 

[RICO](http://interactionmining.org/rico) by Nick Jonas dataset is a collection of 66k unique UI screens from 9.3k apps from 27 Google Play app categories. According to RICO the most common Android UI element types are- 
Container, followed by (in order) image, # icon (a small interactive image), text, text button, web view, input, list item, switch (a toggle element), map view,slider, and checkbox. Among the icons most common elements are back, followed by menu (the hamburger), cancel (close), search (loupe), plus (add), avatar (user image), home (house), share, settings (gear), star (rating), edit, more, refresh, and forward.

The [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) contains 345~sketch categories (from ``aircraft carrier'' to ``zigzag''), with some 100k samples each, drawn by anonymous users. QuickDraw contains some categories that seemed like a good fit for UI Element but fails to represent most common UI Elements. Doodle to UI Dataset contains 16 categories that can be used to represent these common UI element. Preview of these 16 categories is shown here- 

![preview](SampleElements.png)


With the advancement of the deep neural network, it is now possible to train a network that can recognize sketches from the order of the strokes.  Most of the designers still prefer simple art supplies to design UI. We are sharing this dataset for the researchers who are enthusiastic about automize this UI generation process. An experiment is shown in the [Doodle to UI](http://pixeltoapp.com/doodle/). This data still contain some inappropriate drawing made by the users. 


## Data Format
Strokes of these 16 categories reside in different folders. We used the same formatting used by Quickdraw.  Each drawing comprises several strokes. A single stroke symbolizes drawing made by a user without detaching a drawing pen from the digital interface.   And each stroke is a collection of straight lines, given by their x/y endpoints coordinates. 
[ 
  [  // First stroke 
    [x0, x1, x2, x3, ...],
    [y0, y1, y2, y3, ...],
  ],
  [  // Second stroke
    [x0, x1, x2, x3, ...],
    [y0, y1, y2, y3, ...],
  ],
]

Here x0,x1 indicates the coordinates of the digital interface. 


## Content

[data.zip](data.zip)  - Contains 11 thousand drawings from 16 categories separated by folder of their corresponding categories.  
PrepareData.py - This python script helps to separate the data into the test and train category. 
StrokeToImage.py - This script file converts the strokes into an image.
qlabel.txt - Text file contains categories that are going to be included for training. 
tfGenerator.py - Script file that converts strokes to tfrecord. (the File format used by TensorFlow)
trainModel.py - A script to train an RNN model that uses the tfrecord file.


## Dependencies


## License
This data made available by Google, Inc. under the [Creative Commons Attribution 4.0 International license.](https://creativecommons.org/licenses/by/4.0/)



 
