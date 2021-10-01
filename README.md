# Burn-Degree-Analysis
This involves analyzing the degree of a burn (1st, 2nd or 3rd degree) using multisimensional scaling (MDS) and a kNN classifier. A major reference for this project is "Burn Depth Analysis using Multidimensional Scaling applied to psychophysical experiment data" by Acha et al. The link to this paper is provided here: https://ieeexplore.ieee.org/document/6488854

The process flow is as follows:

1)Get a dataset of burn images. (The dataset for this project was obtained from the following sources: (i) Dr Mathivanan of Sri Venkataeswara Hospitals, Chennai  (ii) Medetec Wound Database, UK  (iii)Shriner Hospital, Cincinatti, USA).

2)Get the degree of burn in each image certified by a doctor (The certification of images for our project was done by Dr Suchithra Rajmohan, Sara skin clinic, Chennai)

3)Perform the training process:
  
   	3a)The doctor assigns a score from 0-10 corresponding to the similarity between each of the training images. with 0 meaning "most similar" and 10     equivalent to "most dissimilar". Therefore, if we have I training images, we get an IxI similarity matrix.
    
  ![Screenshot (329)](https://user-images.githubusercontent.com/70104287/135692580-ed8698a7-827f-440a-b563-d955e8376d30.png)

   	3b) This similarity matrix is given as input to the MDS algorithm which gives a "n"-Dimensional plot as output corresponding to the no of eigenvalues "n" that we choose. We chose n=2 and n=3 and obtained both 2D and 3D MDS output plots. The MDS output plots clusters the image points according to the degree of burn. We opt to go for 3D MDS since using 3 features could provide more accuracy than just 2 features.
    
  
   	3c) By analyzing the coordinates and by consulting our reference IEEE paper, the X,Y and Z coordinates are obtained as the amount of redness, texture and saturation respectively. These 3 parameters are then calculated for each of the training images.
  
   	3d) The KNN algorithm is trained using the training set to predict the degree of burn given the X,Y and Z coordinates as input.

![Screenshot (330)](https://user-images.githubusercontent.com/70104287/135692708-4ca7cba0-7b9d-4901-9920-89fa488ee261.png)

4) Perfom the testing process:
  
      4a)The test images are segmented and only the burn portion alone is obtained by using the Simple Linear Iterative Clustering (SLIC) algorithm.
      
  ![Screenshot (331)](https://user-images.githubusercontent.com/70104287/135692857-68656d82-0055-41bf-85fa-538f1b201e61.png)

      4b)The redness, texture and saturation of each of the segmented test images are determined and from these 3 parameters, the corresponding X, Y and Z coordinates are determined using a custom-created signum function. These obtained X, Y and Z values are then fed as input to the KNN classifier which gives the degree of the burn as output.
      
      4c) As an additional feature, the Total Burn Surface Area (TBSA) of the burn portion is also determined by using medical charts such as the Lund-Browder chart and Rule of Nines as reference.
  
