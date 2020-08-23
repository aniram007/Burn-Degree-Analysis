# import the necessary packages
from skimage.segmentation import slic
import numpy as np
import cv2
import time
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import math
import array as arr
from xassign import assignx
from yassign import assigny
from zassign import assignz
from mds_func import cmdscale
from ml_trial_3d import classify_3d
import subprocess as sp
sp.call('cls',shell=True)
numsegments=10
count=[]  #count of pixels in each segment
count_2=[]
order=[]#to arrange in descending order of segment size
red=0
redness=0
cnt=0
cnt2=0
pix=0
severity=0 #To check severity of burn
######## Age group Specification for determining body area (unit in m^2)####
a_men=1.9
a_women=1.6
teen=1.33
child=1.1
#################
####Body part specification for finding % of body area ####
hand=1.5
forearm=6
face=3.5
neck=2
torso=18
leg=9
#################
print("\t\t            BURN CLASSIFICATION SYSTEM \n\n                    ")
folder="test_images/"
im=input("Enter the image file to be analyzed: ")
im_format=input("Enter the format of the image(A/B/C): A)jpg B)jpeg C)png ")
if(im_format=='A' or im_format=='a'):
    im2=".jpg"
if(im_format=='B' or im_format=='b'):
    im2=".jpeg"
if(im_format=='C' or im_format=='c'):
    im2=".png"
im3=folder+im+im2
image=cv2.imread(im3)
#image=cv2.imread('F:/College/8th sem/project/burn images/9.jpg')
image2=image.copy()
#####finding size of image ####
h,w,c=image.shape
#print("No of pixels= ",h*w)
bg_sub=0.8*h*w   #assuming background occupies 20% of total image
###################

######### Assigning body surface area baed on age###
age=input("Patient is adult man(M)/adult woman (W)/teenager(T)/child (C)?")
part=input("Part of the body affected: Hand(H)/Forearm(B)/Face(F)/Neck(N)/Torso(T)/leg(L)?")
if(age=='M'):
    size=a_men
if(age=='W'):
    size=a_women
if(age=='T'):
    size=teen
if(age=='C'):
    size=child
if(part=='H'):
    ratio=hand
if(part=='B'):
    ratio=forearm
if(part=='F'):
    ratio=face
if(part=='N'):
    ratio=neck
if(part=='T'):
    ratio=torso
if(part=='L'):
    ratio=leg

area=(size*ratio)/100
############################
# Apply SLIC and extract (approximately) the supplied number of segments
segments = slic(img_as_float(image2), n_segments=numsegments, sigma = 5,convert2lab=True)  #'segments' gives label associated with eac pixel of the image
for var in np.unique(segments):
    y=np.count_nonzero(segments==var)
    order.append(var)
    count.append(y)

#count.sort(reverse=True)
#print("Original=",count)
count_2=count.copy()
#print("Original Order=",order)
#print("count2= ",count_2)
for i in range(0,len(count)):
    for j in range(i+1,len(count)):
        if count[i]<count[j]:
            temp=count[i]
            temp2=order[i]
            count[i]=count[j]
            order[i]=order[j]
            count[j]=temp;
            order[j]=temp2

#print("Sorted=",count)
#print("Sorted order=",order)
select=order #selecting largest 5 segments
#print("Selected segments:",select)
       
# show the output of SLIC
fig = plt.figure("Overall image with segments-CLOSE THE WINDOW TO CONTINUE")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
 
# show the plots
plt.show()
print("\n\nINSTRUCTIONS FOR THE DOCTOR")
print("1)THE SEGMENTS OF THE IMAGE WILL NOW APPEAR. CHOOSE THE APPROPRIATE SEGMENT")
print("2)THERE ARE TOTALLY %d SEGMENTS AVAILABLE NUMBERED 0 TO %d"%((len(np.unique(segments))),(len(np.unique(segments))-1)))
print("3)AFTER CHOOSING THE APPROPRIATE SEGMENT NUMBER, PRESS ANY KEY TO CLOSE ALL IMAGES")
key=input("\nPRESS ENTER ONCE YOU HAVE STUDIED THE INSTRUCTIONS")
print("THE SEGMENTS WILL NOW LOAD IN ANOTHER 5 SECONDS. THE LEFT IMAGE WILL BE THE ORIGINAL IMAGE AND THE RIGHT IMAGE WILL BE THE SEGMENTED PART")
time.sleep(5)
#cv2.imshow("ORIGINAL IMAGE",cv2.resize(mark_boundaries(image2,segments),(512,256)))
image2=cv2.resize(image2,(512,256))
# loop over the unique segment values
for (i, segVal) in enumerate(np.unique(segments)):
#for (i,segVal) in enumerate(select):
        # construct a mask for the segment
        #print ("[x] inspecting segment %d" %(segVal))
        #print("Segval=",segVal)
        mask = np.zeros(image.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 255
        # show the masked region
        #cv2.imshow("Mask", mask)
        applied=cv2.bitwise_and(image, image, mask = mask)
        applied=cv2.resize(applied,(512,256))
        #cv2.imshow("Segmented part %d"%(segVal), applied)
        img_concatenate=np.concatenate((image2,applied),axis=1)
        cv2.imshow("Segmented part %d"%(segVal),img_concatenate)
        
        #cv2.waitKey(0)
        
        #time.sleep(2)
        #save=input("Do you want to save this image? (y/n)")
        #if(save=='y'):
        #cv2.imwrite('F:/College/8th sem/project/dataset/3rd degree seg/medetec_1_seg_%d.jpg'%segVal,applied)
        #tbsa=(count[i]*area*100*100)/bg_sub #Using proportionality concept. multiply by 100*100 to convert into cm^2
        #print("TBSA=",tbsa," cm^2")
#print("ONCE YOU HAVE CHOSEN A SEGMENT, PRESS ANY KEY TO CONTINUE: ")
cv2.waitKey(0)
cv2.destroyAllWindows()
###### Section for determination of coordinates #######
      
image3=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image4=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
image5=image.copy()
x=input("Enter the segment no. you want to choose: ")
x=int(x)
tbsa=(count_2[x]*area*100*100)/bg_sub   #Using proportionality concept. multiply by 100*100 to convert into cm^2
print("Approximate Total Burn Surface Area=",tbsa," cm^2")
tbsa_ratio=tbsa/(size*100*100)
for i in range(0,h):
    for j in range(0,w):
        if(segments[i,j]==x):
            pix=pix+image3[i,j]
            if(image[i,j,2]>image[i,j,1] and image[i,j,2]>image[i,j,0] and image[i,j,2]>200):
                red=red+1
#print("No of red pixels= ",red)
redness=red/count_2[x]
redness=round(redness,3)
mean=pix/count_2[x]
#print("Normalized redness= ",redness)
#print("Mean pixel value= ",mean)
for i in range(0,h):
    for j in range(0,w):
        if(segments[i,j]==x):
         if(image3[i,j]>mean+30 or image3[i,j]<mean-30):
            cnt=cnt+1

texture=cnt/count_2[x]
#texture=round(texture,3)
#print("No. of pixels deviating from mean value= ",texture)

for i in range(0,h):
    for j in range(0,w):
        if(segments[i,j]==x):
            cnt2=cnt2+(image5[i,j,1]/255)
            
saturation=cnt2/count_2[x]
#print("Normalized saturation= ",saturation)

xco=assignx(redness)
yco=assigny(texture)
zco=assignz(saturation)
#print("Estimated X-coordinate= ",xco)
#print("Estimated Y-coordinate= ",yco)
#print("Estimated Z-coordinate= ",zco)
##############################################

##############Section for combining MDS analysis#########
D=np.array([[0,10,8,5,10,10,4,2,9,10,8,9,6,7,9,2,4,6,9,10],
            [10,0,1,7,2,3,9,6,4,3,9,10,7,9,2,3,4,8,8,10],
            [8,1,0,6,3,4,9,4,1,2,9,10,6,8,3,7,5,9,9,10],
            [5,7,6,0,10,10,1,3,8,10,3,3,5,1,7,2,4,2,2,1],
            [10,2,3,10,0,0.5,9,8,4,2,9,10,6,8,3,7,8,8,9,10],
            [10,3,4,10,0.5,0,10,7,2,2,8,9,5,7,3,5,6,8,9,10],
            [4,9,9,1,9,10,0,4,9,10,1,1,4,1,7,7,5,2,1,1],
            [2,6,4,3,8,7,4,0,6,8,4,5,7,2,6,5,4,3,4,9],
            [9,4,1,8,4,2,9,6,0,1,9,10,6,8,3,3,4,7,8,9],
            [10,3,2,10,2,2,10,8,1,0,9,9,8,9,3,6,7,8,9,10],
            [8,9,9,3,9,8,1,4,9,9,0,1,6,2,7,7,8,3,0.5,1],
            [9,10,10,3,10,9,1,5,10,9,1,0,8,3,7,8,8,3,2,1],
            [6,7,6,5,6,5,4,7,6,8,6,8,0,6,7,3,4,3,6,8],
            [7,9,8,1,8,7,1,2,8,9,2,3,6,0,8,7,6,3,2,2],
            [9,2,3,7,3,3,7,6,3,3,7,7,7,8,0,3,4.5,8,8.5,9],
            [2,3,7,2,7,5,7,5,3,6,7,8,3,7,3,0,6,4,8,9],
            [4,4,5,4,8,6,5,4,4,7,8,8,4,6,4.5,6,0,9,7,9],
            [6,8,9,2,8,8,2,3,7,8,3,3,3,3,8,4,9,0,6,9],
            [9,8,9,2,9,9,1,4,8,9,0.5,2,6,2,8.5,8,7,6,0,3],
            [10,10,10,1,10,10,1,9,9,10,1,1,8,2,9,9,9,9,3,0]])
n=len(D)
num=-1
xcoord=[]
ycoord=[]
zcoord=[]
Y,evals=cmdscale(D)
for i in range(n):
    for j in range(3):
        num=num+1
        if num%3==0:
            xcoord.append(Y[i][j])
        if num%3==1:
            ycoord.append(Y[i][j])
        if num%3==2:
            zcoord.append(Y[i][j])
#print("X coordinates= ",xcoord)
#print(len(xcoord))
#print("Y coordinates= ",ycoord)
#print(len(ycoord))
#print("Z coordinates= ",zcoord)
#print(len(zcoord))
n2=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
n3=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
for i in range(n):
    ax.scatter(xcoord[i],ycoord[i],zcoord[i])  #for 3D plot
    ax.text(xcoord[i],ycoord[i],zcoord[i],'%s'%(str(n2[i])),size=10,zorder=1,color='k')
    
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
#plt.show()

point=[xco,yco,zco]
result=classify_3d(xco,yco,zco)
if(result==1):
    print("\nITS A FIRST DEGREE BURN!")
    print("\nMEDICAL INSTRUCTIONS:")
    print("\n1. Apply cold water to the burn")
    print("\n2. Cover the burn with a sterile bandage")
    print("\n3. Prescribe burn ointment/gel for the patient")
    print("\n4. Prescribe painkiller if required")
    print("TBSA ratio: ",tbsa_ratio)
elif(result==2):
    print("\nITS A SECOND DEGREE BURN!")
    print("\nMEDICAL INSTRUCTIONS:")
    print("\n1. Apply cold water to the burn")
    print("\n2. Cover the burn with a gauze or loose dressing")
    print("\n3. Clean the burn and apply")
    print("\n4. Prescribe painkiller if required")
    print("\nTBSA ratio: ",tbsa_ratio)
    if((age=='M' and tbsa_ratio>=0.2) or (age=='W'and tbsa_ratio>=0.2) or (age=='T' and tbsa_ratio>=0.1) or (age=='C' and tbsa_ratio>=0.05)):
        severity=1
    if(severity==1):
        print("\nSEVERE BURN! PATIENT NEEDS TO BE ADMITTED IN THE HSPITAL AND MONITORED. SKIN GRAFTING HAS TO BE PERFORMED")
    else:
          print("\nHOSPITAL ADMISSION NOT NECESSARY. PRESCRIBE ANTIBOTICS")
    
else:
    print("\nITS A THIRD DEGREE BURN!")
    print("\nMEDICAL INSTRUCTIONS:")
    print("\n1. Remove dead skin and tissue from the burn area")
    print("\n2. Intravenous fluids containing electrolytes needs to be given")
    print("\n3. Prescribe antibotics to prevent infection") 
    print("\n4. Prescribe painkiller if required")
    print("\nTBSA ratio: ",tbsa_ratio)
    if(tbsa_ratio>=0.05):
        severity=1
    if(severity==1):
        print("\nSEVERE BURN! PATIENT NEEDS TO BE ADMITTED IN THE HSPITAL AND MONITORED. SKIN GRAFTING HAS TO BE PERFORMED")
    else:
          print("\nHOSPITAL ADMISSION NOT NECESSARY")
    


    
    

"""
####KNN combining portion####
point=[xco,yco,zco]#for 3D
deg=[2,3,3,2,3,3,1,2,3,3,1,1,2,1,3,2,2,2,1,1]
values=[]
distance=[]  #for Euclidean distance
result=[0,0,0]
for i in range(len(deg)):
    values.append([xcoord[i],ycoord[i],zcoord[i],deg[i]]) #for 3D
print("Appended values= ",values)
for value in values:
    # Print each row's length and its elements.
    #print( value)
    d= math.sqrt( ((value[0]-point[0])**2)+((value[1]-point[1])**2)+((value[2]-point[2])**2)) #for 3D
    value.append(d)
    distance.append(value)

distance = sorted(distance, key=lambda x: x[4]) #for 3D
print("Sorted distance matrix= ",distance)
for x in range(3):
    temp=distance[x]
    temp1=temp[3] #for 3D
    if temp1==1 :
        result[0]=result[0]+1
    elif temp1==2 :
        result[1]=result[1]+1
    else :
        result[2]=result[2]+1
        
for x in range(3):
    if result[x] >1 :
        if x==0 :
            print("first degree")
        elif x==1 :
            print("second degree")
        else :
            print("third degree")
"""
    
xcoord.append(point[0])
ycoord.append(point[1])
zcoord.append(point[2])
fig2=plt.figure()
ay=fig2.add_subplot(111,projection='3d')
for i in range(n+1):
    ay.scatter(xcoord[i],ycoord[i],zcoord[i])           #for 3D plot
    ay.text(xcoord[i],ycoord[i],zcoord[i],'%s'%(str(n3[i])),size=7,zorder=1,color='k')   #for labelling the point
#for i,txt in enumerate(n2):
  # plt.annotate(txt,(xcoord[i],ycoord[i],zcoord[i]))
    #plt.annotate(txt,(xcoord[i],ycoord[i]))
ay.set_xlabel('X-axis')
ay.set_ylabel('Y-axis')
ay.set_zlabel('Z-axis')
#plt.show()

q=input("Do you want to quit? Press Enter to quit ")

    
   
