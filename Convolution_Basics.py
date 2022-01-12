import sys #system import
import numpy as np #for handling arrays
import matplotlib.pyplot as plt #for plotting images
import matplotlib.image as mpimg #for reading images

DIR = #Enter file location of image to be processed here

def pad(img, num): #adds layers of zeros so that filter can detect border features
    w = img.shape[0] #image width
    h = img.shape[1] #image height
    output = np.zeros((w+num*2, h+num*2)) #output array, initialized to zeros
    
    for r in range(h+2):
        for c in range(w+2):
            if r<num or c<num or r>h+num-1 or c>w+num-1: 
                output[c,r] = 0 #add a zero to the first num and last num rows/columns
            else:
                output[c,r] = img[c-num,r-num] #everything else is the same
    
    return output

def ReLU(img): #helps with image output - turns negative numbers into zeros
    w = img.shape[0] #image width
    h = img.shape[1] #image height
    output = np.zeros((w, h)) #output array, initialized to zeros
    
    for r in range(h):
        for c in range(w):
            output[c,r]= max(img[c,r], 0) #max between current value and 0 (0>negative number)
    
    return output

def maxPool(img, sz=2, str=2): #max pooling function for downsizing; sz = window size; str = stride
    w = img.shape[0] #image width
    h = img.shape[1] #image height
    output = np.zeros((int((w-sz+1)/str), int((h-sz+1)/str))) #output array, initialized to zeros
    if output.shape[0] < 1: #make sure we're not going to get rid of the image
        print("image is not big enough to pool with given pool size and stride")
        return img
    outR = 0 #for keeping track of row on output
    for r in np.arange(0, h-sz-1, str): #rows from 0 to h-sz-1 with increments of str
        outC = 0 #for keeping track of column on output
        for c in np.arange(0, w-sz-1, str): #columns from 0 to w-sz-1 with increments of str
            output[outC, outR] = np.max(img[c:c+sz, r:r+sz]) #maximum from the current window
            outC += 1
        outR += 1
        
    return output

def invertImg(img):
    for r in range(img.shape[1]): #moving down the image
        for c in range(img.shape[0]): #moving across the image
            img[c,r] = 1-img[c,r] #swap the large and small values (all are in range [0-1]) 
    return img

def convolution(img, filt, s=1): #convolution function (feature detection)
    img = pad(img, int((filt.shape[1]-1)/2)) #pad the image with the above pad function
    w = img.shape[0] #image width
    h = img.shape[1] #image height
    numFilts = filt.shape[0] #number of filters we are given
    filtDim = filt.shape[1] #dimensions of the filters
    output = np.zeros((w-int((filtDim-1)/2), h-int((filtDim-1)/2))) #output array init to zeros
    
    #Prerequisites for the convolution function to work correctly
    if(filt.shape[1] != filt.shape[2]): #filter width and height must be the same
        print("Width and height of filter aren't equal")
        sys.exit()
    if(filtDim%2==0): #there is no center pixel in the filter if it does not have odd dimensions
        print("Filter must have odd dimensions")
        sys.exit()
    
    for f in range(numFilts): #for each of the filters
        currF = filt[f, :] #get the current filter
        for r in range(h-filtDim-1): #move the filter vertically down the image
            for c in range(w-filtDim-1): #move the filter horizontally across the image
                output[c,r] = np.sum(img[c:c+filtDim, r:r+filtDim]*currF[:,:]) + output[c,r]
                '''For the above line:
                   img[c:c+filtDim, r:r+filtDim]*currF[:,:] multiplies the current filter with
                   the image area and then np.sum sums up the resulting array.
                   Since there could be more than one filter, we add up the values from all
                   the filters to get the combined result.
                '''
    return output

def doConv(img, imgNum, filtMd, filtLg): #different iterations of the program
    if(imgNum-1 in range(4)):
        print("Using function number",imgNum)
    else:
        print("Function number not defined")
        sys.exit() 
        
    if(imgNum == 1):                    #Image 1 Start
        print("convolution layer 1")
        for j in range(2):                  #2 convolutions with large filters
            print("Large running")
            img = convolution(img, filtLg)
        for j in range(2):                  #2 convolutions with medium filters
            print("Medium running")
            img = convolution(img, filtMd)
        print("Pooling")
        img = maxPool(img)                  #Max pool
        print("ReLU")
        img = ReLU(img)                     #ReLU
        
        print("convolution layer 2")
        for j in range(2):                  #2 convolutions with large filters
            print("Large running")
            img = convolution(img, filtLg)
        for j in range(2):                  #2 convolutions with medium filters
            print("Medium running")
            img = convolution(img, filtMd)
        print("Single Convolution - Large")
        img = convolution(img, filtLg)      #1 convolution with large filters
        print("Pooling")
        img = maxPool(img)                  #Max pool
        print("ReLU")
        img = ReLU(img)                     #ReLU
                                        #End Image 1

    elif(imgNum == 2):                  #Image 2 Start
        for i in range(3):                   #3x loop Start
            print("convolution layer", i+1)
            print("Large running")
            img = convolution(img, filtLg)      #1 convolution with large filters
            print("Medium running")
            img = convolution(img, filtMd)      #1 convolution with medium filters
        print("Pooling")
        img = maxPool(img)                      #Max pool
                                             #End 3x loop
        for i in range(2):                   #2x loop Start
            print("convolution layer", i+4)
            print("Large running")
            img = convolution(img, filtLg)      #1 convolution with large filters
            print("Medium running")
            img = convolution(img, filtMd)      #1 convolution with medium filters
        print("Pooling")
        img = maxPool(img)                      #Max pool
                                            #End 2x Loop
                                        #End Image 2

    elif(imgNum == 3):                  #Image 3 Start
        img = invertImg(img)                #re-invert image because of the output of one layer
                
        diagonals = np.zeros((2, 5, 5))     #new array for filters for diagonals
        diagonals[0, :, :] = np.array([[[  1, .25,  -1,  -1,  -1],
                                     [.25,   1, .25,  -1,  -1],
                                     [ -1, .25,   1, .25,  -1],
                                     [ -1,  -1, .25,   1, .25],
                                     [ -1,  -1,  -1, .25,   1]]])
        for y in range(5):
            for x in range(5):
                diagonals[1, x, y] = diagonals[0, y, 4-x]
                
        print("Single Convolution - Large")
        img = convolution(img, diagonals)   #1 convolution with large filters
                                        #End Image 3

    elif(imgNum == 4):                  #Image 4 Start
        for i in range(2):                  #2x Loop Start
            print("convolution layer",i+1)
            for j in range(2):                  #2 convolutions with large filters
                print("Large running")
                img = convolution(img, filtLg)
            for j in range(2):                  #2 convolutions with medium filters
                print("Medium running")
                img = convolution(img, filtMd)
            print("Pooling")
            img = maxPool(img)                  #Max pool
                                            #End 2x Loop
        print("ReLU")
        img = ReLU(img)                     #ReLU
        print("convolution layer 3")
        for j in range(2):                  #2 convolutions with large filters
            print("Large running")
            img = convolution(img, filtLg)
        for j in range(2):                  #2 convolutions with medium filters
            print("Medium running")
            img = convolution(img, filtMd)
        print("Pooling")
        img = maxPool(img)                  #Max pool
        print("Single Convolution - Large")
        img = convolution(img, filtLg)      #1 convolution with large filters
        print("Pooling")
        img = maxPool(img)                  #Max pool
        print("ReLU")
        img = ReLU(img)                     #ReLU
    
    
    return img

def setFilt(size):
    if(size == "Md"):
        filt = np.zeros((4, 3, 3)) #array for the medium filters (straight lines)
        filt[0, :,:] = np.array([[[ -1, .25, 1],  #vertical line
                                [ -1, .25, 1],
                                [ -1, .25, 1]]])
        filt[1,:,:] = np.array([[[  1,   1,   1], #horizontal line
                                   [.25, .25, .25],
                                   [ -1,  -1,  -1]]])
        
        filt[2,:,:] = np.array([[[  1, .25,  -1], #diagonal line (negative slope)
                                   [.25,   1, .25],
                                   [ -1, .25,   1]]])
        filt[3,:,:] = np.array([[[ -1, .25,   1], #diagonal line (positive slope)
                                   [.25,   1, .25],
                                   [  1, .25,  -1]]])
    elif(size == "Lg"):
        filt = np.zeros((6, 5, 5)) #array for large filters (curves and long diagonals)
        filt[0, :, :] = np.array([[[  1, .25,  -1,  -1,  -1], #long diagonal line (negative slope)
                                     [.25,   1, .25,  -1,  -1],
                                     [ -1, .25,   1, .25,  -1],
                                     [ -1,  -1, .25,   1, .25],
                                     [ -1,  -1,  -1, .25,   1]]])
    
        for y in range(5): 
            for x in range(5):
                filt[1, x, y] = filt[0, y, 4-x] #long diagonal line (positive slope)
        
        filt[2, :, :] = np.array([[[  1,   1, .25, .10,  -1], #curve (from top, right then down)
                                     [.25, .25,   1, .25, .10],
                                     [.10, .10, .25,   1, .25],
                                     [ -1, .05, .10, .25,   1],
                                     [ -1,  -1, .10, .25,   1]]])
        
        for y in range(5):
            for x in range(5):
                filt[3, x, y] = filt[2, y, 4-x] #curve (from bottom, up then right)
                
        for y in range(5):
            for x in range(5):
                filt[4, x, y] = filt[2, 4-y, x] #curve (from bottom, right then up)  
                
        for y in range(5):
            for x in range(5):
                filt[5, x, y] = filt[4, 4-y, x] #curve (from top, down then right)
    else:
        print("Filter Size not Defined")
        sys.exit()
    
    return filt

def comparisonPlot(img, n1, n2, n3, n4): #plots a side-by-side of four different outputs
    fig, axarr = plt.subplots(2, 2) #2x2 array of images to be displayed
    img1 = img2 = img3 = img4 = img #avoid interference between algorithms
    axarr[0, 0].imshow(doConv(img1, n1, setFilt("Md"), setFilt("Lg")), "gist_heat") #plot image 1
    axarr[0, 0].set_title('Image 1') #label image 1
    axarr[0, 1].imshow(doConv(img2, n2, setFilt("Md"), setFilt("Lg")), "gist_heat") #plot image 2
    axarr[0, 1].set_title('Image 2') #label image 2
    axarr[1, 0].imshow(doConv(img3, n3, setFilt("Md"), setFilt("Lg")), "gist_heat") #plot image 3
    axarr[1, 0].set_title('Image 3') #label image 3
    axarr[1, 1].imshow(doConv(img4, n4, setFilt("Md"), setFilt("Lg")), "gist_heat") #plot image 4
    axarr[1, 1].set_title('Image 4') #label image 4
    fig.subplots_adjust(hspace=0.5) #adjust the space beteen the images
    plt.show() #display the images

def main(): #master function to manage subfunctions above, read, and write
    #read in the original image
    originalImg = mpimg.imread(DIR)
    plt.imshow(originalImg) #plots the original image

    img = originalImg[:, :, 0] #gets one layer of the image since it is black and white
    img = invertImg(img) #inverts image so black is large numbers and white is small numbers
    img = doConv(img, 1, setFilt("Md"), setFilt("Lg")) #does the pre-defined functions 
                                                       #currently in range [1,4]
    plt.imshow(img, "gist_heat") #plots the feature-detection image with red filter
    comparisonPlot(img, 1, 2, 3, 4) #side-by-side comparison of four images with red filter
    
main() #runs the main function
