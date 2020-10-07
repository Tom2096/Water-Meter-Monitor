Water-Meter-Monitor
===================
## Overview
This is a program developed in **Python** using **OpenCV** and **Numpy**. It can take a snapshot of a water meter and read the value indicated by the arm of the odometer.

## Development
A sample picture of the water meter is taken, with the hundredths reading indicated by the red arm of the odometer. 

![Inital Img of water meter](https://github.com/Tom2096/Water-Meter-Monitor/blob/main/Imgs/image.png)

Then, three points (in red) pinpointing the center, the y-axis, and the x-axis are manually plotted. These points will be used as a frame for all future pictures. 

![Three points plotted](https://github.com/Tom2096/Water-Meter-Monitor/blob/main/Imgs/Figure_2.png)

The vertices of the box bounding the odometer is then created by adding and subtracting the length of the x-axis and y-axis. 

![clip bounding box](https://github.com/Tom2096/Water-Meter-Monitor/blob/main/Imgs/bbx.png)

Then, the picture is read and converted into HSV color format to filter out the red arm of the odometer. 

![Filtered Image](https://github.com/Tom2096/Water-Meter-Monitor/blob/main/Imgs/Figure_3.png)

The image is then denoised by first eroding to rid the image of outlying pixels, then dilating to fill in any holes in the middle. Finally, the image is eroded again to trim the edges. 

![Denoised Image](https://github.com/Tom2096/Water-Meter-Monitor/blob/main/Imgs/Figure_4.png)

The eigenvalues and right eigenvectors are then computed to determine the direction of the arm

The eigenvector corresponding  with the largest eigenvalue is then projected onto the x and y axis, which can be calculated to give the angle of rotation and the odometer reading.   
![Denoised Image](https://github.com/Tom2096/Water-Meter-Monitor/blob/main/Imgs/Figure_5.png)

Overall setup:
![Denoised Image](https://github.com/Tom2096/Water-Meter-Monitor/blob/main/Imgs/setup.jpg)

