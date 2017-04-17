# Advanced (curved) line tracking - Peter Tempfli

Please see the the final video here:

[![video](http://img.youtube.com/vi/Xi9nykQUFHE/0.jpg)](https://www.youtube.com/watch?v=Xi9nykQUFHE)

Please see the [jupyter notebook here](./adv_lane_finding.ipynb).

## Distortion correction

I'm using the following function, using the previously _mtx_ coefficients.

```
def undistort(image):
    return cv2.undistort(image, mtx, dist, None, mtx)
```
![distortion fix](./output_images/distorted.png)


