import cv2
import numpy as np

def alignImages(imgRef, img):
    """
    PROCESS OF THE ALIGNMENT
    1) Change the images to GrayScale color
    2) Detect the features of each image by using ORB
    3) Match the features of each image
    4) Find the Homography matrix
    5) Align the photo
    """

    # Changing the color
    imgRefGray = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("refImage", imgRefGray)
    cv2.imshow("photo", imgGray)
    cv2.waitKey(0)

    # Detect features by ORB
    orb = cv2.ORB_create()
    keypoints_ref, descriptors_ref = orb.detectAndCompute(imgRefGray, None)
    keypoints_img, descriptors_img = orb.detectAndCompute(imgGray, None)

    # Match the features of each image
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(descriptors_ref, descriptors_img)
    matches = sorted(matches, key = lambda x: x.distance)
    matches = matches[:200]
    
    pointsRef = np.zeros((len(matches), 2))
    pointsImg = np.zeros((len(matches), 2))
    for i, m in enumerate(matches):
        pointsRef[i, :] = keypoints_ref[m.queryIdx].pt
        pointsImg[i, :] = keypoints_img[m.trainIdx].pt
    
    img_draw = cv2.drawMatches(
        imgRef, keypoints_ref, 
        img, keypoints_img, 
        matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("img_draw", img_draw)
    cv2.waitKey(0)

    # Find Homography matrix
    homography, mask = cv2.findHomography(pointsImg, pointsRef, cv2.RANSAC)

    # Align the photo
    height, width, channel = imgRef.shape
    imgAligned = cv2.warpPerspective(img, homography, (width, height))

    return imgAligned



if __name__ == "__main__":
    # Reading the images
    imColor = cv2.IMREAD_COLOR
    refImagePath = "form2.jpg"
    photoPath = "photo2.jpg"
    photo = cv2.imread(photoPath, imColor)
    refImage = cv2.imread(refImagePath, imColor)

    # Printing the images
    cv2.imshow("refImage", refImage)
    cv2.imshow("photo", photo)
    cv2.waitKey(0)

    # Aligning the photo
    photoAligned = alignImages(refImage, photo)

    cv2.imshow("Image Aligned", photoAligned)
    cv2.waitKey(0)