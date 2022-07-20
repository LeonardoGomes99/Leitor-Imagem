import cv2
import pytesseract
# import matplotlib.pyplot as plt
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Leo\\Documents\\pytesseract\\tesseract.exe'
img = cv2.imread('cupom.png')

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "LapSRN_x8.pb"
sr.readModel(path)
sr.setModel("lapsrn",8)

resized = cv2.resize(img,dsize=None,fx=3,fy=3)

result = sr.upsample(img)

cv2.imwrite('cupom_upscaled_xLapSRN_X8.png', result)




#FIX SHAPNESS
#input
# read image as grayscale
img = cv2.imread('cupom_upscaled_xLapSRN_X8.png',0)

# threshold to white only values above 127
img_thresh = img
img_thresh[ img > 255 ] = 255

# view result
# cv2.imshow("threshold", img_thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# save result
cv2.imwrite("cupom_upscaled_xLapSRN_X8-Fixed.png", img_thresh)

# leitura da imagem
text = pytesseract.image_to_string(img)
print(text)
