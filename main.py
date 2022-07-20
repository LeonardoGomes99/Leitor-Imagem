import cv2
import pytesseract
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Leo\\Documents\\pytesseract\\tesseract.exe'

img = cv2.imread('cupom.png')

#UPSCALE IMAGE - GET DATASET TO USE AS BASE
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "ESPCN_x3.pb"
sr.readModel(path)
sr.setModel("espcn",3)

#UPSCALE IMAGE - RESIZE TO UPSCALER
resized = cv2.resize(img,dsize=None,fx=3,fy=3)

#UPSCALE IMAGE - UPSCALE IMAGE HERE
result = sr.upsample(resized)

#UPSCALE IMAGE - SAVE RESULT AS CUPOM_UPSCALED
cv2.imwrite('cupom_upscaled.png', result)

#IMAGE TO TEXT
text = pytesseract.image_to_string(img)
print(text)
