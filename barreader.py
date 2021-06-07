import cv2
import numpy as np
from pyzbar.pyzbar import decode

def read(img):
    # Decodes the barcode image
    detectedBarcodes = decode(img)
       
    # Traverses through all the detected barcodes in image
    if not detectedBarcodes:
        print("Barcode Not Detected or your barcode is blank/corrupted!")
        return None, None
    else:
        for barcode in detectedBarcodes:  
            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect
            # Put the rectangle in image using 
            # cv2 to heighlight the barcode
            cv2.rectangle(img, (x-10, y-10),
                          (x + w+10, y + h+10), 
                          (255, 0, 0), 2)
              
            # Print the barcode data
            if barcode.data != "":
                print(barcode.data)
                print(barcode.type)
                  
    #Display the image
    cv2.imshow("Bar Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (barcode.data, barcode.rect)