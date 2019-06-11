
from urllib.request import urlopen
# from bs4 import BeautifulSoup
import random
import numpy as np
import cv2

class ImageGrabber:
    def __init__(self):
        print ('hi')

    def getImage(self, url):
        # html = urlopen(url)
        # soup = BeautifulSoup(html, "html.parser")
        #
        # image = soup.find(alt="camera_image", src=True)
        # if image is None:
        #     print('No matching image found')
        #     return

        imageSrc = 'http://207.251.86.238/cctv458.jpg' + '?math=' + str(random.random())

        print(imageSrc)

        resp = urlopen(imageSrc)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image

        # image_link = image['src']
        # filename = image_link.split('/')[-1]
        #
        # print (image_link, filename)