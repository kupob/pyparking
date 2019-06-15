
from urllib.request import urlopen
from bs4 import BeautifulSoup
import random
import numpy as np
import cv2
import re

class ImageGrabber:
    imageUrl = ''

    def __init__(self, url):
        html = urlopen(url)
        soup = BeautifulSoup(html, "html.parser")

        for script in soup(["script", "style"]):
            if script.contents:
                scriptContent = script.contents[0]
                src = re.search("http(.*)jpg", scriptContent)
                if src:
                    self.imageUrl = src.group(0)
                    break

    def getImage(self):
        resp = urlopen(self.imageUrl + '?math=' + str(random.random()))
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image