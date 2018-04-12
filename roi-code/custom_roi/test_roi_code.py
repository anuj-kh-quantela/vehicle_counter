import numpy as np
import cv2
import draw_custom_roi

def main():
	frame = cv2.imread('0.jpg')
	copy_frame = frame.copy()

	test = draw_custom_roi.define_roi(frame, copy_frame)
	print("selected coordinates: ")
	print(test)


if __name__ == "__main__":
	main()

