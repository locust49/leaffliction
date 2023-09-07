from enum import Enum
import cv2
import matplotlib.pyplot as plt
import argparse
from leaffliction import configure_logger


# create an enum for the augmentation methods (rotate, flip, blur, contrast, scale and combine)
class MethodEnum(Enum):
	ORIGINAL = 0
	ROTATE = 1
	FLIP = 2
	BLUR = 3
	CONTRAST = 4
	SCALE = 5


class Augmentation:
	"""
	Class used to augment images
	"""
	def __init__(self, image_path, nargs=6, combine_methods=False):
		"""
		Constructor of the Augmentation class
		-> (initialize the image, title, extension, number of arguments and axes)
		:param image_path:
		:param nargs:
		:param combine_methods:
		"""
		self.image = cv2.imread(image_path)
		self.title = image_path.split('/')[-1].split('.')[0]
		self.extension = image_path.split('/')[-1].split('.')[1]
		self.nargs = nargs + (1 if combine_methods else 0)
		self.fig, self.axes = plt.subplots(1,  nargs + (1 if combine_methods else 0), figsize=(25, 10))

	def rotate(self, angle: int = 45, **kwargs):
		"""
		Rotate the image by the given angle
		:param angle: the angle to rotate the image
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the rotated image
		"""
		(h, w) = self.image.shape[:2]
		center = (w / 2, h / 2)
		matrix = cv2.getRotationMatrix2D(center, angle, 1)
		rotated = cv2.warpAffine(self.image, matrix, (w, h))
		return rotated

	def flip(self, **kwargs):
		"""
		Flip the given image around the x-axis
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the flipped image
		"""
		flipped = cv2.flip(self.image, 0)
		return flipped

	def blur(self, **kwargs):
		"""
		Blur the given image using a Gaussian filter
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the blurred image
		"""
		blurred = cv2.GaussianBlur(self.image, (7, 7), 0)
		return blurred

	def heighten_contrast(self, **kwargs):
		"""
		Raise the contrast of the given image
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the image with higher contrast
		"""
		higher_contrast = cv2.convertScaleAbs(self.image, alpha=1.5)
		return higher_contrast

	def scale(self, **kwargs):
		"""
		Scale the given image
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the scaled image
		"""
		scaled = cv2.resize(self.image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
		return scaled

	def plot_augmentation(self, axe, augmented_image, suffix):
		"""
		Plot the augmented image
		:param axe: axe where to plot the augmented_image
		:param augmented_image: image to be plotted
		:param suffix: augmentation method name to be added to the title
		:return: None
		"""
		if augmented_image is not None:
			axe.imshow(augmented_image)
			axe.set_title(f"{self.title}_{suffix}.{self.extension}")

	def apply_augmentation(self, method_enum, combine=False, **kwargs):
		"""
		Apply the specified augmentation method based on the MethodEnum enum.
		:param method_enum: the method specified using the MethodEnum enum
		:param combine: if True, combine the augmented images in a single image
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the augmented image
		"""
		if method_enum == MethodEnum.ORIGINAL.value:
			self.plot_augmentation(self.axes[MethodEnum.ORIGINAL.value], self.image, "original")
		elif method_enum == MethodEnum.ROTATE.value:
			rotated = self.rotate(**kwargs)
			self.plot_augmentation(self.axes[MethodEnum.ROTATE.value], rotated, "rotated")
		elif method_enum == MethodEnum.FLIP.value:
			flipped = self.flip(**kwargs)
			self.plot_augmentation(self.axes[MethodEnum.FLIP.value], flipped, "flipped")
		elif method_enum == MethodEnum.BLUR.value:
			blurred = self.blur(**kwargs)
			self.plot_augmentation(self.axes[MethodEnum.BLUR.value], blurred, "blurred")
		elif method_enum == MethodEnum.CONTRAST.value:
			raised_contrast = self.heighten_contrast(**kwargs)
			self.plot_augmentation(self.axes[MethodEnum.CONTRAST.value], raised_contrast, "raised_contrast")
		elif method_enum == MethodEnum.SCALE.value:
			scaled = self.scale()
			self.plot_augmentation(self.axes[MethodEnum.SCALE.value], scaled, "scaled")
		elif combine:
			combined = self.combine_all_methods_in_image()  # tbd
			self.plot_augmentation(self.axes[self.nargs - 1], combined, "combined")

	def combine_all_methods_in_image(self):
		"""
		Combine the augmented images in a single image
		:return: the combined image
		"""
		self.image = self.rotate()
		self.image = self.flip()
		self.image = self.blur()
		self.image = self.heighten_contrast()
		combined = self.scale()
		# TODO: add other combinations probably using enums (methods , number of methods)
		return combined


def configurate_parser():
	"""
	Parse the command line arguments
	:return: the parsed arguments
	"""
	_parser = argparse.ArgumentParser(description="Augment images of leaves dataset")
	_parser.add_argument("-f", "--files", nargs='+', help="image files to be augmented",
																						required=True)
	_parser.add_argument("-o", "--output", help="Output directory where to store the plots",
																						default="./augmented_directory")
	_parser.add_argument("-v", "--verbose", help="Enable verbose mode", action="store_true")
	_parser.add_argument("-n", "--number", help="Number of augmented images to be generated", default=6)
	_parser.add_argument("-c", "--combined", help="Combine the augmented images in a single image", action="store_true")
	return _parser


if __name__ == "__main__":
	parser = configurate_parser()
	args = parser.parse_args()
	logger = configure_logger(args.verbose)
	image = args.files[0]
	if args.number < 6:
		logger.error("Using default number of augmented images (6)")
		args.number = 6
	elif args.number > 10:  # TBD (depending on how many method we have)
		logger.error("Using maximum number of augmented images (10)")
		args.number = 10
	augmentation_instance = Augmentation(image, args.number, args.combined)
	augmentation_instance.fig.suptitle(f"{image.split('/')[-1]} augmentations")
	for method in range(0, args.number + (1 if args.combined else 0)):
		augmentation_instance.apply_augmentation(method, args.combined)
	plt.show()
