from enum import Enum
import os
import cv2
import matplotlib.pyplot as plt
import argparse
from leaffliction import configure_logger
from dotenv import load_dotenv
import numpy as np


class MethodEnum(Enum):
	ORIGINAL = 0
	ROTATE = 1
	FLIP = 2
	BLUR = 3
	CONTRAST = 4
	SCALE = 5
	SHEAR = 6


class Augmentation:
	"""
	Class used to augment images
	"""
	def __init__(self, parsed_args):
		"""
		Constructor of the Augmentation class
		-> (initialize the images, input directory, titles, extensions, number of arguments and axes)
		:param parsed_args: the command line parsed arguments
		"""
		all_images = parsed_args.files
		self.images = [img for img in all_images if img is not None and os.path.exists(img)]
		images_filename = [path.replace('\\', '/').split('/')[-1] for path in self.images]
		n_rows = min(len(self.images), os.getenv("MAX_NUMBER_OF_ROWS", 10)) if parsed_args.directory else len(self.images)
		self.input_directory = [
			"/".join(path.replace('\\', '/').split('/')[:-1]) if '/' in path else "." for path in self.images]
		self.title = [filename.split('.')[0] if '.' in filename else "" for filename in images_filename]
		self.extension = [filename.split('.')[1] if '.' in filename else "" for filename in images_filename]
		self.nargs = parsed_args.number + (1 if parsed_args.combined else 0)
		self.number_of_rows = n_rows
		self.fig, self.axes = plt.subplots(self.number_of_rows, self.nargs, figsize=(self.nargs * 5, self.number_of_rows * 5))

	@staticmethod
	def rotate(img, angle: int = 45, **kwargs):
		"""
		Rotate the image by the given angle
		:param img: the image to be rotated
		:param angle: the angle to rotate the image
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the rotated image
		"""
		(height, width) = img.shape[:2]
		center = (width / 2, height / 2)
		matrix = cv2.getRotationMatrix2D(center, angle, 1)
		rotated = cv2.warpAffine(img, matrix, (width, height), borderValue=(255, 255, 255))
		return rotated

	@staticmethod
	def flip(img, **kwargs):
		"""
		Flip the given image around the x-axis
		:param img: the image to be flipped
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the flipped image
		"""
		flipped = cv2.flip(img, 0)
		return flipped

	@staticmethod
	def blur(img, **kwargs):
		"""
		Blur the given image using a Gaussian filter
		:param img: the image to be blurred
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the blurred image
		"""
		blurred = cv2.GaussianBlur(img, (7, 7), 0)
		return blurred

	@staticmethod
	def heighten_contrast(img, **kwargs):
		"""
		Raise the contrast of the given image
		:param img: the image to raise the contrast of
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the image with higher contrast
		"""
		higher_contrast = cv2.convertScaleAbs(img, alpha=1.5)
		return higher_contrast

	@staticmethod
	def scale(img, zoom_factor=1.2, **kwargs):
		"""
		Scale the given image
		:param img: the image to be scaled
		:param zoom_factor: the factor to scale the image
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the scaled image
		"""
		(height, width) = img.shape[:2]
		center = (width // 2, height // 2)
		new_height = int(height / zoom_factor)
		new_width = int(width / zoom_factor)
		scaled = img[center[1] - new_height // 2:center[1] + new_height // 2,
															center[0] - new_width // 2:center[0] + new_width // 2]
		return scaled

	@staticmethod
	def shear(img, axis=1):
		"""
		Shear the given image along the specified axis.
		:param img: The image to be sheared.
		:param axis: The axis along which the image is to be sheared. (0 for vertical, 1 for horizontal)
		:return: The sheared image.
		"""
		(height, width) = img.shape[:2]
		shear_amount = min(height, width) // 4
		starting_points = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
		pts1 = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)

		if axis == 0:  # over vertical axis
			destination_points = np.array([
				[0, 0],
				[width, 0],
				[shear_amount, height],
				[width - shear_amount, height]],
				dtype=np.float32)
		elif axis == 1:  # over horizontal axis
			destination_points = np.array([
				[0, 0],
				[width - shear_amount, 0],
				[shear_amount, height],
				[width, height]],
				dtype=np.float32)
		else:
			raise ValueError("Invalid axis value. Use 0 for vertical shear or 1 for horizontal shear.")

		matrix = cv2.getPerspectiveTransform(starting_points, destination_points)
		sheared = cv2.warpPerspective(img, matrix, (width, height))
		return sheared

	def combine_all_methods_in_image(self, img):
		"""
		Combine the augmented images in a single image
		:return: the combined image
		"""
		rotated = self.rotate(img)
		flipped = self.flip(rotated)
		blurred = self.blur(flipped)
		raised_contrast = self.heighten_contrast(blurred)
		combined = self.scale(raised_contrast)
		# combined = self.distort(scaled) not include distortion
		# TODO: add other methods
		return combined

	def plot_augmentation(self, applied_method, augmented_image, img_index, suffix):
		"""
		Plot the augmented image
		:param applied_method: the method applied on the image for augmentation
		:param augmented_image: the augmented image to be plotted
		:param img_index: index of the image
		:param suffix: augmentation method name to be added to the title
		:return: None
		"""
		if augmented_image is not None:
			if img_index < self.number_of_rows:
				self.axes[0][applied_method].set_title(f"{suffix.capitalize()}", fontsize=self.nargs * 7)
				self.axes[img_index][applied_method].imshow(augmented_image)
				self.axes[img_index][applied_method].axis("off")
			if method == 0:
				pass
			path = os.path.join(self.input_directory[img_index], f"{self.title[img_index]}_{suffix}.{self.extension[img_index]}")
			cv2.imwrite(path, augmented_image)
			logger.info(f"Saved augmented image \"{self.title[img_index]}_{suffix}.{self.extension[img_index]}\""
															+ f" in \"{self.input_directory[img_index]}\"")

	def apply_augmentation(self, method_enum, combine=False, **kwargs):
		"""
		Apply the specified augmentation method based on the MethodEnum enum.
		:param method_enum: the method specified using the MethodEnum enum
		:param combine: if True, combine the augmented images in a single image
		:param kwargs: additional keyword arguments for the specific augmentation method
		:return: the augmented image
		"""

		augmentation_functions = {
			MethodEnum.ORIGINAL.value: None,
			MethodEnum.ROTATE.value: self.rotate,
			MethodEnum.FLIP.value: self.flip,
			MethodEnum.BLUR.value: self.blur,
			MethodEnum.CONTRAST.value: self.heighten_contrast,
			MethodEnum.SCALE.value: self.scale,
			MethodEnum.SHEAR.value: self.shear
		}

		for img_index, img in enumerate(self.images):
			# Verify if the image is valid (not None and path exists)
			if len(self.images) == 0:
				logger.error("No valid images found.\tExiting now...")
				exit(1)
			loaded_image = cv2.imread(img)
			if method_enum == self.nargs - 1:
				if combine:
					augmented_img = self.combine_all_methods_in_image(loaded_image)
					self.plot_augmentation(method_enum, augmented_img, img_index, "combined")
			elif method_enum != MethodEnum.ORIGINAL.value:
				augmented_img = augmentation_functions[method_enum](loaded_image, **kwargs)
				self.plot_augmentation(method_enum, augmented_img, img_index, MethodEnum(method_enum).name.lower())

			elif method_enum == MethodEnum.ORIGINAL.value:
				augmented_img = loaded_image
				self.plot_augmentation(method_enum, augmented_img, img_index, MethodEnum(method_enum).name.lower())
			else:
				logger.error("Invalid method_enum value")


def configurate_parser():
	"""
	Parse the command line arguments
	:return: the parsed arguments
	"""
	_parser = argparse.ArgumentParser(description="Augment images of leaves dataset")
	_parser.add_argument("-d", "--directory", help="Directory where to get the images to be augmented")
	_parser.add_argument("-f", "--files", nargs='+', help="image files to be augmented")
	_parser.add_argument("-v", "--verbose", help="Enable verbose mode", action="store_true")
	_parser.add_argument("-n", "--number", type=int, help="Number of augmented images to be generated", default=6)
	_parser.add_argument(
		"-c",
		"--combined",
		help="Apply a combination of all the augmented method in a single image",
		action="store_true")
	_parser.add_argument(
		"-s", "--save-plot", help="Save the resulted plot in the root directory",
		action="store_true", default=False)
	_parser.add_argument("-max", "--max-number-of-rows", help="Maximum number of rows in the resulted plot", default=10)
	arguments = _parser.parse_args()

	# Check if both -f and -d are provided
	if arguments.files and arguments.directory:
		_parser.error("Both -f and -d options cannot be used simultaneously. Use either -f or -d.")
	if not arguments.files and not arguments.directory:
		_parser.error("Either -f or -d option must be used.")
	if arguments.max_number_of_rows < 1:
		_parser.error("Maximum number of rows must be at least 1.")
	elif arguments.files and len(arguments.files) > arguments.max_number_of_rows and not arguments.save_plot:
		error_msg = ("Maximum number of rows that can be showed cannot be greater than 10 for performance issues.\n" +
															f"{' ' * len('Augmentation.py: error: ')}" +
															"Use -s option to save the resulted plot in the root directory.")
		_parser.error(error_msg)
	return arguments


def get_files_in_directory(directory):
	"""
	Recursively get all files within a directory and its subdirectories.
	:param directory: The directory to start the search.
	:return: A list of file paths.
	"""
	file_paths = []
	for root, _, files in os.walk(directory):
		for file in files:
			file_paths.append(os.path.join(root, file))
	return file_paths


if __name__ == "__main__":
	args = configurate_parser()
	logger = configure_logger(args.verbose)
	load_dotenv()
	if args.number < 6:
		logger.error("Using default number of augmented methods (6)")
		args.number = 6
	elif args.number > 10:  # TBD (depending on how many method we have)
		logger.error("Using maximum number of augmented images (10)")
		args.number = 10
	if args.directory and not args.files:
		logger.info("Using images from directory")
		args.files = get_files_in_directory(args.directory)
		logger.info(f"{len(args.files)} images found in {args.directory} and its subdirectories.")
		if len(args.files) == 0:
			logger.error("No images found in the specified directory.\tExiting now...")
			exit(1)
	augmentation_instance = Augmentation(args)
	for method in range(0, args.number + (1 if args.combined else 0)):
		augmentation_instance.apply_augmentation(method, args.combined)
	if len(args.files) > 0:
		if args.save_plot:
			plt.tight_layout()
			plt.savefig("augmented_images_plot.png")
			logger.info("Saved augmented images plot in the root directory as \"augmented_images_plot.png\"")
		elif not args.save_plot and len(augmentation_instance.axes) <= 10:
			print("len plots: ", len(augmentation_instance.axes))
			plt.show()
