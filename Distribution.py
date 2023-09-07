import errno
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from leaffliction import configure_logger


def get_categories_and_subcategories(directory_name):
	"""
	Get the categories and subcategories of the dataset
	:param directory_name: the directory where the dataset is stored
	:return: a dictionary containing the categories and subcategories
	"""
	category_dict = {}
	dir_list = os.listdir(directory_name)
	for label in dir_list:
		parts = label.split("_")
		category_name = parts[0]
		subcategory = "_".join(parts[1:])
		if category_name not in category_dict:
			category_dict[category_name] = {}
		if subcategory not in category_dict[category_name]:
			category_dict[category_name][subcategory] = 0
		subcategory_count = len(os.listdir(directory_name + "/" + label))
		category_dict[category_name][subcategory] = subcategory_count
	return category_dict


def display_analysis_plots(categories_dict, output_directory):
	"""
	Display the analysis of the dataset
	:param categories_dict: the dictionary containing the categories and subcategories
	:param output_directory: the directory where to store the plots (if None, the plots will be displayed)
	:return: None
	"""
	for category_name, subcategories in categories_dict.items():
		palette = sns.color_palette(None, subcategories.__len__())
		# Set the color palette for both plots
		sns.set_palette(palette)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5))
		fig.suptitle(f"{category_name.capitalize()} class distribution")
		axes[0].pie(subcategories.values(), autopct='%.2f%%', colors=palette)
		sns.barplot(ax=axes[1], x=list(f"{category_name}_{name}" for name in subcategories),
														y=list(subcategories.values()), width=0.7, palette=palette)
		axes[1].grid(linestyle='-.', linewidth=0.5)
		if output_directory is None:
			logger.info("Displaying analysis plots...")
			plt.show()
		else:
			logger.info(f"Saving {categories} analysis plots to '{output_directory}'...")
			plt.savefig(f"{output_directory}/{category_name}.png")
			plt.close()
			logger.info("Done.")


def configurate_parser():
	"""
	Parse the command line arguments
	:return: the parsed arguments
	"""
	_parser = argparse.ArgumentParser(description="Analyze categories and subcategories of leaves dataset")
	_parser.add_argument("-d", "--directory", nargs=1, help="Input directory containing categories and subcategories", required=True)
	_parser.add_argument("-o", "--output", help="Output directory where to store the plots")
	_parser.add_argument("-v", "--verbose", help="Enable verbose mode", action="store_true")
	return _parser


def validate_directory(directory_path_list: list, module_parser) -> str or None:
	"""
	Validate the directory path
	:param directory_path_list: the directory path to validate
	:param module_parser: the parser used to print the usage message
	:return: the validated directory path
	"""
	if len(directory_path_list) != 1:
		logger.error("Expected exactly one argument for 'directory'.")
		module_parser.print_usage()
		return None
	if not os.path.isdir(directory_path_list[0]):
		logger.error(f"Directory '{directory_path_list[0]}' does not exist.")
		module_parser.print_usage()
		return None
	elif len(os.listdir(directory_path_list[0])) == 0:
		logger.error(f"Directory '{directory_path_list[0]}' is empty.")
		module_parser.print_usage()
		return None
	logger.info(f"Using directory '{directory_path_list[0]}'")
	return directory_path_list[0]


if __name__ == "__main__":
	parser = configurate_parser()
	args = parser.parse_args()
	logger = configure_logger(args.verbose)
	directory = validate_directory(args.directory, parser)
	if directory is None:
		sys.exit(errno.EINVAL)
	logger.info("Starting analysis of the dataset")
	categories = get_categories_and_subcategories(directory)
	logger.info(f"We have {len(categories.keys())} categories:")
	for category in categories:
		logger.info(f"\t- {category}: {categories[category]}")
	display_analysis_plots(categories, args.output)
