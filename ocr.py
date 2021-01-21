

DEBUG = True

if DEBUG:
	from PIL import Image
	import numpy as np
	def read_image(path):
		return np.asarray(Image.open(path).convert('L'))#bytes to pixel
	def write_image(image, path):
		img = Image.fromarray(np.array(image), 'L')
		img.save(path)
	

DATA_DIR = 'data/'
TEST_DIR = 'test/'
TEST_DATA_FILENAME = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels.idx1-ubyte'

def bytes_to_int(byte_data):
	return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images=None):
	images = []
	with open(filename, 'rb') as f:
		_ = f.read(4)  # magic number
		n_images = bytes_to_int(f.read(4))
		if n_max_images:
			n_images = n_max_images
		n_rows = bytes_to_int(f.read(4))
		n_columns = bytes_to_int(f.read(4))
		for image_idx in range(n_images):
			image = []
			for row_idx in range(n_rows):
				row = []
				for col_idx in range(n_columns):
					pixel = f.read(1)
					row.append(pixel)
				image.append(row)
			images.append(image)
		return images
		
		
def read_labels(filename, n_max_labels=None):
	labels = []
	with open(filename, 'rb') as f:
		_ = f.read(4)  # magic number
		n_labels = bytes_to_int(f.read(4))
		if n_max_labels:
			n_labels = n_max_labels
		for label_idx in range(n_labels):
			label = bytes_to_int(f.read(1))
			labels.append(label)
	return labels

def flatten_list(l):
	return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
	return [flatten_list(sample) for sample in X]
	
def dist(x, y):
	"""
	Returns the Euclidean distance between vectors `x` and `y`.
	"""
	return sum(
		[
			(bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
			for x_i, y_i in zip(x, y)
		]
	) ** (0.5)	
	

def get_training_distances_for_test_sample(X_train, test_sample):
	return [dist(train_sample, test_sample) for train_sample in X_train]
	
def get_most_frequent_element(l):
	return max(l, key=l.count)
'''funning k nearest neighbours for every sample which we had to find out we are comparing it to all the other digits
getting the ones which were closest and out them we get the top k and hen we get he most frequen out of those top k'''
def knn(X_train, y_train, X_test, k=3):
	y_pred = []
	for test_sample_idx, test_sample in enumerate(X_test):
		training_distances = get_training_distances_for_test_sample(X_train, test_sample)
		sorted_distance_indices = [
		pair[0]
		for pair in sorted(
			enumerate(training_distances),
			key=lambda x: x[1]
		)
		]
		
		candidates = [
		   y_train[idx]
		    for idx in sorted_distance_indices[:k]
		]
		
		top_candidate = get_most_frequent_element(candidates)
		y_pred.append(top_candidate)
		
		
	return y_pred
	
def main():
	x_train = read_images(TRAIN_DATA_FILENAME,1000)
	y_train = read_labels(TRAIN_LABELS_FILENAME,1000)
	x_test = read_images(TEST_DATA_FILENAME,50)
	y_test = read_labels(TEST_LABELS_FILENAME,50)
	
	if DEBUG:
		for idx, test_sample in enumerate(x_test):
			write_image(test_sample,f'{TEST_DIR}{idx}.png')
		#x_test=[read_image(f'{DATA_DIR}our_test.png')]
		#y_test=[3]
			
	x_train = extract_features(x_train)
	x_test = extract_features(x_test)
	
	y_pred = knn(x_train,y_train,x_test,3)#y_pred has a prediction of each of or x_tests
	
	print(y_pred)
	'''comparing every predicted label to every true label'''
	accuracy = sum([
	int(y_pred_i ==y_test_i)
	for y_pred_i, y_test_i
	in zip(y_pred, y_test)
	]) / len(y_test)

	print(f'Predicted labels: {y_pred}')
	print(f'Accuracy: {accuracy * 100}%')

	#print(f'point is {bytes_to_int(y_test[test_sample_idx])} and we guessed {candidates}')
		
		#y_sample=5
	
	
if __name__ == '__main__':
	main()

