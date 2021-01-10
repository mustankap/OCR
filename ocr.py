#home/OCR

DATA_DIR='/home/OCR/data/'

TRAIN_DATA_FILENAME = DATA_DIR + 'train-images'
TRAIN_LABEL_FILENAME = DATA_DIR + 'train-labels'
TEST_DATA_FILENAME = DATA_DIR + 't10k-images'
TEST_LABEL_FILENAME = DATA_DIR + 't10k-labels'

def read_images(filename):
	images = []
	with open(filename,'rb') as f:
		_ = f.read(4) #magic number
		no_of_images = f.read(4) 
		no_of_rows = f.read(4)
		no_of_columns = f.read(4)
		for img_index in range(no_of_images):
			img=[]
			for row_index in range(no_of_rows):
				row=[]
				for column_index in range(no_of_columns):
					pixel = f.read(1)
					row.append(pixel)
				img.append(row)
			images.append(img)

def main():
	X_train = read_images(TRAIN_DATA_FILENAME)
	print(len(X_train))
	# Y_train = 
	# X_test = 
	# Y_test = 

if __name__=="__main__":
	main()


