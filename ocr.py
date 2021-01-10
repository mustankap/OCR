#home/OCR

DATA_DIR='home/OCR/data/'

TRAIN_DATA_FILENAME = DATA_DIR + 'train-images'
TRAIN_LABEL_FILENAME = DATA_DIR + 'train-labels'
TEST_DATA_FILENAME = DATA_DIR + 't10k-images'
TEST_LABEL_FILENAME = DATA_DIR + 't10k-labels'

def read_images(filename):
	with open(filename,'rb') as f:
		_ = f.read(4) #magic number
		no_of_images = f.read(4) 
		no_of_rows = f.read(4)
		no_of_columns = f.read(4)  
		
	
def main():
	print(read_images(TRAIN_DATA_FILENAME))

def __name__='__main__':
	main()


