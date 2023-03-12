import fastText

def train_fasttext(training_file, output_file, lr=0.1, epoch=25, wordNgrams=2):
    """
    Train a fastText model on the product data.
    """
    model = fastText.train_supervised(input=training_file, lr=lr, epoch=epoch, wordNgrams=wordNgrams)
    model.save_model(output_file)

if __name__ == "__main__":
    train_fasttext('/workspace/datasets/fasttext/training_data.txt', '/workspace/fasttext_models/product_classifier.bin')