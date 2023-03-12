import sys
import fasttext
import pandas as pd


def train_fasttext(training_file, output_file, lr=1., epoch=25, wordNgrams=2):
    """
    Train a fastText model on the product data.
    """
    model = fasttext.train_supervised(input=training_file, lr=lr, epoch=epoch, wordNgrams=wordNgrams)
    model.save_model(output_file)

def test_fasttext(test_file, model_file, k=1):
    """
    Test a fastText model on the product data.
    """
    model = fasttext.load_model(model_file)
    samples, precision, recall = model.test(test_file, k=k)
    print({"samples": samples, "precision": precision, "recall": recall})

def generate_synonyms(word, threshold, model):
    """
    Generate synonyms for a word using fasttext
    """
    
    synonyms = [synonym[1] for synonym in model.get_nearest_neighbors(word) if synonym[0] > threshold]
    synonyms = ",".join(synonyms)

    # Add the word itself to the synonyms
    synonyms = word + "," + synonyms
    
    return synonyms

def get_synonyms_from_pandas(
    threshold, 
    input_file="/workspace/datasets/fasttext/top_words.txt", 
    output_file="/workspace/datasets/fasttext/synonyms.csv",
    model_file="/workspace/datasets/fasttext/title_model_100.bin",
    ):
    """
    Get synonyms for each row in a pandas dataframe
    """

    model = fasttext.load_model(model_file)
    
    df = pd.read_csv(input_file, names=["word"])
    df["synonyms"] = df["word"].apply(lambda x: generate_synonyms(x, threshold, model))
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        print("Training fastText model...")
        train_fasttext(
            '/workspace/datasets/fasttext/training_data.txt',
            '/workspace/fasttext_models/product_classifier.bin',
            )
    elif sys.argv[1] == "test":
        print("Testing fastText model...")
        test_fasttext(
            '/workspace/datasets/fasttext/test_data.txt',
            '/workspace/fasttext_models/product_classifier.bin',
            k=int(sys.argv[2]) if len(sys.argv) > 2 else 1
            )
    elif sys.argv[1] == "synonyms":
        print("Generating synonyms...")
        get_synonyms_from_pandas(
            threshold=float(sys.argv[2]) if len(sys.argv) > 2 else 0.8,
        )
    else:
        raise ValueError("Invalid argument. Use 'train' or 'test'.")