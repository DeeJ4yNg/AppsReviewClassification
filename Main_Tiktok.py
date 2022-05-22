from Dataset_Tiktok import Data_Sets
from Train_val_Tiktok import Train_Validate
import warnings
warnings.filterwarnings("ignore")

#Parameters
data_path = "./tiktok_google_play_reviews.csv"
batch_size = 64
learning_rate = 0.00001
adam_epsilon = 1e-8
epochs = 5
path_to_save = "./model_save/"
Vocab_Path = "./vocab.txt"
Bert_Path = "./distilbert"

#Training set and Validation set
train_dataloader, validation_dataloader, tokenizer = Data_Sets(data_path, batch_size, Vocab_Path)

#Train model
Train_Validate(epochs, learning_rate, adam_epsilon, train_dataloader, validation_dataloader, path_to_save, tokenizer, Bert_Path)
