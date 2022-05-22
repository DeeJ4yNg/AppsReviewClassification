from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup, DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from Dataset_Tiktok import Data_Sets
import warnings
warnings.filterwarnings("ignore")
#Bert pretrained Model


def Device_to_use():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        #print('Oooops, seems you do not have any GPU, go go go order a 3090Ti now!!!. Have to use CPU instead for now...TAT')
        device = torch.device("cpu")

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))



                                        # ========================================
                                        #               Training
                                        # ========================================

def Train_Validate(epochs, learning_rate, adam_epsilon, train_dataloader, validation_dataloader, output_dir, tokenizer, Bert_Path):
    #Bert Pretrained Model
    model = DistilBertForSequenceClassification.from_pretrained(
        Bert_Path,
        num_labels = 5,
        output_attentions = False,
        output_hidden_states = False
        #return_dict=False
    )
    print(model)
    #Optimizer AdamW (RMSProp + Momentom + weight decay)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=adam_epsilon
                      )
    #Loss_Func = torch.nn.CrossEntropyLoss()
    total_steps = len(train_dataloader) * epochs
    #Optimizer scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    model.to(Device_to_use())
    training_stats = []
    total_t0 = time.time()
    print('Ready for training done, total steps: ', total_steps)
    step_count = 0
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            step_count += 1
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(Device_to_use())
            b_input_mask = batch[1].to(Device_to_use())
            b_labels = batch[2].to(Device_to_use())
            model.zero_grad()
            loss_train = model(b_input_ids,
                             #token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
            loss = loss_train[0]
            print('Loss monitoring:', loss.item(), '   ----->   Training is in progress: {:.2%}'.format(step_count/total_steps))
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #梯度裁剪 g = min(1, theta / ||g||) * g
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

                                # ========================================
                                #                Validation
                                # ========================================

        print("")
        print("Running Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

    # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(Device_to_use())
            b_input_mask = batch[1].to(Device_to_use())
            b_labels = batch[2].to(Device_to_use())
            with torch.no_grad():
                loss_val = model(b_input_ids,
                                   #token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = loss_val[0]
            total_eval_loss += loss.item()

        # Move logits and labels to CPU
            logits = loss_val[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
        )

                                # ========================================
                                #                save result
                                # ========================================

    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv('./Results/result.csv')

    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([i for i in range(1, epochs+1)])
    plt.show()

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    #Save model
    #output_dir = "./model_save/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)