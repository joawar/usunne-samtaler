from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import config
from dataset import (
    get_clean_df,
    dataframe_from_json,
    make_binary_df,
    UCCDataset,
    balance_df,
    remove_zero_labels,
    oversampling,
    get_highest_confidence_datapoints,
    english_oversampling
)
# from embedding_visualization import VisualizationCallback, get_hidden_states
import pandas as pd
from eval_module import get_multilabel_metrics, get_binary_metrics
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import os
import torch as th
import numpy as np
import json
import pprint
import pathlib


class UCCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
        # loss_fn = th.nn.BCEWithLogitsLoss()
        weight_tensor = th.tensor(config.LOSS_WEIGHTS).cuda()
        loss_fn = th.nn.CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fn(outputs['logits'], inputs['labels'])
        return (loss, outputs) if return_outputs else loss


def model_init():
    return BertForSequenceClassification.from_pretrained(
        config.MODEL_STR, return_dict=True
    )


def objective(metrics):
    try:
        return metrics[config.COMPUTE_OBJECTIVE]
    except KeyError:
        return metrics[f'eval_{config.COMPUTE_OBJECTIVE}']


def hp_space(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 5e-6, 1e-4,
                                             log=False),
        'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 5),
        'seed': trial.suggest_int('seed', 1, 50),
        'weight_decay': trial.suggest_uniform('weight_decay', 0, 0.3),
        # 'per_device_train_batch_size': trial.suggest_categorical(
        #     'per_device_train_batch_size', [10, 15, 20]),
        'warmup_steps': trial.suggest_categorical(
            'warmup_steps', [round(0.05*total_steps), round(0.1*total_steps),
                             round(0.15*total_steps)]
        )
    }


def compute_metrics(eval_pred: EvalPrediction):
    scores = eval_pred.predictions  # np.array (N,2)
    labels = eval_pred.label_ids  # np.array (N,)
    if config.BINARY:
        return get_binary_metrics(scores, labels, config.METRIC_FILE)
    else:
        return get_multilabel_metrics(scores, labels) 


def cross_validation(raw_df):
    clean_df = get_clean_df(raw_df)
    train_df = make_binary_df(clean_df, config.CHARACTERISTIC)
    run_number = 0
    metrics = dict()
    for train_idx, val_idx in rskf.split(list(train_df.text), list(train_df.labels)):
        X_train, X_val = train_df.text[train_idx], train_df.text[val_idx]
        Y_train, Y_val = train_df.labels[train_idx], train_df.labels[val_idx]
        fold_df = pd.concat([X_train, Y_train], axis=1).reset_index(drop=True)
        val_df = pd.concat([X_val, Y_val], axis=1).reset_index(drop=True)
        print(f'val proportion: {len(val_df)/(len(val_df) + len(fold_df))}')
        print(f'train proportion: {len(fold_df)/(len(val_df) + len(fold_df))}')
        if config.UNDERSAMPLING:
            fold_df = balance_df(fold_df).reset_index(drop=True)
        if config.OVERSAMPLING:
            fold_df = oversampling(fold_df).reset_index(drop=True)
        if config.ENG_OVERSAMPLING:
            full_UCC = None
            fold_df = None

        train_data = UCCDataset(fold_df, tokenizer, config.MAX_LEN)
        val_data = UCCDataset(val_df, tokenizer, config.MAX_LEN)
        total_steps = config.EPOCHS*len(train_data)/config.TRAIN_BATCH_SIZE
        warmup_steps = round(0.1*total_steps)
        log_interval = round(total_steps/config.N_LOGS)

        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            do_train=True,
            do_eval=config.EVALUATE_DURING_TRAINING,
            evaluation_strategy=config.EVAL_STRAT,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            logging_steps=log_interval,
            per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            seed=config.SEED,
            num_train_epochs=config.EPOCHS,
            disable_tqdm=True,
            run_name=config.RUN_NAME,
            load_best_model_at_end=False,
            metric_for_best_model=config.COMPUTE_OBJECTIVE,
            logging_first_step=True,
            lr_scheduler_type='linear',
            warmup_steps=warmup_steps
        )
        trainer = UCCTrainer(
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics
        )
        trainer.train()
        run_metrics = trainer.evaluate()
        if len(metrics) == 0:
            for key in run_metrics:
                metrics[key] = [run_metrics[key][-1]]
        else:
            for key in run_metrics:
                metrics[key].append(run_metrics[key][-1])

        # os.rename(
        #     f'{config.METRIC_FILE}.json',
        #     f'{config.METRIC_FILE}_{run_number}.json'
        #     )
        print(f'Finished run {run_number}')
        run_number += 1

    for key in metrics:
        avg = np.mean(metrics[key])
        print(f'{key}: {avg}')

def training(save=False):
    train_data = UCCDataset(train_df, tokenizer, config.MAX_LEN)
    val_data = UCCDataset(val_df, tokenizer, config.MAX_LEN)
    total_steps = config.EPOCHS*len(train_data)/config.TRAIN_BATCH_SIZE
    warmup_steps = round(0.1*total_steps)
    log_interval = round(total_steps/config.N_LOGS)
    training_args = TrainingArguments(
        save_strategy=config.SAVE_STRAT,
        output_dir=config.OUTPUT_DIR,
        do_train=True,
        do_eval=config.EVALUATE_DURING_TRAINING,
        evaluation_strategy=config.EVAL_STRAT,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=log_interval,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        seed=config.SEED,
        num_train_epochs=config.EPOCHS,
        disable_tqdm=True,
        run_name=config.RUN_NAME,
        load_best_model_at_end=config.LOAD_BEST_LAST,
        metric_for_best_model=config.COMPUTE_OBJECTIVE,
        logging_first_step=True,
        lr_scheduler_type='linear',
        warmup_steps=warmup_steps
    )

    # model_config = BertConfig(
    #     vocab_size=tokenizer.vocab_size,
    #     pretrained_model_name_or_path=config.MODEL_STR,
    #     num_labels=config.N_LABELS,
    #     return_dict=True
    # )

    trainer = UCCTrainer(
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        model_init=model_init,
        compute_metrics=compute_metrics
    )
    # visualization_callback = VisualizationCallback(trainer)
    # trainer.add_callback(visualization_callback)
    trainer.train()

    if save:
        pathlib.Path(config.SAVE_DIR).mkdir(exist_ok=True, parents=True)
        trainer.save_model(config.SAVE_DIR)


def get_train_val_set():
    print(f'Using {config.TRAIN_PATH} as training set')
    train_df = pd.read_csv(config.TRAIN_PATH)

    if config.VAL_PATH is not None:
        print(f'Using {config.VAL_PATH} as validation set')
        val_df = pd.read_csv(config.VAL_PATH)
    else:
        val_df = None

    if config.COMBINE_TRAIN_VAL:
        print('Combining training set and validation set')
        train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # if config.UNDERSAMPLING:
    #     print('Undersampling the train dataset')
    #     clean_df = get_clean_df(train_df)
    #     train_df = make_binary_df(clean_df, config.CHARACTERISTIC)
    #     train_df = balance_df(train_df).reset_index(drop=True)
    
    # if config.ENG_OVERSAMPLING:
    #     eng_df = pd.read_csv('data/UCC/full.csv')
    #     if config.PURE_ENG_OVERSAMPLE:
    #         print(f'Sampling pure {config.CHARACTERISTIC_NAME} datapoints')
    #         train_df = english_oversampling(train_df, eng_df, config.CHARACTERISTIC_NAME, pure_only=True)
    #     else:
    #         print(f'Sampling {config.N_ENG_DATAPOINTS} extra {config.CHARACTERISTIC_NAME} datapoints')
    #         old_len = len(train_df)
    #         train_df = english_oversampling(train_df, eng_df, config.CHARACTERISTIC_NAME, n_eng=config.N_ENG_DATAPOINTS)
    #         print(f'Added {len(train_df) - old_len} pure {config.CHARACTERISTIC_NAME} english datapoints')

    # clean_train = get_clean_df(train_df)
    # train_df = make_binary_df(clean_train, config.CHARACTERISTIC)
    # clean_val = get_clean_df(val_df)
    # val_df = make_binary_df(clean_val, config.CHARACTERISTIC)

    return train_df, val_df


if __name__ == '__main__':
    if config.SAVE_MODEL:
        pathlib.Path(config.SAVE_DIR).mkdir(exist_ok=True, parents=True)

    print(f'Using {config.MODEL_STR} to load tokenizer')
    tokenizer = BertTokenizer.from_pretrained(
        config.MODEL_STR,
        do_lower_case=False
    )

    if config.LOSS_WEIGHTS is None:
        print('No loss weights')
        UCCTrainer = Trainer

    train_df, val_df = get_train_val_set()

    if config.CROSS_VALIDATION:
        rskf = RepeatedStratifiedKFold(
            n_splits=config.N_SPLITS,
            n_repeats=config.N_REPEATS
        )
        print('Running cross validation')
        cross_validation(train_df)
    else:
        print('Running training')
        print(f'Load best model at end: {config.LOAD_BEST_LAST}')
        print(f'Save model: {config.SAVE_MODEL}')
        training(config.SAVE_MODEL)