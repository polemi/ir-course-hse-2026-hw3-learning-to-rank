#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


SEED = 42
THREADS = 8
FEATURE_MODE = 'standard'
PARAMS = {
    'loss_function': 'YetiRank',
    'iterations': 1100,
    'learning_rate': 0.04,
    'depth': 6,
    'l2_leaf_reg': 10,
    'random_seed': SEED,
    'thread_count': THREADS,
    'task_type': 'CPU',
    'allow_writing_files': False,
}


def make_features(data, columns, mode='raw'):
    values = data.loc[:, columns].astype(np.float32)
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0)
    if mode == 'raw':
        return values
    groups = values.groupby(data['QueryId'], sort=False)
    ranks = groups.rank(method='average', pct=True).astype(np.float32)
    ranks.columns = [f'{column}__rank' for column in columns]
    minimum = groups.transform('min')
    spread = groups.transform('max') - minimum
    scaled = ((values - minimum) / spread.replace(0, np.nan)).fillna(0).astype(np.float32)
    scaled.columns = [f'{column}__scaled' for column in columns]
    if mode == 'rank':
        return pd.concat([values, ranks], axis=1)
    if mode == 'scaled':
        return pd.concat([values, scaled], axis=1)
    if mode == 'both':
        return pd.concat([values, ranks, scaled], axis=1)
    if mode == 'standard':
        deviation = groups.transform('std').replace(0, np.nan)
        standardized = ((values - groups.transform('mean')) / deviation).fillna(0).astype(np.float32)
        standardized.columns = [f'{column}__standard' for column in columns]
        return pd.concat([values, ranks, scaled, standardized], axis=1)
    raise ValueError(f'Unknown feature mode: {mode}')


def ndcg_by_query(labels, scores, query_ids):
    result = []
    boundaries = np.r_[0, np.flatnonzero(query_ids[1:] != query_ids[:-1]) + 1, len(query_ids)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        gains = np.exp2(labels[start:end]) - 1
        size = min(10, end - start)
        discounts = 1 / np.log2(np.arange(size) + 2)
        ideal = np.sort(gains)[::-1][:size] @ discounts
        ranking = np.argsort(-scores[start:end], kind='stable')[:size]
        result.append(float(gains[ranking] @ discounts / ideal) if ideal else 0.0)
    return np.asarray(result)


def train(data_dir, model_file):
    data = pd.read_csv(data_dir / 'train.csv').sort_values('QueryId', kind='stable').reset_index(drop=True)
    columns = [column for column in data if column not in ('QueryId', 'Label')]
    features = make_features(data, columns, FEATURE_MODE)
    query_ids = np.sort(data['QueryId'].unique())
    np.random.default_rng(SEED).shuffle(query_ids)
    validation_ids = query_ids[:max(1, len(query_ids) // 5)]
    validation = data['QueryId'].isin(validation_ids).to_numpy()
    fit_pool = Pool(features.loc[~validation], label=data.loc[~validation, 'Label'],
                    group_id=data.loc[~validation, 'QueryId'])
    validation_pool = Pool(features.loc[validation], label=data.loc[validation, 'Label'],
                           group_id=data.loc[validation, 'QueryId'])
    check_model = CatBoostRanker(**PARAMS, eval_metric='NDCG:top=10;type=Exp')
    check_model.fit(fit_pool, eval_set=validation_pool, use_best_model=False, verbose=300)
    validation_scores = check_model.predict(features.loc[validation], thread_count=THREADS)
    metric = ndcg_by_query(data.loc[validation, 'Label'].to_numpy(), validation_scores,
                           data.loc[validation, 'QueryId'].to_numpy()).mean()
    print(f'validation_ndcg@10={metric:.6f}')
    del check_model, fit_pool, validation_pool
    pool = Pool(features, label=data['Label'], group_id=data['QueryId'])
    model = CatBoostRanker(**PARAMS, metadata={
        'input_features': json.dumps(columns),
        'feature_mode': FEATURE_MODE,
    })
    model.fit(pool, verbose=300)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_file))


def predict(data_dir, model_file, submission_file):
    model = CatBoostRanker()
    model.load_model(str(model_file))
    metadata = model.get_metadata()
    columns = json.loads(metadata['input_features'])
    mode = metadata['feature_mode']
    test = pd.read_csv(data_dir / 'test.csv')
    features = make_features(test, columns, mode)
    scores = model.predict(features, thread_count=THREADS)
    if not np.isfinite(scores).all():
        raise ValueError('Predictions contain non-finite values')
    if not test['ObjectId'].is_unique:
        raise ValueError('ObjectId must be unique')
    submission_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores}).to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model_file', required=True, type=Path)
    parser.add_argument('--submission_file', type=Path)
    parser.add_argument('data_dir', type=Path)
    args = parser.parse_args()
    if not args.train and args.submission_file is None:
        parser.error('--submission_file is required for prediction')
    start = perf_counter()
    if args.train:
        train(args.data_dir, args.model_file)
    else:
        predict(args.data_dir, args.model_file, args.submission_file)
    print(f'finished, elapsed={perf_counter() - start:.3f}')


if __name__ == '__main__':
    main()
