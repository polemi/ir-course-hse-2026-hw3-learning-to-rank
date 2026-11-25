# ДЗ 3. Learning to Rank

Kaggle: polemi

Публичный скор: 0.62730, бейзлайн: 0.61621.

CatBoostRanker с YetiRank. Признаки из задания + ранги, min-max нормировка и стандартизация внутри запроса.

Параметры:

```text
iterations=1100
learning_rate=0.04
depth=6
l2_leaf_reg=10
random_seed=42
```

Разбиение по запросам, 80/20. nDCG@10 на валидации: 0.62581, бейзлайн: 0.61996.

Python 3.12. Запуск:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./solution.py --train --model_file=model.bin PATH-TO-DATA
./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA
```
