from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


@dataclass
class DataConfig:
    csv_path: str
    question_col: str = "Question"
    answer_col: str = "Answer"
    val_size: float = 0.03
    seed: int = 42
    dedupe_on_answer: bool = True
    max_answer_chars: Optional[int] = 2000


def _read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        pass

    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        pass

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data_rows = []
    width = len(header)
    for row in rows[1:]:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[: width - 1] + [" ".join(row[width - 1 :])]
        data_rows.append(row)

    return pd.DataFrame(data_rows, columns=header)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def _build_text(tokenizer, question: str, answer: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{question.strip()}\n"
        f"<|assistant|>\n{answer.strip()}"
    )


def load_sft_dataset(
    tokenizer,
    config: DataConfig,
    system_prompt: str,
    limit_rows: Optional[int] = None,
    max_seq_length: int = 2048,
) -> DatasetDict:
    df = _read_csv_robust(config.csv_path)

    required_cols = {config.question_col, config.answer_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df = df.dropna(subset=[config.question_col, config.answer_col]).copy()
    df[config.question_col] = df[config.question_col].astype(str).str.strip()
    df[config.answer_col] = df[config.answer_col].astype(str).str.strip()

    if config.dedupe_on_answer:
        df["_answer_norm"] = df[config.answer_col].map(_normalize_text)
        df = df.drop_duplicates(subset=["_answer_norm"]).copy()
        df = df.drop(columns=["_answer_norm"])

    if config.max_answer_chars is not None and config.max_answer_chars > 0:
        max_chars = int(config.max_answer_chars)
        df[config.answer_col] = df[config.answer_col].str.slice(0, max_chars)

    if limit_rows is not None and limit_rows > 0:
        df = df.iloc[:limit_rows]

    if len(df) == 0:
        raise ValueError("No valid rows found after filtering null Question/Answer values")

    df["text"] = df.apply(
        lambda row: _build_text(
            tokenizer=tokenizer,
            question=str(row[config.question_col]),
            answer=str(row[config.answer_col]),
            system_prompt=system_prompt,
        ),
        axis=1,
    )

    # Keep the pipeline resilient when running smoke tests on tiny subsets.
    if len(df) < 2:
        train_df = df[["text"]]
        val_df = df[["text"]]
    else:
        train_df, val_df = train_test_split(
            df[["text"]], test_size=config.val_size, random_state=config.seed, shuffle=True
        )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True), preserve_index=False)
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True), preserve_index=False)

    def _tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(max_seq_length),
            padding=False,
        )
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded

    # Thêm remove_columns để xóa column "text" gốc
    train_ds = train_ds.map(
        _tokenize_batch,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
    )
    val_ds = val_ds.map(
        _tokenize_batch,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,
    )

    return DatasetDict({"train": train_ds, "validation": val_ds})
