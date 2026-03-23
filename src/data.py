from __future__ import annotations

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
    val_size: float = 0.02
    seed: int = 42


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
) -> DatasetDict:
    df = pd.read_csv(config.csv_path)

    required_cols = {config.question_col, config.answer_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df = df.dropna(subset=[config.question_col, config.answer_col]).copy()

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

    return DatasetDict({"train": train_ds, "validation": val_ds})
