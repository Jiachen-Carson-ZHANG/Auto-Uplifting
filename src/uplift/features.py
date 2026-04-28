"""Feature table construction for RetailHero-style uplift experiments."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence

import pandas as pd

from src.models.uplift import (
    UpliftFeatureArtifact,
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    _stable_hash,
)


FeatureCohort = Literal["train", "scoring", "all"]
_SAMPLE_BYTES = 1_048_576


def _file_sample_hash(path: Path, sample_bytes: int = _SAMPLE_BYTES) -> str:
    """Hash a whole small file or stable head/tail samples from a large file."""
    stat = path.stat()
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        if stat.st_size <= sample_bytes * 2:
            digest.update(handle.read())
        else:
            digest.update(handle.read(sample_bytes))
            handle.seek(max(stat.st_size - sample_bytes, 0))
            digest.update(handle.read(sample_bytes))

    return digest.hexdigest()


def compute_dataset_fingerprint(contract: UpliftProjectContract) -> str:
    """Compute a cheap deterministic fingerprint of the contract's source tables."""
    schema = contract.table_schema
    sources = {
        "clients": schema.clients_table,
        "purchases": schema.purchases_table,
        "train": schema.train_table,
        "scoring": schema.scoring_table,
    }
    if schema.products_table:
        sources["products"] = schema.products_table

    payload = []
    for source_name, source_path in sorted(sources.items()):
        path = Path(source_path)
        stat = path.stat()
        payload.append(
            {
                "source": source_name,
                "file_name": path.name,
                "size_bytes": stat.st_size,
                "sample_hash": _file_sample_hash(path),
            }
        )

    return _stable_hash(payload)


def validate_feature_table(
    feature_df: pd.DataFrame,
    *,
    entity_key: str,
    forbidden_columns: Sequence[str],
    expected_ids: Iterable[str],
) -> None:
    """Validate feature table shape and leakage-sensitive column exclusions."""
    if entity_key not in feature_df.columns:
        raise ValueError(f"feature table missing entity key: {entity_key}")

    duplicate_count = int(feature_df[entity_key].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"feature table has duplicate {entity_key} rows: {duplicate_count}")

    forbidden = sorted(set(forbidden_columns).intersection(feature_df.columns))
    if forbidden:
        raise ValueError(f"forbidden feature columns present: {forbidden}")

    expected = set(expected_ids)
    observed = set(feature_df[entity_key])
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"feature table ids mismatch: missing={len(missing)}, extra={len(extra)}"
        )


def _cohort_ids(contract: UpliftProjectContract, cohort: FeatureCohort) -> List[str]:
    schema = contract.table_schema
    entity_key = contract.entity_key

    if cohort == "train":
        ids = pd.read_csv(schema.train_table, usecols=[entity_key])[entity_key]
    elif cohort == "scoring":
        ids = pd.read_csv(schema.scoring_table, usecols=[entity_key])[entity_key]
    elif cohort == "all":
        train_ids = pd.read_csv(schema.train_table, usecols=[entity_key])[entity_key]
        scoring_ids = pd.read_csv(schema.scoring_table, usecols=[entity_key])[entity_key]
        ids = pd.concat([train_ids, scoring_ids], ignore_index=True)
    else:  # pragma: no cover - Literal type guards normal callers
        raise ValueError(f"unsupported feature cohort: {cohort}")

    return [str(value) for value in ids.tolist()]


def _build_client_features(
    clients: pd.DataFrame,
    *,
    entity_key: str,
    expected_ids: Sequence[str],
) -> pd.DataFrame:
    cohort_df = pd.DataFrame({entity_key: list(expected_ids)})
    client_df = cohort_df.merge(clients, on=entity_key, how="left")

    age = pd.to_numeric(client_df.get("age"), errors="coerce")
    valid_age = age.between(14, 100)
    client_df["age_invalid_flag"] = (~valid_age | age.isna()).astype(int)
    client_df["age_clean"] = age.where(valid_age, -1).fillna(-1).astype(float)

    issue_date = pd.to_datetime(client_df.get("first_issue_date"), errors="coerce")
    redeem_date = pd.to_datetime(client_df.get("first_redeem_date"), errors="coerce")
    delay_days = (redeem_date - issue_date).dt.total_seconds() / 86_400

    client_df["redeem_missing_flag"] = redeem_date.isna().astype(int)
    client_df["issue_redeem_delay_days"] = delay_days.fillna(-1).round(3)
    client_df["issue_year"] = issue_date.dt.year.fillna(-1).astype(int)
    client_df["issue_month"] = issue_date.dt.month.fillna(-1).astype(int)

    gender = client_df.get("gender", pd.Series(["U"] * len(client_df))).fillna("U")
    gender = gender.astype(str).str.upper()
    for value in ["F", "M", "U"]:
        client_df[f"gender_{value}"] = (gender == value).astype(int)
    client_df["gender_other"] = (~gender.isin(["F", "M", "U"])).astype(int)

    return client_df[
        [
            entity_key,
            "age_clean",
            "age_invalid_flag",
            "redeem_missing_flag",
            "issue_redeem_delay_days",
            "issue_year",
            "issue_month",
            "gender_F",
            "gender_M",
            "gender_U",
            "gender_other",
        ]
    ]


def _read_purchase_transactions(
    purchases_path: str,
    *,
    entity_key: str,
    expected_ids: Sequence[str],
    chunksize: int,
) -> pd.DataFrame:
    cohort_ids = set(expected_ids)
    usecols = [
        entity_key,
        "transaction_id",
        "transaction_datetime",
        "regular_points_received",
        "express_points_received",
        "regular_points_spent",
        "express_points_spent",
        "purchase_sum",
        "product_quantity",
    ]
    grouped_chunks: List[pd.DataFrame] = []

    for chunk in pd.read_csv(purchases_path, usecols=usecols, chunksize=chunksize):
        chunk[entity_key] = chunk[entity_key].astype(str)
        chunk = chunk[chunk[entity_key].isin(cohort_ids)]
        if chunk.empty:
            continue

        chunk["transaction_datetime"] = pd.to_datetime(
            chunk["transaction_datetime"],
            errors="coerce",
        )
        for column in [
            "regular_points_received",
            "express_points_received",
            "regular_points_spent",
            "express_points_spent",
            "purchase_sum",
            "product_quantity",
        ]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").fillna(0.0)

        grouped = (
            chunk.groupby(
                [entity_key, "transaction_id", "transaction_datetime"],
                dropna=False,
                as_index=False,
            )
            .agg(
                {
                    "purchase_sum": "max",
                    "product_quantity": "sum",
                    "regular_points_received": "max",
                    "express_points_received": "max",
                    "regular_points_spent": "max",
                    "express_points_spent": "max",
                }
            )
        )
        grouped_chunks.append(grouped)

    if not grouped_chunks:
        return pd.DataFrame(columns=usecols)

    transactions = pd.concat(grouped_chunks, ignore_index=True)
    return (
        transactions.groupby(
            [entity_key, "transaction_id", "transaction_datetime"],
            dropna=False,
            as_index=False,
        )
        .agg(
            {
                "purchase_sum": "max",
                "product_quantity": "sum",
                "regular_points_received": "max",
                "express_points_received": "max",
                "regular_points_spent": "max",
                "express_points_spent": "max",
            }
        )
    )


def _aggregate_transactions(
    transactions: pd.DataFrame,
    *,
    entity_key: str,
    expected_ids: Sequence[str],
    suffix: str,
    reference_date: pd.Timestamp | None,
) -> pd.DataFrame:
    base = pd.DataFrame({entity_key: list(expected_ids)})
    if transactions.empty:
        aggregate = base.copy()
        aggregate[f"purchase_txn_count_{suffix}"] = 0
        aggregate[f"purchase_sum_{suffix}"] = 0.0
        aggregate[f"avg_transaction_value_{suffix}"] = 0.0
        aggregate[f"basket_quantity_{suffix}"] = 0.0
        aggregate[f"avg_basket_quantity_{suffix}"] = 0.0
        aggregate[f"recency_days_{suffix}"] = -1.0
        aggregate[f"points_received_total_{suffix}"] = 0.0
        aggregate[f"points_spent_total_{suffix}"] = 0.0
        aggregate[f"points_received_to_purchase_ratio_{suffix}"] = 0.0
        aggregate[f"points_spent_to_purchase_ratio_{suffix}"] = 0.0
        return aggregate

    tx = transactions.copy()
    tx["points_received_total"] = (
        tx["regular_points_received"] + tx["express_points_received"]
    )
    tx["points_spent_total"] = tx["regular_points_spent"] + tx["express_points_spent"]

    grouped = (
        tx.groupby(entity_key, as_index=False)
        .agg(
            purchase_txn_count=("transaction_id", "nunique"),
            purchase_sum=("purchase_sum", "sum"),
            basket_quantity=("product_quantity", "sum"),
            last_purchase_datetime=("transaction_datetime", "max"),
            points_received_total=("points_received_total", "sum"),
            points_spent_total=("points_spent_total", "sum"),
        )
    )
    grouped = base.merge(grouped, on=entity_key, how="left")

    count = grouped["purchase_txn_count"].fillna(0)
    purchase_sum = grouped["purchase_sum"].fillna(0.0)
    basket_quantity = grouped["basket_quantity"].fillna(0.0)

    result = pd.DataFrame({entity_key: grouped[entity_key]})
    result[f"purchase_txn_count_{suffix}"] = count.astype(int)
    result[f"purchase_sum_{suffix}"] = purchase_sum.round(6)
    result[f"avg_transaction_value_{suffix}"] = (
        purchase_sum / count.replace(0, pd.NA)
    ).fillna(0.0).round(6)
    result[f"basket_quantity_{suffix}"] = basket_quantity.round(6)
    result[f"avg_basket_quantity_{suffix}"] = (
        basket_quantity / count.replace(0, pd.NA)
    ).fillna(0.0).round(6)

    if reference_date is None:
        recency = pd.Series([-1.0] * len(grouped))
    else:
        recency = (
            reference_date - pd.to_datetime(grouped["last_purchase_datetime"], errors="coerce")
        ).dt.total_seconds() / 86_400
        recency = recency.fillna(-1.0)
    result[f"recency_days_{suffix}"] = recency.round(3)

    points_received = grouped["points_received_total"].fillna(0.0)
    points_spent = grouped["points_spent_total"].fillna(0.0)
    result[f"points_received_total_{suffix}"] = points_received.round(6)
    result[f"points_spent_total_{suffix}"] = points_spent.round(6)
    result[f"points_received_to_purchase_ratio_{suffix}"] = (
        points_received / purchase_sum.replace(0, pd.NA)
    ).fillna(0.0).round(6)
    result[f"points_spent_to_purchase_ratio_{suffix}"] = (
        points_spent / purchase_sum.replace(0, pd.NA)
    ).fillna(0.0).round(6)

    return result


def _build_purchase_features(
    contract: UpliftProjectContract,
    *,
    recipe: UpliftFeatureRecipeSpec,
    expected_ids: Sequence[str],
    chunksize: int,
) -> tuple[pd.DataFrame, Optional[str]]:
    entity_key = contract.entity_key
    transactions = _read_purchase_transactions(
        contract.table_schema.purchases_table,
        entity_key=entity_key,
        expected_ids=expected_ids,
        chunksize=chunksize,
    )

    if recipe.reference_date is not None:
        reference_date = pd.Timestamp(datetime.fromisoformat(recipe.reference_date))
    else:
        reference_date = (
            pd.to_datetime(transactions["transaction_datetime"], errors="coerce").max()
            if not transactions.empty
            else None
        )
    if pd.isna(reference_date):
        reference_date = None

    if reference_date is not None and not transactions.empty:
        transactions = transactions[
            pd.to_datetime(transactions["transaction_datetime"], errors="coerce")
            <= reference_date
        ]

    feature_df = _aggregate_transactions(
        transactions,
        entity_key=entity_key,
        expected_ids=expected_ids,
        suffix="lifetime",
        reference_date=reference_date,
    )

    if reference_date is None:
        for window in recipe.windows_days:
            window_features = _aggregate_transactions(
                pd.DataFrame(columns=transactions.columns),
                entity_key=entity_key,
                expected_ids=expected_ids,
                suffix=f"{window}d",
                reference_date=reference_date,
            )
            feature_df = feature_df.merge(window_features, on=entity_key, how="left")
        reference_date_str = reference_date.isoformat() if reference_date is not None else None
        return feature_df, reference_date_str

    for window in recipe.windows_days:
        cutoff = reference_date - pd.Timedelta(days=window)
        window_transactions = transactions[
            pd.to_datetime(transactions["transaction_datetime"], errors="coerce") >= cutoff
        ]
        window_features = _aggregate_transactions(
            window_transactions,
            entity_key=entity_key,
            expected_ids=expected_ids,
            suffix=f"{window}d",
            reference_date=reference_date,
        )
        feature_df = feature_df.merge(window_features, on=entity_key, how="left")

    reference_date_str = reference_date.isoformat() if reference_date is not None else None
    return feature_df, reference_date_str


def build_feature_table(
    contract: UpliftProjectContract,
    *,
    recipe: UpliftFeatureRecipeSpec,
    output_dir: str | Path,
    cohort: FeatureCohort = "train",
    chunksize: int = 100_000,
    force: bool = False,
) -> UpliftFeatureArtifact:
    """Build or load a cached customer-level feature table artifact."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset_fingerprint = compute_dataset_fingerprint(contract)
    feature_artifact_id = recipe.compute_feature_artifact_id(dataset_fingerprint)
    artifact_path = output / f"uplift_features_{cohort}_{feature_artifact_id}.csv"
    metadata_path = output / f"uplift_features_{cohort}_{feature_artifact_id}.metadata.json"

    if not force and artifact_path.exists() and metadata_path.exists():
        return UpliftFeatureArtifact.model_validate_json(metadata_path.read_text())

    expected_ids = _cohort_ids(contract, cohort)
    clients = pd.read_csv(contract.table_schema.clients_table)
    clients[contract.entity_key] = clients[contract.entity_key].astype(str)

    feature_df = _build_client_features(
        clients,
        entity_key=contract.entity_key,
        expected_ids=expected_ids,
    )
    reference_date: Optional[str] = recipe.reference_date

    purchase_groups = {"rfm", "basket", "points"}
    if purchase_groups.intersection(recipe.feature_groups):
        purchase_df, reference_date = _build_purchase_features(
            contract,
            recipe=recipe,
            expected_ids=expected_ids,
            chunksize=chunksize,
        )
        feature_df = feature_df.merge(purchase_df, on=contract.entity_key, how="left")

    validate_feature_table(
        feature_df,
        entity_key=contract.entity_key,
        forbidden_columns=[contract.target_column, contract.treatment_column],
        expected_ids=expected_ids,
    )

    feature_df.to_csv(artifact_path, index=False)
    columns = [str(column) for column in feature_df.columns]
    generated_columns = [column for column in columns if column != contract.entity_key]

    artifact = UpliftFeatureArtifact(
        feature_recipe_id=recipe.feature_recipe_id,
        feature_artifact_id=feature_artifact_id,
        dataset_fingerprint=dataset_fingerprint,
        builder_version=recipe.builder_version,
        artifact_path=str(artifact_path),
        metadata_path=str(metadata_path),
        cohort=cohort,
        entity_key=contract.entity_key,
        reference_date=reference_date,
        row_count=len(feature_df),
        columns=columns,
        generated_columns=generated_columns,
        source_tables=recipe.source_tables,
        windows_days=recipe.windows_days,
    )
    metadata_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return artifact
