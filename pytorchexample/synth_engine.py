"""pytorchexample.synth_engine: Synthetic Data Generation via SDV / CTGAN.

Experiment 5 (FedSynAI) uses this module to augment client data with
privacy-safe synthetic rows before local training.

Toggle via pyproject.toml:
    use-synthetic = true
"""

from __future__ import annotations

import pandas as pd


def generate_synthetic(
    real_df: pd.DataFrame,
    target_col: str = "label",
    num_rows: int | None = None,
    model_type: str = "ctgan",
    epochs: int = 10,
) -> pd.DataFrame:
    """Train a generative model on real_df and return synthetic rows.

    Supports both SDV 2.x (Metadata) and SDV 1.x (SingleTableMetadata).

    Args:
        real_df:    The client's local partition as a pandas DataFrame.
        target_col: Name of the label / target column.
        num_rows:   Number of synthetic rows to generate (defaults to len(real_df)).
        model_type: One of "ctgan" (default) or "gaussian_copula".
        epochs:     Training epochs for CTGAN (default: 10 for fast federated simulation).

    Returns:
        A DataFrame of synthetic samples with identical columns to real_df.
    """
    try:
        from sdv.metadata import SingleTableMetadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_df)
    except (ImportError, AttributeError):
        try:
            from sdv.metadata import Metadata
            metadata = Metadata.detect_from_dataframes(data={"data": real_df})
        except ImportError as exc:
            raise ImportError(
                "sdv is required for synthetic data generation. "
                "Install it with: pip install sdv"
            ) from exc

    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

    if num_rows is None:
        num_rows = len(real_df)

    print(f"[SynthEngine] Training {model_type.upper()} on {len(real_df)} real rows...")

    if model_type == "ctgan":
        synthesizer = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    elif model_type == "gaussian_copula":
        synthesizer = GaussianCopulaSynthesizer(metadata)
    else:
        raise ValueError(f"Unsupported model_type='{model_type}'")

    try:
        synthesizer.fit(real_df)
    except Exception:
        synthesizer.fit({"data": real_df})

    try:
        sampled = synthesizer.sample(num_rows=num_rows)
    except Exception:
        sampled = synthesizer.sample({"data": num_rows})

    if isinstance(sampled, dict):
        synthetic_df = sampled.get("data", list(sampled.values())[0])
    else:
        synthetic_df = sampled

    # Ensure target column is long integer type if classification label
    if target_col in synthetic_df.columns and target_col in real_df.columns:
        if pd.api.types.is_integer_dtype(real_df[target_col]):
            synthetic_df[target_col] = synthetic_df[target_col].round().astype(int)

    # Align columns to match real_df
    synthetic_df = synthetic_df[real_df.columns]
    print(f"[SynthEngine] Generated {len(synthetic_df)} synthetic rows.")
    return synthetic_df


def augment_partition(
    real_df: pd.DataFrame,
    target_col: str = "label",
    synth_ratio: float = 1.0,
    model_type: str = "ctgan",
) -> pd.DataFrame:
    """Augment a real DataFrame with synthetic rows and return the combined set.

    Args:
        real_df:     Client's local data.
        target_col:  Label column name.
        synth_ratio: Fraction of len(real_df) to generate as synthetic.
                     1.0 = match real size; 0.5 = add half as many synthetic rows.
        model_type:  Generative model to use (``"ctgan"`` or ``"gaussian_copula"``).

    Returns:
        Concatenated (real + synthetic) DataFrame, shuffled.
    """
    num_synth = int(len(real_df) * synth_ratio)
    synth_df = generate_synthetic(
        real_df, target_col=target_col, num_rows=num_synth, model_type=model_type
    )
    combined = pd.concat([real_df, synth_df], ignore_index=True).sample(frac=1, random_state=42)
    print(
        f"[SynthEngine] Augmented partition: {len(real_df)} real + "
        f"{num_synth} synthetic = {len(combined)} total rows."
    )
    return combined
