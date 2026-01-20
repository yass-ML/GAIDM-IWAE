"""
Checkpoint utilities for auto-discovery and centralized results storage.
"""

import os
import re
import yaml
from glob import glob
from typing import Dict, List, Optional, Any


def discover_checkpoints(checkpoint_dir: str = 'checkpoints') -> List[Dict[str, Any]]:
    """
    Discover all checkpoint files and parse metadata from filenames.

    Expected filename format: {model}_{k}{K}_epochs{E}_seed{S}.pt
    Examples:
        - vae_k1_epochs50_seed42.pt
        - iwae_k5_epochs50_seed42.pt
        - iwae_k20_epochs50_seed42.pt

    Returns:
        List of dicts with keys: name, path, type, k, epochs, seed
    """
    checkpoints = []
    pattern = os.path.join(checkpoint_dir, '*.pt')

    # Regex to parse checkpoint filenames
    filename_regex = re.compile(
        r'^(?P<model_type>vae|iwae)_k(?P<k>\d+)_epochs(?P<epochs>\d+)_seed(?P<seed>\d+)\.pt$'
    )

    for filepath in sorted(glob(pattern)):
        filename = os.path.basename(filepath)
        match = filename_regex.match(filename)

        if match:
            model_type = match.group('model_type')
            k = int(match.group('k'))
            epochs = int(match.group('epochs'))
            seed = int(match.group('seed'))

            # Create human-readable name
            if model_type == 'vae':
                name = f'VAE (K={k})'
            else:
                name = f'IWAE (K={k})'

            checkpoints.append({
                'name': name,
                'path': filepath,
                'type': model_type,
                'k': k,
                'epochs': epochs,
                'seed': seed
            })
        else:
            # Non-standard filename, include with minimal info
            print(f"Warning: Could not parse checkpoint filename: {filename}")
            checkpoints.append({
                'name': filename.replace('.pt', ''),
                'path': filepath,
                'type': 'unknown',
                'k': None,
                'epochs': None,
                'seed': None
            })

    return checkpoints


def get_model_key(checkpoint_path: str) -> str:
    """Get a unique key for a model from its checkpoint path."""
    return os.path.basename(checkpoint_path).replace('.pt', '')


def load_results(results_path: str = 'results/evaluations.yaml') -> Dict[str, Any]:
    """
    Load evaluation results from YAML file.

    Returns:
        Dict mapping model keys to their evaluation metrics
    """
    if not os.path.exists(results_path):
        return {}

    with open(results_path, 'r') as f:
        data = yaml.safe_load(f)

    return data if data else {}


def _convert_to_native(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for YAML serialization.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_results(
    results_path: str,
    model_key: str,
    metrics: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save or update evaluation results for a model in the YAML file.

    Args:
        results_path: Path to the YAML file
        model_key: Unique identifier for the model (typically filename without .pt)
        metrics: Dict of metric names to values (e.g., {'log_likelihood': -81.06})
        model_info: Optional dict with model metadata (type, k, path, etc.)
    """
    # Load existing results
    results = load_results(results_path)

    # Initialize entry if doesn't exist
    if model_key not in results:
        results[model_key] = {}

    # Update with model info if provided (convert numpy types)
    if model_info:
        results[model_key].update(_convert_to_native(model_info))

    # Update with new metrics (convert numpy types)
    results[model_key].update(_convert_to_native(metrics))

    # Ensure directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Save back to YAML
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def get_models_config(
    checkpoint_dir: str = 'checkpoints',
    results_path: str = 'results/evaluations.yaml'
) -> List[Dict[str, Any]]:
    """
    Get full model configurations by merging checkpoint discovery with stored results.

    This is the main function that analysis scripts should use to get model configs
    instead of hardcoding them.

    Returns:
        List of model configs with all available metrics
    """
    # Discover checkpoints
    checkpoints = discover_checkpoints(checkpoint_dir)

    # Load stored results
    results = load_results(results_path)

    # Merge checkpoint info with stored results
    models_config = []
    for checkpoint in checkpoints:
        model_key = get_model_key(checkpoint['path'])
        config = checkpoint.copy()

        # Add stored metrics if available
        if model_key in results:
            stored = results[model_key]
            # Add metrics that aren't already in config
            for key, value in stored.items():
                if key not in config:
                    config[key] = value

        models_config.append(config)

    return models_config


if __name__ == '__main__':
    # Quick test
    print("Discovering checkpoints...")
    checkpoints = discover_checkpoints('checkpoints')
    for cp in checkpoints:
        print(f"  {cp['name']}: {cp['path']}")

    print("\nLoading results from YAML...")
    results = load_results('results/evaluations.yaml')
    print(f"  Found {len(results)} model entries")

    print("\nMerged model configs:")
    configs = get_models_config()
    for cfg in configs:
        print(f"  {cfg['name']}: K={cfg['k']}, LL={cfg.get('log_likelihood', 'N/A')}")
