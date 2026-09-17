from datetime import datetime
from pathlib import Path

from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve


def plot_ROC(y, y_p, acc, testpath, pklpath, output_path='outputs/evaluation.txt'):
    """Compute ROC statistics and append a compact evaluation record."""
    fpr, tpr, thresholds = roc_curve(y, y_p)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = float(interp1d(fpr, thresholds)(eer))

    result_path = Path(output_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open('a', encoding='utf-8') as stream:
        stream.write('\n' + '=' * 50 + '\n')
        stream.write(f'Test run: {datetime.now().isoformat(timespec="seconds")}\n')
        stream.write(f'Test list: {testpath}\n')
        stream.write(f'Checkpoint: {pklpath}\n')
        stream.write(f'Accuracy@0.5: {acc:.6f}\n')
        stream.write(f'AUC: {roc_auc:.6f}\n')
        stream.write(f'EER: {eer:.6f}\n')
        stream.write(f'EER threshold: {threshold:.6f}\n')

    return eer, roc_auc
