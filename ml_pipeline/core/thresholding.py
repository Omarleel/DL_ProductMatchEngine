import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
class ThresholdOptimizer:
    @staticmethod
    def find_best(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, float]:
        best_th = 0.50
        best_score = -1.0
        best_precision = -1.0
        best_recall = -1.0

        for th in np.arange(0.30, 0.98, 0.01):
            y_hat = (probs >= th).astype(int)
            prec = precision_score(y_true, y_hat, zero_division=0)
            rec = recall_score(y_true, y_hat, zero_division=0)
            f1 = fbeta_score(y_true, y_hat, beta=1.0, zero_division=0)

            if (f1 > best_score) or (f1 == best_score and prec > best_precision):
                best_score = float(f1)
                best_precision = float(prec)
                best_recall = float(rec)
                best_th = float(th)

        return best_th, best_score, best_precision, best_recall