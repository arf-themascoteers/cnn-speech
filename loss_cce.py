from loss import Loss
import numpy as np

class LossCategoricalCrossentropy ( Loss ):
    def forward ( self , y_pred , y_true ):
        samples = len (y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )
        correct_confidences = y_pred_clipped[range(samples),y_true]
        negative_log_likelihoods = - np.log(correct_confidences)
        return negative_log_likelihoods
