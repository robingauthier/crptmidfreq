from gluonts.dataset.common import ListDataset, load_datasets
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.n_beats import NBEATSEnsembleEstimator
from gluonts.mx import SimpleFeedForwardEstimator, Trainer
from gluonts.trainer import Trainer

train_ds = ListDataset([
      {
          FieldName.TARGET: target,
          FieldName.START: start
      }
      for (target, start) in zip(train_target_values,
                                          m5_dates
                                          )
  ], freq="D")

prediction_length=12
estimator = NBEATSEnsembleEstimator(
    prediction_length=prediction_length,
    #context_length=7*prediction_length,
    meta_bagging_size = 3,  # 3, ## Change back to 10 after testing??
    meta_context_length = [prediction_length * mlp for mlp in [3,5,7] ], ## Change back to (2,7) // 3,5,7
    meta_loss_function = ['sMAPE'], ## Change back to all three MAPE, MASE ...
    num_stacks = 30,
    widths= [512],
    freq="D",
    trainer=Trainer(
                learning_rate=6e-4,
                #clip_gradient=1.0,
                epochs=12, #10
                num_batches_per_epoch=1000,
                batch_size=16
                #ctx=mx.context.gpu()
            )
    
predictor = estimator.train(train_ds)