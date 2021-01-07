from model.model import SOLO
from train.loss import SOLOLoss
from data.tfrecord_decode import Parser
from config import *
import argparse
from datetime import datetime
import time
import os
import tensorflow as tf
from tensorflow.keras.utils import Progbar

tf.config.run_functions_eagerly(False)   # for debugging

@tf.function
def train_step(model, loss_fn, optimizer, images, cat_true, mask_true, cat_metric, mask_metric):
    with tf.GradientTape() as tape:
        cat_pred, mask_pred = model(image, training=True)
        total_loss, l_cate, l_mask = loss_fn((cat_true, mask_true), (cat_pred, mask_pred))
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    cat_metric.update_state(cat_true, cat_pred)
    mask_metric.update_state(mask_true, mask_pred)

    return total_loss, l_cate, l_mask


@tf.function
def test_step(model, loss_fn, images, cat_true, mask_true, cat_metric, mask_metric):
    cat_pred, mask_pred = model(image, training=False)
    total_loss, l_cate, l_mask = loss_fn(cat_true, mask_true, cat_pred, mask_pred)

    cat_metric.update_state(cat_true, cat_pred)
    mask_metric.update_state(mask_true, mask_pred)

    return total_loss, l_cate, l_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SOLO network training script')
    parser.add_argument("--dataset_train", type=str,
                        help="path to training dataset tfrecord BASE path")
    parser.add_argument("--dataset_val", type=str,
                        help="path to validation dataset tfrecord BASE path")
    args = parser.parse_args()

    print("Training SOLO network")
    display_config("train")

    # Load model
    model = SOLO(**MODEL_HYPERPARAMETERS)

    # add weight decay
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(lambda: tf.keras.regularizers.l2(TRAINING_PARAMETERS['weight_decay'])(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: tf.keras.regularizers.l2(TRAINING_PARAMETERS['weight_decay'])(layer.bias))

    # Training scheme
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=tf.math.multiply(TRAINING_PARAMETERS['epochs'], TRAINING_PARAMETERS['steps_per_epoch']),
                                                                       values=tf.constant(TRAINING_PARAMETERS['learning_rates']))
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=TRAINING_PARAMETERS['momentum'])
    loss_fn = SOLOLoss()

    # Load data
    train_parser = Parser(MODEL_HYPERPARAMETERS['input_size'],
                          MODEL_HYPERPARAMETERS['grid_sizes'][0],
                          MODEL_HYPERPARAMETERS['num_class'],
                          mode='train')
    val_parser = Parser(MODEL_HYPERPARAMETERS['input_size'],
                        MODEL_HYPERPARAMETERS['grid_sizes'][0],
                        MODEL_HYPERPARAMETERS['num_class'],
                        mode='val')
    train_dataset = train_parser.build_dataset(args.dataset_train,
                                               batch_size=TRAINING_PARAMETERS['batch_size'],
                                               num_epoch=TRAINING_PARAMETERS['num_epoch'])
    val_dataset = val_parser.build_dataset(args.dataset_val)

    """Training using built-in method
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join('logs', model.model_name), update_freq='batch')
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('weights', model.model_name, 'weight_' + model.model_name + '.h5'),
                                                       save_best_only=True,
                                                       save_weights_only=True)
    model.compile(optimizer=optimizer,
                  loss=[loss_fn.get_category_loss(), loss_fn.get_mask_loss()],
                  loss_weights=loss_fn.weights,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.MeanIoU(num_classes=MODEL_HYPERPARAMETERS['num_class'])])
    model.fit(x=train_dataset,
              batch_size=TRAINING_PARAMETERS['batch_size'],
              epochs=TRAINING_PARAMETERS['num_epoch'],
              shuffle=True,
              steps_per_epoch=TRAINING_PARAMETERS['steps_per_epoch'],
              validation_data=val_dataset,
              validation_batch_size=TRAINING_PARAMETERS['batch_size'],
              verbose=1,
              callbacks=[tb_callback, ckpt_callback])
    """

    # Training using low-level API
    # Load/create Checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(-1, trainable=False, dtype=tf.int64),
                               optimizer=optimizer,
                               model=model,
                               metric=tf.Variable(1000, trainable=False, dtype=tf.float32))
    manager = tf.train.CheckpointManager(ckpt, os.path.join('checkpoints', model.model_name), max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # Define Losses
    train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
    train_cat_loss = tf.keras.metrics.Mean(name='train_cat_loss', dtype=tf.float32)
    train_mask_loss = tf.keras.metrics.Mean(name='train_mask_loss', dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)
    val_cat_loss = tf.keras.metrics.Mean(name='val_cat_loss', dtype=tf.float32)
    val_mask_loss = tf.keras.metrics.Mean(name='val_mask_loss', dtype=tf.float32)

    # Define metrics
    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc', dtype=tf.float32)
    train_meaniou = tf.keras.metrics.MeanIoU(num_classes=2, name='train_meaniou', dtype=tf.float32)
    val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc', dtype=tf.float32)
    val_meaniou = tf.keras.metrics.MeanIoU(num_classes=2, name='val_meaniou', dtype=tf.float32)

    # Create logger
    log_dir = os.path.join('logs', model.model_name, datetime.now().strftime("%Y%m%d%H%M%S"))
    summary_writer = tf.summary.create_file_writer(log_dir)

    step = ckpt.step.numpy()
    val_metric = ckpt.metric.numpy()
    total_val_sample = 5000
    progbar = None
    start_time = time.perf_counter()

    # Start training
    for image, cat_true, mask_true in train_dataset:
        ckpt.step.assign_add(1)
        step += 1

        # On epoch start
        epoch_step = (step % TRAINING_PARAMETERS['steps_per_epoch']) + 1
        if epoch_step == 1:
            print("Epoch {}/{}".format((step // TRAINING_PARAMETERS['steps_per_epoch']) + 1, TRAINING_PARAMETERS['num_epoch']))
            progbar = Progbar(TRAINING_PARAMETERS['steps_per_epoch'], interval=1, stateful_metrics=['train_acc', 'train_meaniou'])

        total_loss, l_cate, l_mask = train_step(model,
                                                optimizer,
                                                loss_fn,
                                                image,
                                                cat_true,
                                                mask_true,
                                                train_acc,
                                                train_meaniou)
        values = [('train_loss', total_loss),
                  ('train_cat_loss',  l_cate),
                  ('train_mask_loss', l_mask),
                  ('train_acc', train_acc.result()),
                  ('train_meaniou', train_meaniou.result())]
        progbar.update(epoch_step, values)

        train_loss.update_state(total_loss)
        train_cat_loss.update_state(l_cate)
        train_mask_loss.update_state(l_mask)
        with summary_writer.as_default():
            tf.summary.scalar('train loss', train_loss.result(), step=step)
            tf.summary.scalar('train category loss', train_cat_loss.result(), step=step)
            tf.summary.scalar('train mask loss', train_mask_loss.result(), step=step)
            tf.summary.scalar('train accuracy', train_acc.result(), step=step)
            tf.summary.scalar('train mean IoU', train_meaniou.result(), step=step)

        # On epoch end
        if epoch_step == TRAINING_PARAMETERS['steps_per_epoch']:
            # Save checkpoint (weights, optimizer states)
            save_path = manager.save()
            print("Saved checkpoint: {}. Loss: {:1.2f}, acc: {:1.2f}, meanIoU: {:1.2f}".format(save_path, train_loss.result(), train_acc.result(), train_meaniou.result()))

            #  Validation
            print("Start validation...")
            val_progbar = Progbar(total_val_sample, interval=1, stateful_metrics=['val_acc', 'val_meaniou'])
            val_step = 0
            for image, cat_true, mask_true in val_dataset:
                val_step += 1
                total_loss, l_cate, l_mask = test_step(model,
                                                       loss_fn,
                                                       image,
                                                       cat_true,
                                                       mask_true,
                                                       val_acc,
                                                       val_meaniou)
                values = [('val_loss', total_loss),
                          ('val_cat_loss',  l_cate),
                          ('val_mask_loss', l_mask),
                          ('val_acc', val_acc.result()),
                          ('val_meaniou', val_meaniou.result())]
                val_progbar.update(val_step, values)

                val_loss.update_state(total_loss)
                val_cat_loss.update_state(l_cate)
                val_mask_loss.update_state(l_mask)
            with summary_writer.as_default():
                tf.summary.scalar('validation loss', val_loss.result(), step=step)
                tf.summary.scalar('validation category loss', val_cat_loss.result(), step=step)
                tf.summary.scalar('validation mask loss', val_mask_loss.result(), step=step)
                tf.summary.scalar('validation accuracy', val_acc.result(), step=step)
                tf.summary.scalar('validation mean IoU', val_meaniou.result(), step=step)

            # Save new best weight
            new_metric = (val_acc.result() + val_meaniou.result()) / 2
            if val_metric < new_metric:
                val_metric = new_metric
                ckpt.metric.assign(new_metric)
                weight_path = os.path.join('weights', model.model_name, 'weight_{}_{}_{}_{}_{}_{}_{}_{}.h5'.format(model.model_name, model.num_class, model.input_size, '_'.join([str(i) for i in model.grid_sizes]), model.head_style, model.head_depth, model.fpn_channel, new_metric))
                print("Val acc: {}, Val meaniou: {}. Saving weight to {}...".format(val_acc.result(), val_meaniou.result(), weight_path))
                model.save_weights(weight_path)
            total_val_sample = val_step

        # Reset metrics state
        train_loss.reset_states()
        train_cat_loss.reset_states()
        train_mask_loss.reset_states()
        val_loss.reset_states()
        val_cat_loss.reset_states()
        val_mask_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()
        train_meaniou.reset_states()
        val_meaniou.reset_states()

    train_time = int(time.perf_counter() - start_time)
    train_hour = train_time // 3600
    train_time = train_time % 3600
    train_minute = train_time // 60
    train_second = train_time % 60
    print("Total training time: {} h {} m {} s".format(train_hour, train_minute, train_second))
