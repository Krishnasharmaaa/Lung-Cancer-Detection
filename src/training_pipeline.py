"""
Training pipeline for cancer detection models.
This module orchestrates the entire training process:
1. Loading dataset paths and labels.
2. Encoding labels.
3. Creating data generators (with augmentation for training).
4. Building or loading the model architecture.
5. Defining Keras callbacks.
6. Training the model.
7. Saving the trained model, history, and label encoder.
8. Evaluating the model on the test set.
"""
import os
import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model # For loading the best model for evaluation

from config import (TARGET_IMAGE_SIZE, RANDOM_STATE,
                     EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, 
                     MONITOR_METRIC, MIN_DELTA_EARLY_STOPPING,
                     TENSORBOARD_LOG_DIR_COLON, TENSORBOARD_LOG_DIR_LUNG, # For TensorBoard
                     get_logger)
from data_preprocessing import (load_image_paths_and_labels_df, 
                                 create_and_save_label_encoder, 
                                 prepare_data_generators)
from model_architecture import get_colon_cancer_model, get_lung_cancer_model , build_vgg16_transfer_model
from evaluation import evaluate_full_model_performance
from utils import save_pickle_object, save_model_summary_and_plot, load_pickle_object

logger = get_logger(__name__)

def execute_training_pipeline(
    model_type_str, 
    dataset_base_dir, 
    defined_num_classes, 
    defined_class_names, 
    target_model_path, 
    training_history_path, 
    label_enc_path, 
    model_report_dir,
    num_epochs, 
    train_batch_size, 
    use_transfer_learning_flag=False
):
    logger.info(f"===== Starting Training Pipeline for {model_type_str.upper()} Cancer Model =====")
    pipeline_start_time = time.time()

    # ✅ FIX 1: Ensure model save directory exists
    model_dir = os.path.dirname(target_model_path)
    os.makedirs(model_dir, exist_ok=True)

    # --- 1. Load Dataset Paths and Labels ---
    logger.info(f"Step 1: Loading dataset from: {dataset_base_dir}")
    dataframe_images = load_image_paths_and_labels_df(dataset_base_dir, defined_class_names)
    if dataframe_images is None or dataframe_images.empty:
        logger.error(f"Failed to load dataset for {model_type_str}. Aborting training pipeline.")
        return None, None

    # --- 2. Encode Labels ---
    logger.info(f"Step 2: Encoding labels...")
    label_encoder_instance, _ = create_and_save_label_encoder(
        dataframe_images['label'], label_enc_path
    )
    if label_encoder_instance is None:
        return None, None

    actual_classes_from_encoder = list(label_encoder_instance.classes_)
    num_classes_for_model = len(actual_classes_from_encoder)

    # --- 3. Data Generators ---
    train_gen, val_gen, test_gen, X_test_data, y_test_labels_cat = prepare_data_generators(
        dataframe_images,
        label_encoder_instance,
        target_img_size=TARGET_IMAGE_SIZE,
        batch_sz=train_batch_size
    )

    # --- 4. Build Model ---
    model_input_shape = (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3)

    if model_type_str == 'colon':
        keras_model = get_colon_cancer_model(model_input_shape, num_classes_for_model)
    else:
        keras_model = get_lung_cancer_model(model_input_shape, num_classes_for_model)

    # --- 5. Callbacks ---
    checkpoint_cb = ModelCheckpoint(
        filepath=target_model_path,
        monitor=MONITOR_METRIC,
        save_best_only=True,
        save_weights_only=False,
        mode='max' if 'accuracy' in MONITOR_METRIC else 'min',
        verbose=1
    )

    early_stopping_cb = EarlyStopping(
        monitor=MONITOR_METRIC,
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        min_delta=MIN_DELTA_EARLY_STOPPING,
        verbose=1,
        mode='max' if 'accuracy' in MONITOR_METRIC else 'min'
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor=MONITOR_METRIC,
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        verbose=1,
        mode='max' if 'accuracy' in MONITOR_METRIC else 'min'
    )

    callbacks_list = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

    # --- 6. Train ---
    history_object = keras_model.fit(
        train_gen,
        epochs=num_epochs,
        validation_data=val_gen,
        callbacks=callbacks_list,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen) if val_gen else None
    )

    training_history = history_object.history

    # --- 7. Save History ---
    save_pickle_object(training_history, training_history_path)

    # ✅ FIX 2 (MOST IMPORTANT): Explicitly save trained model
    try:
        keras_model.save(target_model_path)
        logger.info(f"Final trained model explicitly saved at: {target_model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # --- 8. Evaluation ---
    best_model_for_eval = load_model(target_model_path)
    evaluate_full_model_performance(
        model=best_model_for_eval,
        test_data_x=X_test_data,
        test_data_y_cat=y_test_labels_cat,
        test_generator=test_gen,
        class_label_names=actual_classes_from_encoder,
        train_history_data=training_history,
        model_type_name=model_type_str,
        reports_output_dir=model_report_dir,
        use_test_generator_for_eval=(test_gen is not None and test_gen.n > 0)
    )

    logger.info(f"===== {model_type_str.upper()} Cancer Model Training Finished =====")
    return best_model_for_eval, training_history



if __name__ == '__main__':
    # This block is for testing the training pipeline directly.
    # It requires dummy data setup similar to data_preprocessing.py's self-test or actual data.
    logger.info("Initiating a self-test run of the training_pipeline.py.")

    # --- Configuration for self-test ---
    # Assuming config.py is in the same parent directory (src)
    from .config import (COLON_DATA_DIR, COLON_NUM_CLASSES, COLON_CLASSES, COLON_MODEL_PATH, 
                         COLON_HISTORY_PATH, COLON_LABEL_ENCODER_PATH, COLON_DEFAULT_EPOCHS, 
                         COLON_DEFAULT_BATCH_SIZE, COLON_REPORT_DIR, PROJECT_ROOT)

    # Create minimal dummy data for the colon model to run the pipeline
    from PIL import Image # For creating dummy images
    
    def setup_minimal_pipeline_test_data(data_dir_path, class_list, num_img_per_cls=30): # Enough for splits
        logger.info(f"Setting up minimal dummy data in {data_dir_path} for pipeline test.")
        for cls_item in class_list:
            cls_dir_path = os.path.join(data_dir_path, cls_item)
            os.makedirs(cls_dir_path, exist_ok=True)
            for i_img in range(num_img_per_cls):
                try:
                    # Create a tiny valid PNG image
                    img_obj = Image.new('RGB', (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1]), color='cyan')
                    img_obj.save(os.path.join(cls_dir_path, f'pipeline_dummy_{cls_item}_{i_img}.png'))
                except Exception as e_img:
                    logger.error(f"Failed to create dummy image {i_img} for {cls_item}: {e_img}")
        logger.info("Minimal dummy data for pipeline test setup complete.")

    # Use a temporary directory for test outputs to avoid overwriting production models/reports
    test_model_dir = os.path.join(PROJECT_ROOT, 'trained_models', 'pipeline_test')
    test_report_dir_colon = os.path.join(PROJECT_ROOT, 'reports', 'colon', 'pipeline_test')
    os.makedirs(test_model_dir, exist_ok=True)
    os.makedirs(test_report_dir_colon, exist_ok=True)

    # Modify paths for test
    test_colon_model_path = os.path.join(test_model_dir, 'colon_cancer_model_pipeline_test.keras')
    test_colon_history_path = os.path.join(test_model_dir, 'colon_model_history_pipeline_test.pkl')
    test_colon_encoder_path = os.path.join(test_model_dir, 'colon_label_encoder_pipeline_test.pkl')
    
    # Setup dummy data specifically for colon (as an example)
    # Ensure COLON_DATA_DIR points to a location where dummy data can be created, or use a dedicated test data dir.
    # For this test, let's use a subdirectory within the existing COLON_DATA_DIR if it's safe,
    # or better, a completely separate dummy data path.
    dummy_colon_test_data_dir = os.path.join(PROJECT_ROOT, 'data', 'dummy_colon_for_pipeline_test')
    setup_minimal_pipeline_test_data(dummy_colon_test_data_dir, COLON_CLASSES)

    logger.info("Attempting to train COLON model with minimal dummy data for pipeline self-test...")
    # Use minimal epochs and batch size for quick testing
    test_epochs = 2 
    test_batch_size = 4 

    trained_model_obj, history_data_dict = execute_training_pipeline(
        model_type_str='colon',
        dataset_base_dir=dummy_colon_test_data_dir, # Use the dedicated dummy data path
        defined_num_classes=COLON_NUM_CLASSES,
        defined_class_names=COLON_CLASSES,
        target_model_path=test_colon_model_path,
        training_history_path=test_colon_history_path,
        label_enc_path=test_colon_encoder_path,
        model_report_dir=test_report_dir_colon, # Specific test report dir
        num_epochs=test_epochs,
        train_batch_size=test_batch_size,
        use_transfer_learning_flag=False # Test custom CNN first
    )

    if trained_model_obj and history_data_dict:
        logger.info("Colon model pipeline self-test completed successfully.")
        logger.info(f"Test model saved to: {test_colon_model_path}")
        logger.info(f"Test reports/plots in: {test_report_dir_colon}")
    else:
        logger.error("Colon model pipeline self-test failed.")

    # --- Cleanup of dummy data and test outputs (optional) ---
    import shutil
    if os.path.exists(dummy_colon_test_data_dir):
        shutil.rmtree(dummy_colon_test_data_dir)
        logger.info(f"Cleaned up dummy data directory: {dummy_colon_test_data_dir}")
    # Test output models/reports in 'pipeline_test' subdirs can be manually reviewed/deleted.
    # if os.path.exists(test_model_dir): shutil.rmtree(test_model_dir)
    # if os.path.exists(test_report_dir_colon): shutil.rmtree(test_report_dir_colon)
    
    logger.info("Training pipeline self-test finished.")
