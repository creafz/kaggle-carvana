import os


def data_path(path):
    return os.path.join('../input/', path)


TRAIN_IMAGES_HQ_PATH = data_path('train_hq')
TRAIN_MASKS_PATH = data_path('train_masks')
TEST_IMAGES_PATH = data_path('test_hq')
SAMPLE_SUBMISSION_PATH = data_path('sample_submission.csv')
SUBMISSIONS_PATH = '../submissions'
SAVE_MODEL_PATH = '../saved_models'
PREDICTIONS_PATH = '../predictions'
VALID_PREDICTIONS_PATH = '../valid_predictions'


NUM_WORKERS = 4
SEED = 42
CRAYON_HOSTNAME = 'http://127.0.0.1'


IMG_MEAN = [0.698228, 0.690886, 0.683951]
IMG_STD = [0.244182, 0.248307, 0.245187]