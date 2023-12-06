import tensorflow as tf
from FaceAging import FaceAging
from utils import show_all_variables

def create_or_load_input_z(batch_size, z_dim):
    return tf.keras.layers.Input(shape=(z_dim,), batch_size=batch_size, name='input_z')

def create_or_load_input_age(batch_size, num_categories):
    return tf.keras.layers.Input(shape=(num_categories,), batch_size=batch_size, name='input_age')

def create_or_load_input_gender(batch_size):
    return tf.keras.layers.Input(shape=(2,), batch_size=batch_size, name='input_gender')

def main():
    batch_size = 64
    z_dim = 100
    num_categories = 10

    input_z_tensor = create_or_load_input_z(batch_size, z_dim)
    input_age_tensor = create_or_load_input_age(batch_size, num_categories)
    input_gender_tensor = create_or_load_input_gender(batch_size)

    options = {
        'dataset': 'UTKFace',
        'savedir': 'save',
        'testdir': 'None',
        # Add other options as needed
    }

    # Assuming FaceAging class expects 'z', 'age', and 'gender' as input tensors
    model = FaceAging(
        dataset_name=options['dataset'],
        save_dir=options['savedir'],
        z=input_z_tensor,
        age=input_age_tensor,
        gender=input_gender_tensor
    )

    show_all_variables()

    model.train()

if __name__ == '__main__':
    main()
