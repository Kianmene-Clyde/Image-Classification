import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from models.python.MLP_multiclass import MLP
from models.python.RBF import RBF
from models.python.SVM import SVM
from models.python.linearModel import LinearModel
from sklearn.metrics import accuracy_score
import keras
import matplotlib.pyplot as plt


def image_transform(image_dir):
    image_vector = []
    image_labels = []
    image_names = []
    classe_name = ['limitation_vitesse', 'interdit_sens', 'stop']

    for image_name in os.listdir(image_dir):
        if image_name.endswith((".jpg", ".png", ".JPG", ".jpeg", ".JPEG", ".PNG")):
            # labels
            if image_name[0] == "l":
                image_labels.append([classe_name.index('limitation_vitesse')])
            elif image_name[0] == "i":
                image_labels.append([classe_name.index('interdit_sens')])
            else:
                image_labels.append([classe_name.index('stop')])

            # image transform
            image_path = os.path.join(image_dir, image_name)
            try:
                imaging = Image.open(image_path)
                imaging = imaging.convert("RGB")
                imaging = imaging.resize((6, 6))
                pixel_values = list(imaging.getdata())
                if len(pixel_values[0]) == 4:
                    pixel_values = [(r, g, b) for r, g, b, _ in pixel_values]
                image_vector.append(pixel_values)
                image_names.append(image_name)
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {image_name}: {e}")

    return (np.array(image_vector), np.array(image_labels), image_names)


def shape_verification(np_array: np.array):
    dimension = len(np_array.shape)
    if dimension == 3:
        np_array = np.reshape(np_array, (np_array.shape[0], np_array.shape[1] * np_array.shape[2]))

    return np_array


def rename_files(directory, new_name_prefix):
    if not os.path.exists(directory):
        print("Le répertoire spécifié n'existe pas.")
        return

    files = os.listdir(directory)

    for i, file_name in enumerate(files):
        extension = os.path.splitext(file_name)[1]
        new_file_name = f"{new_name_prefix}_{i}{extension}"
        new_file_path = os.path.join(directory, new_file_name)
        old_file_path = os.path.join(directory, file_name)

        if os.path.exists(new_file_path):
            count = 1
            while os.path.exists(new_file_path):
                new_file_name = f"{new_name_prefix}_{i}_{count}{extension}"
                new_file_path = os.path.join(directory, new_file_name)
                count += 1

        os.rename(old_file_path, new_file_path)
        print(f"Renommage de {file_name} en {new_file_name}")


def add_extension_to_images(directory, extension=".png"):
    if not os.path.exists(directory):
        print("Le répertoire spécifié n'existe pas.")
        return

    files = os.listdir(directory)

    for file_name in files:
        old_file_path = os.path.join(directory, file_name)

        # Ignore directories
        if os.path.isdir(old_file_path):
            continue

        # Split the file name and extension
        name, ext = os.path.splitext(file_name)

        # Check if the file has no extension and is not hidden (name should not start with a dot)
        if ext in ["", "html", "webp"] and not name.startswith('.'):
            new_file_name = f"{file_name}{extension}"
            new_file_path = os.path.join(directory, new_file_name)

            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renommage de {file_name} en {new_file_name}")


if __name__ == '__main__':

    """CODE TO RENAME ALL FILES IN A DIRECTORY"""
    directory_path = "D:/school/ProjetAnnuel2024_datasets/test interdit"
    # add_extension_to_images(directory_path, ".png")

    # directory_path = "D:\school\ProjetAnnuel2024\dataset_for_labeling\sense_interdit"
    new_name_prefix = "interdit"
    # rename_files(directory_path, new_name_prefix)

    # ACTUELLEMENT POUR LIMITATION VITESSE J'AI 5067, STOP 3458, INTERDIT 3215

    """DATASET (IMAGES) NORMALISATION AND DISTRIBUTION"""
    # il faut mettre le repertoire de tout le dataset, training et test
    full_dataset_labels_img_names = image_transform("D:/school/ProjetAnnuel2024_datasets/full_dataset")
    full_dataset = full_dataset_labels_img_names[0]
    full_labels = full_dataset_labels_img_names[1]
    full_names = full_dataset_labels_img_names[2]

    if len(full_dataset) != len(full_labels) or len(full_dataset) != len(full_names):
        raise ValueError("Les tailles des datasets sont incohérentes")

    training_data, test_data, training_labels, test_labels, training_names, test_names = train_test_split(
        full_dataset, full_labels, full_names, test_size=0.3, random_state=42)

    training_data = shape_verification(training_data)
    test_data = shape_verification(test_data)

    train_mean = np.mean(training_data, axis=0)
    train_std = np.std(training_data, axis=0)

    training_data = (training_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    training_labels = np.squeeze(training_labels)
    test_labels = np.squeeze(test_labels)

    # print(f"Training data shape: {training_data.shape}\n Test data shape: {test_data.shape}\n "
    #     f"Training labels shape: {training_labels.shape}\n Test labels shape: {test_labels.shape}")

    inputs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    labels = np.array([1, 1, 0, 0])

    # oublie pas que tu dois changer les classes de tes labels en 0 ou 1
    # tu dois toujours mettre les labels en float parce à cause de tanh et parce que en C il s'attend à des float

    base_logdir = "./logs"
    seeds = [20, 17, 7, 3, 28]
    classes = ['limitation_vitesse', 'interdit_sens', 'stop']

    for seed in seeds:
        """RBF"""
        MyRBF = RBF(num_of_classes=3, k_num_clusters=99, learning_rate=0.001, gamma=0.01, epochs=300,
                    std_from_clusters=False, is_classified=True)
        experiment_name = f"conf_seed_{seed}_num_clusters_{MyRBF.k_num_clusters}_gamma_{MyRBF.gamma}_epochs_{MyRBF.epochs}"
        final_logdir = os.path.join(f"{base_logdir}/RBF_experiments/{experiment_name}")
        keras.utils.set_random_seed(seed)

        # MyRBF.fit(training_data, training_labels, test_inputs=test_data, test_labels=test_labels, logdir=final_logdir)
        # MyRBF.save_data_json('rbf_model.json')
        MyRBF.load_data_json('rbf_model.json')

        print(test_labels[0], test_names[0])
        print(f"RBF final predictions for seed {seed}\n {MyRBF.predict(test_data[0])}")

        '''END TEST RBF'''

    exit(0)

    '''TEST SVM'''

    MySVM = SVM(0.01, 0.01, 100, 3)
    experiment_name = f"conf_seed_{seed}_lr_{MySVM.learning_rate}_lambda_{MySVM.lambda_param}_epochs_{MySVM.epochs}"
    final_logdir = os.path.join(f"{base_logdir}/svm_experiments/{experiment_name}")
    keras.utils.set_random_seed(seed)

    # MySVM.fit(training_data, training_labels, test_data, test_labels, logdir=final_logdir)
    # MySVM.save_data_json('svm_model.json')
    MySVM.load_data_json('svm_model.json')
    prediction = MySVM.predict(test_data[0])
    print(test_labels[0], test_names[0])

    print(f"SVM final prediction\n {prediction}")

    '''END TEST SVM'''

    """MLP"""

    MyMLP = MLP([108, 120, 120, 3], epochs=100, learning_rate=0.01, is_classification=True)
    experiment_name = f"conf_seed_{seed}_structure_{MyMLP.structure}_lr_{MyMLP.learning_rate}_epochs_{MyMLP.epochs}"
    final_logdir = os.path.join(f"{base_logdir}/MLP_experiments/{experiment_name}")
    keras.utils.set_random_seed(seed)

    training_labels = [[label] for label in training_labels]
    test_labels = [[label] for label in test_labels]

    MyMLP.fit(training_data, training_labels, test_data, test_labels, logdir=final_logdir)
    MyMLP.save_data_json('mlp_model.json')
    MyMLP.load_data_json('mlp_model.json')
    predictions = MyMLP.predict_all(test_data, test_labels)

    print(f"final predictions for seed {seed} \n {predictions}\n")

    '''END TEST MLP'''

    """LINEAR MODEL"""

    MyLM = LinearModel(learning_rate, epochs, classification)
    print(f"{inputs}, {labels}")
    MyLM.fit(inputs, labels)
    print(f"final prediction\n{MyLM.predict(inputs)}")

    '''END TEST LINEAR MODEL'''
