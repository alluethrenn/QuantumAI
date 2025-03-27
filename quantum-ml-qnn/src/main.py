from qnn.models import QuantumModel
from utils.helpers import load_data, evaluate_model

def main():
    # Load the dataset
    data = load_data('path/to/dataset')


    # Initialize the quantum model
    model = QuantumModel()

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(data['train'], data['train_labels'], epochs=10)

    # Evaluate the model
    results = evaluate_model(model, data['test'], data['test_labels'])
    print(f"Model evaluation results: {results}")

if __name__ == "__main__":
    main()