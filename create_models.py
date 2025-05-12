from model_ensemble import ModelEnsemble

# Create ensemble instance
ensemble = ModelEnsemble()

# Create MNIST model
print("Creating MNIST model...")
ensemble.create_new_model(dataset_type='mnist')

# Create EMNIST model
print("Creating EMNIST model...")
ensemble.create_new_model(dataset_type='emnist')

# Print final model counts
counts = ensemble.get_model_counts()
print(f"\nFinal model counts:")
print(f"MNIST models: {counts['mnist']}")
print(f"EMNIST models: {counts['emnist']}")