from neural_network import create_synthetic_dataset, calculate_weights, binary_classifier, calculate_accuracy, train_test_split

# Step 1: Run synthetic dataset generator
print("Generating synthetic dataset...")
dataset = create_synthetic_dataset(100)
print(f"Dataset created with {len(dataset)} samples.")

# Step 2: Split into train and test sets
print("\nSplitting dataset into train and test sets...")
train_set, test_set = train_test_split(dataset, test_size=0.2)
print(f"Train set: {len(train_set)} samples, Test set: {len(test_set)} samples.")

# Extract labels for accuracy calculation
train_labels = [sample[1] for sample in train_set]
test_labels = [sample[1] for sample in test_set]

# Step 3: Run weight-calculation function twice, once per activation function
print("\nCalculating weights for step activation...")
weights_step, bias_step = calculate_weights(train_set, 'step')
print(f"Weights: {weights_step}, Bias: {bias_step}")

print("\nCalculating weights for sigmoid activation...")
weights_sigmoid, bias_sigmoid = calculate_weights(train_set, 'sigmoid')
print(f"Weights: {weights_sigmoid}, Bias: {bias_sigmoid}")

# Step 4: Run the classifier twice on test set, once per weight set
print("\nRunning classifier with step weights on test set...")
predictions_step = binary_classifier(test_set, weights_step, bias_step, 'step')
accuracy_step = calculate_accuracy(predictions_step, test_labels)
print(f"Predictions (first 10): {predictions_step[:10]}")
print(f"Test Accuracy: {accuracy_step}")

print("\nRunning classifier with sigmoid weights on test set...")
predictions_sigmoid = binary_classifier(test_set, weights_sigmoid, bias_sigmoid, 'sigmoid')
accuracy_sigmoid = calculate_accuracy(predictions_sigmoid, test_labels)
print(f"Predictions (first 10): {predictions_sigmoid[:10]}")
print(f"Test Accuracy: {accuracy_sigmoid}")