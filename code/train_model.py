import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_path = "C:/Users/NAGABHOOSHAN/OneDrive/Desktop/EMOTIONIQ/dataset/train"
test_path  = "C:/Users/NAGABHOOSHAN/OneDrive/Desktop/EMOTIONIQ/dataset/test"
model_path = "C:/Users/NAGABHOOSHAN/OneDrive/Desktop/EMOTIONIQ/models/emotion_model.keras"
classes_path = "C:/Users/NAGABHOOSHAN/OneDrive/Desktop/EMOTIONIQ/models/class_indices.json"

# Image data generators
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_path,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

test_gen = datagen.flow_from_directory(
    test_path,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

# Print classes
print("Classes found:", train_gen.class_indices)
print("Number of classes:", train_gen.num_classes)

# Save class indices as JSON
with open(classes_path, "w") as f:
    json.dump(train_gen.class_indices, f)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=test_gen
)

# Save model
model.save(model_path)
print(f"✅ Model saved at: {model_path}")
print(f"✅ Classes saved at: {classes_path}")
