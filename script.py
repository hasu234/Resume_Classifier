import sys
import os
import csv
from sklearn.preprocessing import LabelEncoder
import joblib
from processData.utils import extract_text_from_pdf, cleanText, transformText


if len(sys.argv) != 2:
    sys.exit(1)

dataset_folder = sys.argv[1]

# Create the 'Output' folder
output_folder = "Output"
os.makedirs(output_folder, exist_ok=True)

output_csv = "categorized_resumes.csv"

class_labels = ["ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR", "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER"]
# Instantiate and fit a LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)

# Load the model from the file
model = joblib.load('./model_weight/RandomForest.pkl')

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["filename", "category"])

        for pdf_filename in os.listdir(dataset_folder):
            print("processing:", pdf_filename)
            pdf_path = os.path.join(dataset_folder, pdf_filename)
            text = extract_text_from_pdf(pdf_path)
            processed_text = cleanText(text)
            transformed_text = transformText(processed_text)

            # Use the loaded model to make predictions
            predicted_labels_encoded  = model.predict(transformed_text)

            # Map integer labels back to original class names
            predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
            predicted_labels = predicted_labels.item()

            # writing the filename and label to csv file
            csv_writer.writerow([pdf_filename, predicted_labels])

            # Create a folder with the predicted label under 'Output' if it doesn't exist
            predicted_folder = os.path.join(output_folder, predicted_labels)
            os.makedirs(predicted_folder, exist_ok=True)

            # Move the image to the predicted label folder
            new_path = os.path.join(predicted_folder, pdf_filename)
            os.rename(pdf_path, new_path)