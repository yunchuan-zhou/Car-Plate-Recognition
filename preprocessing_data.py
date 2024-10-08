import os
import csv

def extract_labels_from_filenames(folder_path, output_csv):
    data = []


    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            # Extract the label from the filename
            label = filename.split('.')[0]

            # Append the filename and label to the data list
            data.append([filename, label])


    # Write the data to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Label'])
        writer.writerows(data)

    print(f"Labels extracted and saved to {output_csv}")


folder_path = 'dataset_final/train'
output_csv = 'labels_train.csv'
extract_labels_from_filenames(folder_path, output_csv)

# also extract labels from val folder
folder_path = 'dataset_final/val'
output_csv = 'labels_val.csv'
extract_labels_from_filenames(folder_path, output_csv)
#also extract labels from test folder
folder_path = 'dataset_final/test'
output_csv = 'labels_test.csv'
extract_labels_from_filenames(folder_path, output_csv)  



