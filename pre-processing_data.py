import os
import csv

def extract_labels_from_filenames(folder_path, output_csv):
    data = []
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith((".jpg",".png",".JPG",".PNG")):
            # Extract the label from the filename
            label = filename.split('.')[0]

            if "_" in label:
                label = label.split("_")[0]

            # Append the filename and label to the data list
            image_path = os.path.join(folder_path, filename)
            data.append((i, image_path, label))


    # Write the data to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('id', 'image_path', 'label'))
        writer.writerows(data)

    print(f"Labels extracted and saved to {output_csv}")


folder_path = 'train_data'
output_csv = 'train.csv'
extract_labels_from_filenames(folder_path, output_csv)


#also extract labels from test folder
folder_path = 'test_data'
output_csv = 'test.csv'
extract_labels_from_filenames(folder_path, output_csv)  


