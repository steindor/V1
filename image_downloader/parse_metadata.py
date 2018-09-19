import os
import csv
import json
import glob

METADATA_PATH = 'ISIC-images-metadata/'
METADATA_NAME = 'metadata_w_diagnosis.csv'

def main():
    no_of_rows = 0
    with open(METADATA_NAME, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'benign_malignant',
                         'diagnosis', 'age', 'sex', 'shape_x', 'shape_y'])
        # for metadata_filename in os.listdir(METADATA_PATH):
        for metadata_path in glob.glob('{}*/*.json'.format(METADATA_PATH)):
            metadata_filename = metadata_path.split('/')[-1]
            if not 'json' in metadata_filename:
                continue

            no_of_rows += 1

            # metadata_path = METADATA_PATH #+ metadata_filename
            with open(metadata_path, 'r') as f:
                metadata = json.loads(f.read())

            meta = metadata['meta']
            aquisition = meta['acquisition']
            clinical = meta['clinical']
            if not 'benign_malignant' in clinical \
                    or not 'age_approx' in clinical\
                    or not 'diagnosis' in clinical\
                    or not 'sex' in clinical:
                continue
            writer.writerow([
                metadata['_id'],
                metadata['name'],
                0 if clinical['benign_malignant'] == 'benign' else 1,
                clinical['diagnosis'],
                clinical['age_approx'],
                clinical['sex'],
                aquisition['pixelsX'],
                aquisition['pixelsY']
            ])

        print("done with {} no of rows".format(no_of_rows))


if __name__ == '__main__':
    main()
