import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from datetime import datetime


class ImagePair:
    def __init__(self, azimuth_1, azimuth_2, off_nadir_1, off_nadir_2, month_1, month_2, gsd_1, gsd_2):
        self.azimuth_1 = azimuth_1
        self.azimuth_2 = azimuth_2
        self.off_nadir_1 = off_nadir_1
        self.off_nadir_2 = off_nadir_2
        self.month_1 = month_1
        self.month_2 = month_2
        self.gsd_1 = gsd_1
        self.gsd_2 = gsd_2


class ImagePairPlot:
    def __init__(self, data_file_path, image_pair_file_path, files_chosen_path):
        self.data_file_path = data_file_path
        self.image_pair_file_path = image_pair_file_path
        self.files_chosen_path = files_chosen_path
        # sortedbydate file
        self.gsd_values = []
        self.month_values = []
        self.off_nadir_values = []
        self.azimuth_values = []
        self.order_id = []
        # PairStats File
        self.image1_orderid = []
        self.image2_orderid = []
        self.pair_intersection_angle = []
        self.incidence_angle = []
        self.matching_filename_index_1 = []
        self.matching_filename_index_2 = []
        self.image_pairs = []
        self.use_data_file_for_angles = False
        self.image_names = []

        # Get list of all images used for AOI
        with open(self.files_chosen_path, mode='r') as infile:
            reader = csv.reader(infile)
            column_id = {k: v for v, k in enumerate(next(reader))}
            for rows in reader:
                image_name = rows[column_id['Image filename']][35:55]
                self.image_names.append(image_name)

        # PairStats
        with open(self.image_pair_file_path, mode='r') as infile:
            reader = csv.reader(infile)
            column_id = {k: v for v, k in enumerate(next(reader))}
            for rows in reader:
                # Ignore discarded rows
                try:
                    if rows[column_id['discarded']] == 'yes':
                        continue
                except KeyError:
                    pass

                image1_orderID = (rows[column_id['Image 1 filename']][35:55])
                image2_orderID = (rows[column_id[' Image 2 filename']][35:55])
                self.image1_orderid.append(image1_orderID)
                self.image2_orderid.append(image2_orderID)
                try:
                    self.pair_intersection_angle.append(float(rows[column_id['intersection_angle']]))
                    self.incidence_angle.append(np.round(float(rows[column_id['incidence_angle']]), 1))
                except KeyError:
                    self.use_data_file_for_angles = True

        # sortedbydate
        with open(self.data_file_path, mode='r') as infile:
            reader = csv.reader(infile)
            column_id = {k: v for v, k in enumerate(next(reader))}
            for rows in reader:
                if not (rows[column_id['spectral range']] == 'PAN'):
                    continue
                # Compare used images with data file
                if rows[column_id['filename']][35:55] not in self.image_names:
                    continue
                self.gsd_values.append(float(rows[column_id['mean product gsd']]))
                self.azimuth_values.append(float(rows[column_id['mean satellite azimuth']]))
                self.off_nadir_values.append(90.0-float(rows[column_id['mean satellite elevation']]))
                year = str(int(float(rows[column_id['date']])))[0:4]
                month = str(int(float(rows[column_id['date']])))[4:6]
                self.month_values.append(int(month))
                orderID = rows[column_id['order id']]
                try:
                    indices = [i for i, x in enumerate(self.image1_orderid) if x == orderID]
                    self.matching_filename_index_1.append(indices)
                except ValueError:
                    self.matching_filename_index_1.append(np.nan)
                try:
                    indices = [i for i, x in enumerate(self.image2_orderid) if x == orderID]
                    self.matching_filename_index_2.append(indices)
                except ValueError:
                    self.matching_filename_index_2.append(np.nan)
                self.order_id.append(rows[column_id['order id']])

        # Sort into pairs
        matching_pairs = []
        for i, image_1_indices in enumerate(self.matching_filename_index_1):
            for j, image_2_indices in enumerate(self.matching_filename_index_2):
                check = any(item in self.matching_filename_index_1[i] for item in self.matching_filename_index_2[j])
                if check:
                    matching_pairs.append([i, j])

        for pair in matching_pairs:
            file_1_index = pair[0]
            file_2_index = pair[1]

            az_1 = self.azimuth_values[file_1_index]
            az_2 = self.azimuth_values[file_2_index]
            el_1 = self.off_nadir_values[file_1_index]
            el_2 = self.off_nadir_values[file_2_index]
            m_1 = self.month_values[file_1_index]
            m_2 = self.month_values[file_2_index]
            gsd_1 = self.gsd_values[file_1_index]
            gsd_2 = self.gsd_values[file_2_index]
            image_pair_temp = ImagePair(az_1, az_2, el_1, el_2, m_1, m_2, gsd_1, gsd_2)
            self.image_pairs.append(image_pair_temp)

    def create_plot(self, figNum):
        pair_lines = []
        for i, val in enumerate(self.image_pairs):
            pair_lines.append([[-(val.azimuth_1 / (180/np.pi))+np.pi/2, val.off_nadir_1],
                               [-(val.azimuth_2 / (180/np.pi))+np.pi/2, val.off_nadir_2]])

        radial_label_angle = 0
        gsd = self.gsd_values
        month = self.month_values
        r = self.off_nadir_values
        theta = self.azimuth_values
        theta = [-(x / (180/np.pi))+np.pi/2 for x in theta]

        # Calculate DOP
        pdop, hdop, vdop = self.calculate_dop()

        # Create scatter with data for Month
        plt.figure(figNum, figsize=(17, 5))
        ax = plt.subplot(121, projection='polar')
        # Plot connecting lines
        for i, points in enumerate(pair_lines):
            sc = ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], '-ro', LineWidth=0.3,
                         MarkerSize=0.1)
        # Plot circles
        sc = ax.scatter(theta, r, s=70, c=month, alpha=1, edgecolors='black')

        # Format plot
        lines, labels = plt.thetagrids(range(0, 360, 30),
                                       ('E', '60°', '30°', 'N', '330°', '300°', 'W', '240°', '210°', 'S', '150°',
                                        '120°'))
        for label, angle in zip(labels, range(0, 360, 30)):
            label.set_rotation(90 - angle)
        # Add axis labels
        ax.text((5 * np.pi) / 6, 100, 'Azimuth (deg)', fontsize=10)
        ax.text((np.pi) / 11, 70, 'Off-nadir (deg)', fontsize=10)
        # Add color bar
        cbar = plt.colorbar(sc, pad=0.25)
        ax.set_rmax(60)
        ax.set_rlabel_position(radial_label_angle)  # get radial labels away from plotted line
        ax.grid(True)
        ax.set_title("Month of Year", y=-0.15, fontsize=18)

        # Create scatter with data for GSD
        ax2 = plt.subplot(122, projection='polar')
        # Plot connecting lines
        for i, points in enumerate(pair_lines):
            sc = ax2.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], '-ro', LineWidth=0.3,
                          MarkerSize=0.1)
        # Plot Circles
        sc = ax2.scatter(theta, r, s=70, c=gsd, alpha=1, edgecolors='black')
        # Format plot
        lines, labels = plt.thetagrids(range(0, 360, 30),
                                       ('E', '60°', '30°', 'N', '330°', '300°', 'W', '240°', '210°', 'S', '150°',
                                        '120°'))
        for label, angle in zip(labels, range(0, 360, 30)):
            label.set_rotation(90 - angle)
            # Add axis labels
        ax2.text((5 * np.pi) / 6, 100, 'Azimuth (deg)', fontsize=10)
        ax2.text(np.pi / 11, 70, 'Off-nadir (deg)', fontsize=10)
        # Add color bar
        cbar = plt.colorbar(sc, pad=0.25)
        ax2.set_rmax(60)
        ax2.set_rlabel_position(radial_label_angle)  # get radial labels away from plotted line
        ax2.grid(True)
        ax2.set_title("Ground Sample Distance (GSD)", y=-0.15, fontsize=18)
        # Plot DOP calculations
        ax.text((7 * np.pi) / 6, 110,
                'PDOP: ' + '%.3f' % pdop + '\n'
                + 'HDOP: ' + '%.3f' % hdop + '\n'
                + 'VDOP: ' + '%.3f' % vdop, fontsize=10)

        return plt

    def calculate_dop(self, altitude=643738.0, num_satellites=10):
        # Define platform coordinates.
        # Approximate with one position per platform for entire block
        ground_ranges = altitude * np.tan(np.array(self.off_nadir_values) * np.pi / 180)
        cx = np.ones((num_satellites, 1))
        cy = np.ones((num_satellites, 1))
        cz = np.ones((num_satellites, 1))
        for i in range(0, num_satellites):
            angle_radians = self.azimuth_values[i] * np.pi/180
            rotation = np.array([[np.cos(angle_radians), -np.sin(angle_radians)], [np.sin(angle_radians),
                                                                                   np.cos(angle_radians)]])
            cp = np.matmul(rotation, np.array([ground_ranges[i], 0]))
            cx[i] = cp[0]
            cy[i] = cp[1]
            cz[i] = altitude
        # Compute Jacobian
        G = np.zeros([num_satellites, 3])
        for p in range(0, num_satellites):
            range_n = np.sqrt(cx[p] ** 2 + cy[p] ** 2 + cz[p] ** 2)
            G[p, 0] = cx[p]/range_n
            G[p, 1] = cy[p]/range_n
            G[p, 2] = cz[p]/range_n
        # Computer PDOP for this point
        Q = np.linalg.inv(np.matmul(np.transpose(G), G))
        pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
        hdop = np.sqrt(Q[0, 0] + Q[1, 1])
        vdop = np.sqrt(Q[2, 2])

        return pdop, hdop, vdop


def main():
    print("Debug")


if __name__ == "__main__":
    main()
