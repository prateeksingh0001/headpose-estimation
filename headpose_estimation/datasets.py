# Datasets.py

import os
import cv2
import csv
import sys
import glob
import dlib
import numpy as np
import scipy.io as sio

from PIL import Image, ImageFilter
from math import cos, sin


class BIWI:
    def __init__(self, data_dir, face_model,
                 image_mode='RGB', fc_start=0):
        self.data_dir = data_dir
        self.images = []
        self.angle_values = []
        self.image_mode = image_mode
        self.frame_count_start = fc_start
        self.model_path = face_model
        self.face_detector = dlib.cnn_face_detection_model_v1(self.model_path)

    def get_euler_angles(self, R):

        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

        return [roll, yaw, pitch]

    def face_from_image(self, img):
        img = np.asarray(img)
        img.setflags(write=True)
        dets = self.face_detector(img)

        for idx, det in enumerate(dets):
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()

            return img[y_min - 60:y_max + 60, x_min - 60:x_max + 60]

    def get_rot_translation_matrices(self, file_path):
        R = []
        with open(file_path, 'r') as pose_annot:
            for line in pose_annot:
                line = line.strip('\n').split(' ')
                l = []
                if line[0] != '':
                    for nb in line:
                        if nb == '':
                            continue
                        l.append(float(nb))
                    R.append(l)
        T = R[3][:]
        R = np.transpose(R[:3][:])
        print(R)
        return R, T

    def prepare_dataset(self):

        print('Preparing dataset...... ')
        sub_folders = glob.glob(self.data_dir + '/*/')
        for folder in sub_folders:
            files = glob.glob(folder + '*.png')

            for file in files:
                print(file)
                file_name = file.split('_rgb')[0]

                img = Image.open(file)
                img = img.convert(self.image_mode)

                R, T = self.get_rot_translation_matrices((file_name + '_pose.txt'))

                self.images.append(self.face_from_image(img))
                self.angle_values.append(self.get_euler_angles(R))

    def get_dataset(self):
        self.prepare_dataset()

        for frame in self.images:
            cv2.imwrite(os.path.join('Images', str(self.frame_count_start) + '.jpg'), frame)
            self.frame_count_start += 1

        with open('Images/angle_values.csv', 'a', newline='') as csvfile:
            angle_writer = csv.writer(csvfile, delimiter='\t')
            for value in self.angle_values:
                angle_writer.writerow(value)

        return self.frame_count_start


class GI4E:
    def __init__(self, data_dir, face_model, fc_start=0):
        self.face_model = face_model
        self.face_detector = dlib.cnn_face_detection_model_v1(self.face_model)
        self.images = []
        self.angle_values = []
        self.data_dir = data_dir
        self.frame_count_start = fc_start

    def get_faces(self, video):
        faces = []
        vid = cv2.VideoCapture(video)

        for _ in range(300):
            success, image = vid.read()

            dets = self.face_detector(image)

            for idx, det in enumerate(dets):
                x_min = det.rect.left()
                y_min = det.rect.top()
                x_max = det.rect.right()
                y_max = det.rect.bottom()

                image = image[y_min - 60:y_max + 60, x_min - 60:x_max + 60]
                cv2.imwrite(os.path.join('Images', str(self.frame_count_start) + '.jpg'), image)
                self.frame_count_start += 1
                break

    def prepare_dataset(self):
        subfolders = glob.glob(self.data_dir + '/*/')
        for folder in subfolders:
            videos = sorted(glob.glob(folder + '*.mp4'))
            gt_files = sorted(glob.glob(folder + '*_groundtruth3D.txt'))

            for video in videos:
                print(video)
                self.get_faces(video)

            for file in gt_files:
                with open(file, 'r') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter='\t')
                    for row in csvreader:
                        self.angle_values.append([float(row[3]), float(row[4]), float(row[5])])

    def get_dataset(self):
        self.prepare_dataset()

        with open('Images/angle_values.csv', 'a', newline='') as csvfile:
            angle_writer = csv.writer(csvfile, delimiter='\t')
            for value in self.angle_values:
                angle_writer.writerow(value)

        return self.frame_count_start


class AFW200:
    def __init__(self, data_dir, face_model, image_mode='RGB', fc_start=0):
        self.face_model = face_model
        self.face_detector = dlib.cnn_face_detection_model_v1(self.face_model)
        self.images = []
        self.angle_values = []
        self.data_dir = data_dir
        self.image_mode = image_mode
        self.frame_count_start = fc_start

    def get_ypr_from_mat(self, mat_path):
        mat = sio.loadmat(mat_path)
        pre_pose_params = mat['Pose_Para'][0]
        pose_params = [float(pre_pose_params[2]) * (180 / np.pi), float(pre_pose_params[0]) * (180 / np.pi),
                       float(pre_pose_params[1]) * (180 / np.pi)]
        return pose_params

    def face_from_image(self, img):
        img = np.asarray(img)
        img.setflags(write=True)
        dets = self.face_detector(img)

        for idx, det in enumerate(dets):
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()

            return img[y_min - 60:y_max + 60, x_min - 60:x_max + 60]

    def prepare_dataset(self):
        images = sorted(glob.glob(self.data_dir + '/*.jpg'))
        gt_files = sorted(glob.glob(self.data_dir + '/*.mat'))

        for image in images:
            print(image)
            img = Image.open(image)
            img = img.convert(self.image_mode)
            self.images.append(self.face_from_image(img))

        for file in gt_files:
            self.angle_values.append(self.get_ypr_from_mat(file))

    def get_dataset(self):
        self.prepare_dataset()

        for frame in self.images:
            cv2.imwrite(os.path.join('Images', str(self.frame_count_start) + '.jpg'), frame)
            self.frame_count_start += 1

        with open('Images/angle_values.csv', 'a', newline='') as csvfile:
            angle_writer = csv.writer(csvfile, delimiter='\t')
            for value in self.angle_values:
                angle_writer.writerow(value)

        return self.frame_count_start


if __name__ == "__main__":

    BIWI_dataset = BIWI(data_dir='hpdb', face_model='mmod_human_face_detector.dat',
                        image_mode='RGB')
    frame_count = BIWI_dataset.get_dataset()

    print('BIWI image count = ', frame_count)

    GI4E_dataset = GI4E(data_dir='Head_Pose_Database_UPNA',
                        face_model='mmod_human_face_detector.dat',
                        fc_start=frame_count)
    frame_count = GI4E_dataset.get_dataset()

    print('GI4E image count', frame_count)

    AFW200_dataset = AFW200(data_dir='AFW', face_model='mmod_human_face_detector.dat',
                            image_mode='RGB', fc_start=frame_count)
    frame_count = AFW200_dataset.get_dataset()

    print('Total images = %d' % frame_count)

    bad_images = 0

    images = glob.glob('Images/*.jpg')

    for img in images:
        if cv2.imread(img, 0) is None:
            bad_images += 1
            os.remove(img)

    print("Total Images:", frame_count)
    print("Corrupt Images:", bad_images)
    print("Total good images:", frame_count - bad_images)
