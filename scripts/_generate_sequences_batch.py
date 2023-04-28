import os
import shutil
import random

def get_file_paths_sorted(folder):
   file_paths = os.listdir(folder)
   file_paths = sorted(file_paths, key=lambda i: os.path.splitext(os.path.basename(i))[0])
   return file_paths

def filter_paths_by_name(string, file_paths):
   wanted_images = []
   for file_path in file_paths:
      if string in file_path:
         wanted_images.append(file_path)
   return wanted_images

def extract_random_image_sequences(file_paths, number_of_sequences, sequence_size): 
   num_imgs = len(file_paths)
   num_selected_images = number_of_sequences * sequence_size
   max_distance = int((num_imgs - num_selected_images)/number_of_sequences)
   
   sequences = []
   current_index = 0
   for _ in range(number_of_sequences):
      step = random.randint(1, max_distance)
      current_index += step
      if (current_index + sequence_size) > num_imgs:
         break
      sequence = file_paths[current_index:current_index+sequence_size]
      current_index += sequence_size
      current_index += max_distance - step
      sequences.append(sequence)

   return sequences

def move_sequences_separately(sequences, source_folder, sequence_base_name, side_camera_suffix):
   sequence_id = 1
   for sequence in sequences:
      sequence_folder = sequence_base_name + '_' + side_camera_suffix + "_" + str(sequence_id)
      os.mkdir(sequence_folder)
      for image in sequence:
         shutil.copy(source_folder + '/' + image, sequence_folder + '/' + image)

      sequence_id += 1

def copy_left_sequences_batch(file_paths, source_folder, sequence_base_name):
   left_images = filter_paths_by_name('left', file_paths)
   left_sequences = extract_random_image_sequences(left_images, 5, 200)
   move_sequences_separately(left_sequences, source_folder, sequence_base_name, 'left')

def copy_right_sequences_batch(file_paths, source_folder, sequence_base_name):
   right_images = filter_paths_by_name('right', file_paths)
   right_sequences = extract_random_image_sequences(right_images, 5, 200)
   move_sequences_separately(right_sequences, source_folder, sequence_base_name, 'right')

if __name__ == '__main__':
   source_folder = '/media/ltoschi/One Touch/SLAM dataset dump/20220815_cornfield/ts_2022_08_15_11h20m26s_two_random_extracted'
   sequence_base_name = 'ts_2022_08_15_11h20m26s_two_random_seq'
   file_paths = get_file_paths_sorted(source_folder)
   copy_left_sequences_batch(file_paths, source_folder, sequence_base_name)
   copy_right_sequences_batch(file_paths, source_folder, sequence_base_name)
   
   