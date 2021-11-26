import librosa
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt

class SoundData:
  def loadFromFile(path, n_fft=512):
    # Load from file
    new_raw, new_fr = librosa.load(path)
    # Normalize sound amplitude
    new_raw /= np.sqrt(new_raw.dot(new_raw))
    # Create new SoundData instance
    sd = SoundData(new_fr, n_fft)
    sd.raw = new_raw
    sd.process()
    return sd
    
  def __init__(self, frame_rate, n_fft=512):
    self.fr = frame_rate
    self.raw = None
    self.stft = None
    self.amp = None
    self.n_fft = n_fft
  
  def process(self):
    # Do fourier transform
    self.stft = librosa.core.stft(self.raw, n_fft=self.n_fft)
    self.amp = np.abs(self.stft)

  def play(self):
    ipd.display(ipd.Audio(self.raw ,rate=self.fr))
  
  def crop_by_raw_index(self, start=0, end=None):
    self.raw = self.raw[start:end]
    self.process()
  
  def crop_by_stft_index(self, start=0, end=None):
    raw_stft_length_rate = round(self.raw.shape[0] / self.amp.shape[1])
    raw_start_index = start*raw_stft_length_rate
    raw_end_index = end*raw_stft_length_rate if end is not None else None
    self.raw = self.raw[raw_start_index:raw_end_index]
    self.process()
  
  def sample_by_raw_index(self, start=0, end=None):
    sample = SoundData(self.fr, self.n_fft)
    sample.raw = self.raw[start:end]
    sample.process()
    return sample
  
  def sample_by_stft_index(self, start=0, end=None):
    sample = SoundData(self.fr, self.n_fft)
    raw_stft_length_rate = round(self.raw.shape[0] / self.amp.shape[1])
    raw_start_index = start*raw_stft_length_rate
    raw_end_index = end*raw_stft_length_rate if end is not None else None
    sample.raw = self.raw[raw_start_index:raw_end_index]
    sample.process()
    return sample
  
  def plot(self, start=0, length=None):
    if length is None: length = self.fr
    plt.plot(self.raw[start:start+length])



def synchronize_by_amp(SD1, SD2, search_range=400, sample_length=400, display_graph=True, no_crop=False, no_postprocessing=False):
  # search_range: range to search for matching offset (from -search_length to search_length)
  # sample_length: length of sample data to compare
  # returns None, updates synchronized SD1, SD2
  if SD1.amp is None: SD1.process()
  if SD2.amp is None: SD2.process()
  totalAmp1 = np.average(SD1.amp, axis=0)
  totalAmp2 = np.average(SD2.amp, axis=0)
  similarity_list = []
  # positive offset = SD2 is delayed(should be cropped at the front)
  for offset in range(-search_range, search_range):
    offset1, offset2 = max(-offset, 0), max(offset, 0)
    sample1 = totalAmp1[offset1:offset1+sample_length]
    sample2 = totalAmp2[offset2:offset2+sample_length]
    similarity = np.dot(sample1, sample2) / (np.sqrt(sample1.dot(sample1) * sample2.dot(sample2)))
    similarity_list.append(similarity)
  best_offset = np.argmax(similarity_list) - search_range
  
  if not no_crop:
    # Crop at the front
    crop_target = SD1 if best_offset<0 else SD2
    crop_offset = abs(best_offset)
    crop_target.crop_by_stft_index(best_offset, None)

    # Crop at the end
    min_raw_length = min(SD1.raw.shape[0], SD2.raw.shape[0])
    SD1.crop_by_raw_index(0, min_raw_length)
    SD2.crop_by_raw_index(0, min_raw_length)

    if not no_postprocessing:
      # Postprocessing
      SD1.process()
      SD2.process()

  # Display graph
  if display_graph:
    plt.plot(similarity_list)
    plt.show()
  
  return best_offset

def synchronize_by_raw(SD1, SD2, search_range=50, sample_length=800, display_graph=True, no_crop=False, no_postprocessing=False):
  # search_range: range to search for matching offset (from -search_length to search_length)
  # sample_length: length of sample data to compare
  # returns None, updates synchronized SD1, SD2
  similarity_list = []
  # positive offset = SD2 is delayed(should be cropped at the front)
  for offset in range(-search_range, search_range):
    offset1, offset2 = max(-offset, 0), max(offset, 0)
    sample1 = SD1.raw[offset1:offset1+sample_length]
    sample2 = SD2.raw[offset2:offset2+sample_length]
    similarity = np.dot(sample1, sample2) / (np.sqrt(sample1.dot(sample1) * sample2.dot(sample2)))
    similarity_list.append(similarity)
  best_offset = np.argmax(similarity_list) - search_range
  
  if not no_crop:
    # Crop at the front
    crop_target = SD1 if best_offset<0 else SD2
    crop_offset = abs(best_offset)
    crop_target.crop_by_raw_index(crop_offset, None)

    # Crop at the end
    min_raw_length = min(SD1.raw.shape[0], SD2.raw.shape[0])
    SD1.crop_by_raw_index(0, min_raw_length)
    SD2.crop_by_raw_index(0, min_raw_length)

    if not no_postprocessing:
      # Postprocessing
      SD1.process()
      SD2.process()

  # Display graph
  if display_graph:
    plt.plot(similarity_list)
    plt.show()
  
  return best_offset

