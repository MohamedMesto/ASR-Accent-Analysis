#####################################################################################
########### set the Data files paths on Conda Notebook on Ubuntu #################### 
import os
### 
os.chdir('/home/mmm2050/QU_DFKI_Thesis/Experimentation/ASR_Accent_Analysis_De')
main_path=os.getcwd()

if not os.path.exists(main_path+'/Data_results'):
  os.makedirs(main_path+'/Data_results')

if not os.path.exists(main_path+'/Figures_results'):
  os.makedirs(main_path+'/Figures_results')

Data_path=main_path+'/Data/'
Data_results_path=main_path+'/Data_results/'
json_file_path=Data_path+'results.json'
validated_tsv_path=Data_path+'validated.tsv'

validated_en_tsv_path=main_path+'/DeepSpeech/data/validated.tsv'


#####################################################################################
########### set the Data files paths on Colab Notebook###############################
# # Import the dataset file by method1 
# # from google.colab import files
# # uploaded = files.upload()
# if not os.path.exists(Data_path'):
#   os.makedirs(Data_path')
# ! cp /content/drive/MyDrive/QU-DFKI-Thesis-ASR/Experimentation/cv-corpus04072022/de/validated.tsv /content/Data/
# # copy the expermintations files to deal with them
# ! cp /content/drive/MyDrive/QU-DFKI-Thesis-ASR/Experimentation/ASR-Accent-Analysis-De/Data/*.* /content/Data/
# # copy the expermintations files from Mozilla Commen Voice v 10 to deal with them

# import shutil
# shutil.rmtree('/content/audio', ignore_errors=True)
 
os.getcwd()


#@title create the Sentences text files according to their Audio files
# and store them in main_path+'/DeepSpeech/data/text'
import csv
import os
import pandas as pd
import re
audio_file_id_list=[]
audio_file_id_txt=[]

# set the paths for audio files and text files
audio_path = main_path+"/audioFiles/"
text_path = main_path+"/DeepSpeech/data/text/"


# create text folder inside DeepSpeech
if not os.path.exists(main_path+'/DeepSpeech/data/text'):
    os.makedirs(main_path+'/DeepSpeech/data/text')

#Open the TSV file and read its contents
tsv_en_data = pd.read_csv( validated_en_tsv_path, sep='\t')
tsv_en_data.drop(tsv_en_data[(tsv_en_data['path'].isna())].index, inplace=True)


for row, content in tsv_en_data.iterrows():
    # Extract the sentence and path from the current row
    sentence = content['sentence']
    audio_file_id = re.split(r'[.]',content['path'])[0]  

    # Create the text file name based on the path column
    text_file_name = os.path.splitext(os.path.basename(audio_file_id))[0] + '.txt'

    # Create the full path to the text file
    text_file_path = os.path.join('DeepSpeech', 'data', 'text', text_file_name)
    #or
    #  text_file_path = text_path
    # Write the sentence to the text file
    with open(text_file_path, 'w') as text_file:
        text_file.write(sentence)

    '''Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
    a comma. Each new line is a different sample. Example below:
    /path/to/audio.wav,/path/to/audio.txt'''


##################################################################################################
# # create the Manifest CSV file from the validated.tsv file contents (Audio files ID , sentences) 
##################################################################################################
#     audio_file_id_list.append(main_path+'/audioFiles/'+ audio_file_id+'.wav')
#     audio_file_id_txt.append(main_path+'/DeepSpeech/data/text/'+audio_file_id+'.txt')

# # Create a new CSV file and write the lists to it
# with open(main_path+'/DeepSpeech/data/test_manifest.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # writer.writerow(['Fruit', 'Number'])
#     for i in range(len(audio_file_id_list)):
#         writer.writerow([audio_file_id_list[i], audio_file_id_txt[i]])




#####################################################################################
# get a list of all the audio,text files in the audio path , text path ##############
# create the Manifest CSV file from them     ########################################      
#####################################################################################
# get a list of all the audio files in the audio path
audio_files = [f for f in os.listdir(audio_path) if f.endswith(".wav")]

# create an empty list to store the audio and text file paths
file_paths = []

# loop through the audio files and check if there is a matching text file
for audio_file in audio_files:
    audio_name = os.path.splitext(audio_file)[0] # get the name of the audio file without the extension
    text_file = text_path + audio_name + ".txt" # create the path for the matching text file
    
    # check if the text file exists
    if os.path.isfile(text_file):
        file_paths.append([audio_path + audio_file, text_file])

# create a CSV file with the list of file paths
with open(main_path+'/DeepSpeech/data/test_manifest.csv', "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # write a columns label
    # writer.writerow(["Audio File", "Text File"])
    # write a columns contents
    writer.writerows(file_paths)




