#@title Utility to zip and download a directory
#@markdown Use this method to zip and download a directory. For ex. a TB logs 
#@markdown directory or a checkpoint(s) directory.

from google.colab import files
import os

dir_to_zip = '/content/CervicalCancerSplitDataset' #@param {type: "string"}
output_filename = 'CervicalCancerSplitDataset.zip' #@param {type: "string"}
delete_dir_after_download = "No"  #@param ['Yes', 'No']

os.system( "zip -r {} {}".format( output_filename , dir_to_zip ) )

if delete_dir_after_download == "Yes":
    os.system( "rm -r {}".format( dir_to_zip ) )

files.download( output_filename )
