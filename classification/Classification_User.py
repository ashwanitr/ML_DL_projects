#Note: The Dataset Folder must be created with 2 folders, train and val. The train and val must contain same number of Class Folders and same names.
Directory=""  #Change This Directory To The Folder That Contains your Dataset Folder
No_Of_Classes= #Change Number Of Classes According To Your Need
Training_Directory=""  #Change This Directory To The Folder That Contains your Train Images
Validation_Directory=""  #Change This Directory To The Folder That Contains your Val Images
Image_Path=""    #Set This To The Val Directory Path
Csv_Name=""    #Set CSV Name To Be Generated
Batch_Size= #64 is usually ideal, however if machine fails try 48 and then 32
Image_Size= #224,244 is usually ideal , but you can try 299,299 as well if architecture supports it
Epochs= #25 is usually ideal
Learning_Rate = #1e-3 is usually ideal, but if you want you can set it lower than that
No_Of_GPUS = #Set as 2 if you are using jediyoda
No_Of_Frozen_Layers= #Set as 0 by default,otherwise try as per your wish
Architecture= #Set as Required. 1 for InceptionResnetV2, 2 for DenseNet121, 3 for ResNet50, 4 for NasNetMobile, 5 for MobileNet, 6 for InceptionV3
