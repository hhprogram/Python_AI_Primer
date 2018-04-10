#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Harrison L
# DATE CREATED: 4/10/2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Main program function defined below
def main():
    # TODO HL-DONE: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time.clock()
    
    # TODO HL-DONE: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()
    
    # TODO HL-DONE: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels()

    # TODO HL-DONE: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images()
    
    # TODO HL-DONE: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog()

    # TODO HL-DONE: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats()

    # TODO HL-DONE: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results()

    # TODO HL-DONE: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time.clock()

    # TODO HL-DONE: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", tot_time)



# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='Directory of image file')
    parser.add_argument('--arch', help='CNN model architecture label to use for classification of image file')
    parser.add_argument('--dogfile', help='txt file that contains labels of dogs')
    return parser.parse_args()

def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these label as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    # list the files within the images directory
    dog_pictures = listdir(image_dir)
    pet_labels = {}
    # loop through each image file which should have the dog breed labels within their file names
    for dog in dog_pictures:
        # making sure not to look at any .Something files as those won't be image files
        if dog[0] != ".":
            # file names have underscores instead of spaces
            dog_label = dog.split("_")
            final_dog_label = ""
            for word in dog_label:
                # for uniqueness files have numbers at the end and we don't want those to be added to the dog label strings
                if word.isalpha():
                    final_dog_label.join(word + " ")
            final_dog_label = final_dog_label.strip() #take out the extra blank space at the end
            pet_labels[dog] = final_dog_label.lower() # currently is duplicate filenames then just overrides 
    return pet_labels



def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    results = {}
    # loop through each image in the IMAGE_DIR
    for image in listdir(images_dir):
        # get the pet label determined by the image filename (see get_pet_labels function)
        pet_label = petlabel_dic[image]
        # create the full path to the image file
        image_path = image_dir + image
        # classify the image using the model, leveraging the prebuilt function classifer
        classified_label = classifier(image_path, model)
        # call split on CLASSIFIED_LABEL as per the documentation in test_classifier.py it's possible to have multiple 
        # words to describe one label. Thus get them into a list and then check if one of them matches PET_LABEL
        classified_labels = classified_label.split(",")
        match = 0
        for label in classified_labels:
            if pet_label.lower() == label.lower():
                match = 1
                break
        results[image] = [pet_label, classified_label, match]
    return results



def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """           
    # read in the text file DOGSFILE. Then since each line is a seperate label then use readlines() method 
    # to make each line in txt file its own element in a list
    dognames = open(dogsfile, 'r').readlines()
    # clean up the labels and take out the trailing new line characters
    dognames = [label.rstrip() for label in dognames]
    # loop through each image file
    for image in results_dic:
        # denotes whether or not the image is a dog (will go in indx 3)
        is_dog = 0
        # denotes whether or not the image is classified as a dog by the model (will go in indx 4)
        classifier_is_dog = 0
        # the way we are checking if this image is a dog is if the label is in the dognames label. Assuming
        # that if it was a dog it would be in dognames and if not then it is not a dog
        if results_dic[image][0] in dognames:
            is_dog = 1
        if results_dic[image][1] in dognames:
            classifier_is_dog = 1
        # making sure the length is only 3, if so append if not then just insert in an index. Assumes if it list
        # has length greater than 3 than we somehow inserted both index 3 and 4 already previously. Assumes no list
        # with length 4 possible
        if len(results_dic[image]) > 3:
            results_dic[image][3] = is_dog
            results_dic[image][4] = classifier_is_dog
        else:
            results_dic[image].append(is_dog)
            results_dic[image].append(classifier_is_dog)
    return

# putting the key names for the result_stats dict out here so both calculates_results_stats and print_results can refer to them
is_dog_key_pct = "pct_is_dog"
is_dog_key_ct = "n_is_dog"
match_dog_key_pct = "pct_match_dog"
match_dog_key_ct = "n_match_dog"

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    results_stats = {}
    # counter to see how many animals we correctly labelled as is dog or isn't dog
    is_dog_correct = 0
    # counter to see how many labels we matched with the model (match to specific breed name)
    match_dog = 0
    for image in results_dic:
        is_dog_correct += results_dic[image][3] == results_dic[image][4]
        match_dog += results_dic[image][2]
    is_dog_pct = is_dog_correct / len(results_dic)
    match_dog_pct = match_dog / len(results_dic)
    results_stats[is_dog_key_ct] = is_dog_correct
    results_stats[is_dog_key_pct] = is_dog_pct
    results_stats[match_dog_key_pct] = match_dog_pct
    results_stats[match_dog_key_ct] = match_dog
    return results_stats


def print_results(results_dic, results_stats, model, print_incorrect_dogs, print_incorrect_breed):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    print("Model used ", model,"\n")
    print("---")
    print("Percent correct classifying as dog: ", round(results_stats[is_dog_key_pct]*100,2), "%")
    print("Number correct classifying as dog: ", results_stats[is_dog_key_ct])
    print("Percent correctly classified dog breed: ", round(results_stats[match_dog_key_pct]*100,2), "%")
    print("Number correctly classified dog breed: ", results_stats[match_dog_key_ct])
    print("---")
    if print_incorrect_dogs:
        print("List of non-dog animals incorrectly classified as dogs:\n")
        non_dogs = [results_dic[image][0] for image in results_dic if results_dic[image][3] == 0 and results_dic[image][4] == 1]
        print(non_dogs)
        print("---")
    if print_incorrect_breed
        print("List of incorrect breed classifications (first element is actual breed, 2nd element is classified breed):\n")
        wrong_breeds = [(results_dic[image][0], results_dic[image][1])  for image in results_dic if results_dic[image][2] == 0]
        print(wrong_breeds)
        print("---")                
                
# Call to main function to run the program
if __name__ == "__main__":
    main()
